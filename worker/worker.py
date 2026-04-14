import os
import ujson
import requests
import redis
import shutil
import boto3
import time
from celery import Celery
from openai import OpenAI
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# --- INITIALIZATION ---
# Fixed: Explicit credentials and region to avoid boto3 lookup errors
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("S3_REGION", "us-east-1")
)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
API_BASE_URL = os.getenv("API_URL", "http://fastapi-api:9000")

client = OpenAI(
    base_url='http://13.217.184.147:80/v1', 
    api_key='neuralninekey',
    timeout=60.0 
)

# New High-Throughput Redis Pool
redis_pool = redis.ConnectionPool(
    host=os.getenv('REDIS_HOST', 'redis'), 
    port=6379, 
    decode_responses=True,
    max_connections=50
)
db = redis.Redis(connection_pool=redis_pool)
NVME_PATH = os.getenv('NVME_PATH', '/app/data')

app = Celery('worker', broker=os.getenv('CELERY_BROKER_URL'))
app.conf.update(
    task_serializer='json',
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=500
)

session = requests.Session()

@app.task(bind=True, name='process_jts_task', max_retries=5)
def process_jts_task(self, job_id, transcript_id, scorecard_id):
    task_id = f"j{job_id}_t{transcript_id}_s{scorecard_id}"
    job_dir = os.path.join(NVME_PATH, f"job_{job_id}")
    task_file = os.path.join(job_dir, f"{task_id}.json")
    result_file = os.path.join(job_dir, f"res_{task_id}.json")

    try:
        if not os.path.exists(task_file):
            return {"status": "error", "detail": f"file_not_found: {task_id}"}

        with open(task_file, 'r', encoding='utf-8') as f:
            task_data = ujson.load(f)

        # FIX: Ensure we handle your nested JSON structure correctly
        q_list_wrapper = task_data.get('questions', [])
        if not q_list_wrapper: return {"status": "error", "detail": "no_questions"}
        
        # Pulling from the scorecard list inside the wrapper
        q_list = q_list_wrapper[0].get('scorecard', []) if 'scorecard' in q_list_wrapper[0] else q_list_wrapper

        # --- PHASE 1: PREP SCHEMA ---
        properties = {}
        q_ids = []
        clean_qs = []
        for q in q_list:
            qid = str(q.get('id'))
            q_ids.append(qid)
            raw_opts = q.get('answer') or q.get('options') or ["Yes", "No", "NA"]
            cleaned_opts = [str(o).strip() for o in raw_opts]
            properties[qid] = {"type": "string", "enum": cleaned_opts}
            clean_qs.append(f"QID {qid}: {q.get('question') or q.get('text')} | CHOICES: {', '.join(cleaned_opts)}")

        schema = {"type": "object", "properties": properties, "required": q_ids, "additionalProperties": False}
        formatted_qs = "\n".join(clean_qs)
        
        # --- PHASE 2: LLM REQUEST ---
        response = client.chat.completions.create(
            model='Qwen/Qwen3-4B-Instruct-2507',
            messages=[
                {"role": "system", "content": "Extract MCQ answers into JSON. Map QIDs to selected options based on the transcript."},
                {"role": "user", "content": f"TRANSCRIPT:\n{task_data.get('transcript', '')}\n\nQUESTIONS:\n{formatted_qs}"}
            ],
            temperature=0,
            extra_body={"guided_json": schema}
        )
          
        raw_content = response.choices[0].message.content
        perfect_answers = ujson.loads(raw_content)

        # --- PHASE 3: ATOMIC SAVE ---
        with open(result_file, 'w', encoding='utf-8') as f:
            ujson.dump({
                "transcript_id": transcript_id, 
                "scorecard_id": scorecard_id, 
                "answers": perfect_answers,
                "task_id": task_id
            }, f)

        # Clean up input file to save NVME space
        if os.path.exists(task_file): os.remove(task_file)

        # --- PHASE 4: LUA STEERED TRIGGER ---
        lua_script = """
        local completed = redis.call('INCR', KEYS[1])
        local total = redis.call('HGET', KEYS[2], 'total')
        if not total then total = redis.call('GET', KEYS[2] .. ':total') end
        if total and tonumber(completed) >= tonumber(total) then
            return 1
        end
        return 0
        """
        is_complete = db.eval(lua_script, 2, f"job:{job_id}:completed", f"job:{job_id}")

        if is_complete == 1:
            logger.info(f"🎯 Job {job_id} complete. Dispatching Stitcher.")
            app.send_task('stitch_job_results', args=[job_id], queue='stitch_queue', priority=10)

        return {"status": "success", "task": task_id}

    except Exception as e:
        logger.error(f"❌ Error in process_jts_task: {e}")
        raise self.retry(exc=e, countdown=5)


@app.task(name='stitch_job_results', bind=True, max_retries=3)
def stitch_job_results(self, job_id):
    lock_key = f"lock:stitch:{job_id}"
    lock = db.lock(lock_key, timeout=300)
    
    if not lock.acquire(blocking=False):
        return {"status": "skipped"}

    try:
        job_dir = os.path.join(NVME_PATH, f"job_{job_id}")
        file_name = f"job_{job_id}.json"
        output_path = os.path.join(NVME_PATH, file_name)

        job_metadata = db.hgetall(f"job:{job_id}")
        if not job_metadata:
             total_val = db.get(f"job:{job_id}:total")
             total_expected = int(total_val) if total_val else 0
        else:
             total_expected = int(job_metadata.get('total', 0))
        
        if not os.path.exists(job_dir):
            raise self.retry(countdown=10)

        result_files = [f for f in os.listdir(job_dir) if f.startswith("res_")]
        if len(result_files) < total_expected:
            raise self.retry(countdown=10)

        # AGGREGATION
        grouped = {}
        for fname in result_files:
            try:
                with open(os.path.join(job_dir, fname), 'r', encoding='utf-8') as f:
                    data = ujson.load(f)
                    tid = data.get("transcript_id", "unknown")
                    if tid not in grouped:
                        grouped[tid] = {"transcript_id": tid, "scorecards": []}
                    grouped[tid]["scorecards"].append({
                        "scorecard_id": data.get("scorecard_id"), 
                        "answers": data.get("answers")
                    })
            except: continue

        # SAVE & UPLOAD
        with open(output_path, 'w', encoding='utf-8') as f:
            ujson.dump(list(grouped.values()), f, indent=4)

        # Define the S3 path with the folder prefix
        s3_key = f"output/{file_name}" 

        # 1. Upload using the new S3 Key
        s3_client.upload_file(
            output_path, 
            S3_BUCKET_NAME, 
            s3_key,  # Use the key with the 'output/' prefix
            ExtraArgs={'ContentType': 'application/json'}
        )

        # 2. Generate the URL using the exact same S3 Key
        url = s3_client.generate_presigned_url(
            'get_object', 
            Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key}, 
            ExpiresIn=604800
        )

        # DYNAMIC CALLBACK
        target_api = job_metadata.get("callback_url")
        target_token = job_metadata.get("callback_token")

        if target_api:
            try:
                # Clear default headers to avoid mixing Bearer/Token types
                callback_headers = {"Content-Type": "application/json"}
                if target_token:
                    callback_headers["Authorization"] = f"Token {target_token}"
                
                session.post(
                    target_api, 
                    json={"file_path": url}, 
                    headers=callback_headers,
                    timeout=30
                )
                logger.info(f"✅ Callback successful for job {job_id}")
            except Exception as e:
                logger.error(f"❌ Callback Error: {e}")

        # CLEANUP
        shutil.rmtree(job_dir, ignore_errors=True)
        if os.path.exists(output_path): 
            os.remove(output_path)
        
        db.delete(f"job:{job_id}", f"job:{job_id}:completed")
        return {"status": "finished", "job_id": job_id}

    finally:
        if lock.owned(): 
            lock.release()