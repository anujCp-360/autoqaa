import os
import uuid
import re
import ast
import requests
import json
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import ujson
from celery import Celery

# --- CONFIGURATION ---
NVME_PATH = os.getenv("NVME_PATH", "/app/data")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
API_TOKEN = os.getenv("API_TOKEN")

API_BASE_URL = os.getenv("API_URL", "http://fastapi-api:9000")
# --- AWS S3 CREDENTIALS ---


fastapi_app = FastAPI(title="TERAFAB Dual-Input API")
# decode_responses=True is important for consistent string handling with Celery/Lua
db = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
bearer_scheme = HTTPBearer()

celery_app = Celery('worker', broker=f'redis://{REDIS_HOST}:6379/0')
celery_app.conf.update(
    task_queue_max_priority=10,
    task_default_priority=0
)

Path(NVME_PATH).mkdir(parents=True, exist_ok=True)

# --- HELPERS ---
def clean_transcript_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    strip_map = str.maketrans({"|": None, "\r": None, "\n": " ", "\u2026": None, "'": None})
    translated = text.translate(strip_map)
    return " ".join(translated.split()).strip()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def extract_nested_list(text: str):
    bracket_count = 0
    start_idx = text.find('[')
    if start_idx == -1: return None
    for i in range(start_idx, len(text)):
        if text[i] == '[': bracket_count += 1
        elif text[i] == ']':
            bracket_count -= 1
            if bracket_count == 0: return text[start_idx : i + 1]
    return None

# --- MAIN ENDPOINT ---
@fastapi_app.post("/job")
async def submit_distributed_job(request: Request, credentials=Depends(verify_token)):
    try:
        body = await request.json()
        file_path = body.get("file_path")
        external_api = body.get("api")
        external_token = body.get("token")
        
        if not file_path:
            raise HTTPException(status_code=400, detail="Missing file_path")

        # Fetch external data
        resp = requests.get(file_path, timeout=60)
        resp.raise_for_status()
        data_to_parse = resp.text

        job_id = str(uuid.uuid4())[:8]
        all_scorecards = []
        transcripts_data = []

        # 1. Parsing Logic
        try:
            data = json.loads(data_to_parse, strict=False)
            if isinstance(data, dict):
                t_input = data.get("transcripts") or data.get("transcript")
                if isinstance(t_input, list): 
                    transcripts_data = t_input
                elif isinstance(t_input, str): 
                    transcripts_data = [{"transcript": t_input, "transcript_id": 0}]
                
                sc_input = data.get("scorecards") or data.get("scorecard")
                if isinstance(sc_input, list): 
                    all_scorecards = sc_input
                elif isinstance(sc_input, dict): 
                    all_scorecards = [sc_input]
        except:
            # Fallback Regex Parsing
            t_pattern = r'["\']transcript["\']\s*:\s*["\'](.*?)["\']\s*,\s*["\']transcript_id["\']\s*:\s*(\d+)'
            t_matches = re.findall(t_pattern, data_to_parse, re.DOTALL)
            transcripts_data = [{"transcript": m[0], "transcript_id": int(m[1])} for m in t_matches]
            
            sc_section = re.search(r'["\']scorecards?["\']\s*:\s*(.*)', data_to_parse, re.DOTALL | re.IGNORECASE)
            if sc_section:
                list_str = extract_nested_list(sc_section.group(1))
                if list_str:
                    try: 
                        parsed = ast.literal_eval(list_str)
                        all_scorecards = parsed if isinstance(parsed, list) else [parsed]
                    except: 
                        pass

        # 2. Sanitization
        if not transcripts_data:
            transcripts_data = [{"transcript": data_to_parse, "transcript_id": "RAW_DUMP"}]
        if not all_scorecards:
            all_scorecards = [{"scorecard_id": 0, "questions": [{"id": "default", "text": "Analyze transcript", "answer": []}]}]

        # 3. Setup Disk & Redis
        job_dir = os.path.join(NVME_PATH, f"job_{job_id}")
        os.makedirs(job_dir, exist_ok=True)
        total_tasks = len(transcripts_data) * len(all_scorecards)
        
        # Mapping for Hash storage
        db.hset(f"job:{job_id}", mapping={
            "status": "processing", 
            "total": total_tasks,
            "callback_url": str(external_api or ""),
            "callback_token": str(external_token or "")
        })

        # 4. Dispatch
        for t_idx, t_obj in enumerate(transcripts_data):
            clean_text = clean_transcript_text(str(t_obj.get("transcript", "")))
            t_id = t_obj.get('transcript_id', t_idx)

            for s_idx, sc in enumerate(all_scorecards):
                sc_id = sc.get("scorecard_id") or sc.get("id") or s_idx
                raw_qs = sc.get("questions")
                
                # Normalize questions list
                if isinstance(raw_qs, list):
                    curr_questions = raw_qs
                elif raw_qs:
                    curr_questions = [raw_qs]
                elif isinstance(sc, list):
                    curr_questions = sc
                else:
                    curr_questions = [sc]

                task_id = f"j{job_id}_t{t_id}_s{sc_id}"
                payload = {
                    "job_id": job_id, 
                    "transcript_id": t_id, 
                    "scorecard_id": sc_id,
                    "transcript": clean_text, 
                    "questions": curr_questions, 
                    "task_id": task_id
                }

                # Atomic write to NVME
                with open(os.path.join(job_dir, f"{task_id}.json"), "w") as f:
                    ujson.dump(payload, f)

                # Send to Celery
                celery_app.send_task('process_jts_task', args=[job_id, t_id, sc_id])

        return {"job_id": job_id, "status": "dispatched", "tasks": total_tasks}
        
    except Exception as e:
        print(f"Submission Failure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
