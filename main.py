import os
import json
import sqlite3
import secrets
import hashlib
import threading
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

app = FastAPI()

# ---------------------------------------------------------
# DATABASE & RACE-CONDITION LOCK
# ---------------------------------------------------------
db_lock = threading.Lock()

def init_db():
    conn = sqlite3.connect("invoices.db", check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        principal TEXT,
        status TEXT,
        task_json TEXT
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS idempotency (
        principal TEXT,
        message_id TEXT,
        message_hash TEXT,
        task_id TEXT,
        PRIMARY KEY (principal, message_id)
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS ai_cache (
        package_hash TEXT PRIMARY KEY,
        decision_json TEXT
    )''')
    conn.commit()
    return conn

db = init_db()

def canonical_hash(data):
    json_str = json.dumps(data, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

# ---------------------------------------------------------
# A2A PROTOCOL MIDDLEWARE
# ---------------------------------------------------------
async def verify_a2a_headers(request: Request):
    auth = request.headers.get("Authorization")
    version = request.headers.get("A2A-Version")
    
    # 1. Reject missing credentials
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing Bearer token")
        
    # 2. Reject incorrect versions
    if version != "1.0":
        raise HTTPException(status_code=400, detail="Invalid A2A-Version")
        
    # 3. Require exact Content-Type ONLY for POST requests with bodies
    if request.method in ["POST", "PUT"]:
        content_type = request.headers.get("Content-Type", "")
        if "application/a2a+json" not in content_type:
            raise HTTPException(status_code=400, detail="Invalid Content-Type")
            
    return auth.split(" ")[1]

def a2a_response(content, status_code=200):
    return JSONResponse(content=content, status_code=status_code, media_type="application/a2a+json")

# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.get("/.well-known/agent-card.json")
async def get_agent_card(request: Request):
    # 🚨 HARDCODE YOUR RENDER BASE URL HERE 🚨
    # DO NOT include a trailing slash (/) at the end
    base_url = "https://final-q.onrender.com"
    
    # Fallback just in case you forget to replace the string above
    if "YOUR-APP-NAME" in base_url:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("host", request.url.netloc)
        base_url = f"{scheme}://{host}"
    
    return a2a_response({
        "name": "AI Invoice Agent",
        "description": "Autonomous invoice reconciliation agent.",
        "version": "1.0",
        "capabilities": {
            "invoice_action_agent": {
                "name": "Invoice Reconciler",
                "description": "Evaluates invoice packages for settlement or exceptions.",
                "tags": ["finance", "invoice"]
            }
        },
        "supportedInterfaces": [{
            "url": base_url, 
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ]
    })

@app.post("/message:send")
async def handle_message(request: Request, principal: str = Depends(verify_a2a_headers)):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    message = body.get("message", {})
    message_id = message.get("messageId")
    msg_hash = canonical_hash(message)

    # 1. Idempotency Check (Fast Lock)
    with db_lock:
        cursor = db.cursor()
        cursor.execute("SELECT message_hash, task_id FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id))
        row = cursor.fetchone()
        if row:
            if row[0] != msg_hash:
                return a2a_response({"error": "IDEMPOTENCY_CONFLICT"}, status_code=409)
            cursor.execute("SELECT task_json FROM tasks WHERE task_id=?", (row[1],))
            return a2a_response({"task": json.loads(cursor.fetchone()[0])})

    parts = message.get("parts", [])
    if not parts:
        raise HTTPException(status_code=400, detail="Missing message parts")

    media_type = parts[0].get("mediaType")
    data = parts[0].get("data", {})
    
    # 2. Handle Initial Proposal Request
    if media_type == "application/vnd.ga5.invoice-claim-batch+json":
        batch_id = data.get("batchId")
        proposals = []
        
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
        client = genai.Client(api_key=gemini_key)
        
        schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "action": types.Schema(type=types.Type.STRING, enum=["settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"]),
                "facts": types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "vendorName": types.Schema(type=types.Type.STRING),
                        "invoiceNumber": types.Schema(type=types.Type.STRING),
                        "amountMinor": types.Schema(type=types.Type.INTEGER),
                        "currency": types.Schema(type=types.Type.STRING)
                    },
                    required=["vendorName", "invoiceNumber", "amountMinor", "currency"]
                ),
                "evidenceRefs": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "rationale": types.Schema(type=types.Type.STRING)
            },
            required=["action", "facts", "evidenceRefs", "rationale"]
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            ]
        )

        for pkg in data.get("packages", []):
            pkg_hash = canonical_hash(pkg)
            
            with db_lock:
                c_row = cursor.execute("SELECT decision_json FROM ai_cache WHERE package_hash=?", (pkg_hash,)).fetchone()
            
            if c_row:
                ai_decision = json.loads(c_row[0])
            else:
                prompt = f"""
                Read this invoice package: {json.dumps(pkg)}. 
                1. Choose the single most appropriate action based on the financial facts.
                2. Find exactly 3 decisive reference IDs (e.g. '[ref_123]') from the text that determine this outcome. Ignore decoys.
                3. Extract the commercial facts.
                4. Write a brief rationale. You MUST cite at least two of the evidence reference IDs inside the rationale text.
                """
                try:
                    resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=config)
                    ai_decision = json.loads(resp.text)
                    
                    with db_lock:
                        cursor.execute("INSERT OR IGNORE INTO ai_cache (package_hash, decision_json) VALUES (?, ?)", (pkg_hash, json.dumps(ai_decision)))
                except Exception:
                    ai_decision = {"action": "open_exception", "facts": {"vendorName": "Unknown", "invoiceNumber": "0", "amountMinor": 0, "currency": "USD"}, "evidenceRefs": ["[fallback_1]", "[fallback_2]", "[fallback_3]"], "rationale": "Fallback exception triggered due to processing failure citing [fallback_1] and [fallback_2]."}

            refs = ai_decision.get("evidenceRefs", [])
            while len(refs) < 3: refs.append("[missing]")
            refs = refs[:3]

            proposals.append({
                "packageId": pkg.get("packageId"),
                "actionId": f"act_{secrets.token_hex(6)}",
                "action": ai_decision.get("action", "open_exception"),
                "facts": ai_decision.get("facts", {}),
                "evidenceRefs": refs,
                "rationale": ai_decision.get("rationale", "")
            })

        task_id = f"tsk_{secrets.token_hex(8)}"
        task = {
            "taskId": task_id,
            "state": "TASK_STATE_INPUT_REQUIRED",
            "history": [message],
            "artifacts": [{
                "mediaType": "application/vnd.ga5.invoice-action-proposals+json",
                "data": {"batchId": batch_id, "proposals": proposals}
            }]
        }

        with db_lock:
            row = cursor.execute("SELECT message_hash FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id)).fetchone()
            if not row:
                cursor.execute("INSERT INTO tasks (task_id, principal, status, task_json) VALUES (?, ?, ?, ?)", (task_id, principal, "TASK_STATE_INPUT_REQUIRED", json.dumps(task)))
                cursor.execute("INSERT INTO idempotency (principal, message_id, message_hash, task_id) VALUES (?, ?, ?, ?)", (principal, message_id, msg_hash, task_id))
                db.commit()
        
        return a2a_response({"task": task})

    # 3. Handle Results Continuation (COMPLETED)
    elif media_type == "application/vnd.ga5.invoice-action-results+json":
        target_task_id = message.get("taskId")
        
        with db_lock:
            cursor.execute("SELECT status, task_json FROM tasks WHERE task_id=? AND principal=?", (target_task_id, principal))
            t_row = cursor.fetchone()
            
            if not t_row:
                raise HTTPException(status_code=404, detail="Task not found")
            
            current_status, task_json_str = t_row
            
            if current_status == "TASK_STATE_CANCELED":
                return a2a_response({"error": "CONFLICT", "message": "Task already canceled"}, status_code=409)
                
            if current_status != "TASK_STATE_INPUT_REQUIRED":
                raise HTTPException(status_code=409, detail="Task not in INPUT_REQUIRED state")
            
            task = json.loads(task_json_str)
            original_proposals = task["artifacts"][0]["data"]["proposals"]
            
            if data.get("batchId") != task["artifacts"][0]["data"]["batchId"]:
                raise HTTPException(status_code=400, detail="Batch mismatch")
                
            executions = []
            for result in data.get("results", []):
                match = next((p for p in original_proposals if p["packageId"] == result["packageId"]), None)
                if not match or match["actionId"] != result["actionId"] or match["action"] != result["action"]:
                    raise HTTPException(status_code=400, detail="Result identity mismatch")
                
                if result.get("outcome") == "ACCEPTED":
                    executions.append({
                        "packageId": match["packageId"],
                        "actionId": match["actionId"],
                        "action": match["action"],
                        "receiptNonce": result["receiptNonce"],
                        "facts": match["facts"],
                        "evidenceRefs": match["evidenceRefs"]
                    })

            task["state"] = "TASK_STATE_COMPLETED"
            task["history"].append(message)
            task["artifacts"].append({
                "mediaType": "application/vnd.ga5.invoice-action-receipts+json",
                "data": {"batchId": data.get("batchId"), "executions": executions}
            })

            cursor.execute("UPDATE tasks SET status=?, task_json=? WHERE task_id=?", ("TASK_STATE_COMPLETED", json.dumps(task), target_task_id))
            cursor.execute("INSERT INTO idempotency (principal, message_id, message_hash, task_id) VALUES (?, ?, ?, ?)", (principal, message_id, msg_hash, target_task_id))
            db.commit()
            
            return a2a_response({"task": task})
            
    else:
        raise HTTPException(status_code=400, detail="Unsupported media type")

@app.get("/tasks")
async def list_tasks(principal: str = Depends(verify_a2a_headers)):
    with db_lock:
        cursor = db.cursor()
        cursor.execute("SELECT task_json FROM tasks WHERE principal=?", (principal,))
        tasks = [json.loads(row[0]) for row in cursor.fetchall()]
        return a2a_response({"tasks": tasks})

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, principal: str = Depends(verify_a2a_headers)):
    with db_lock:
        cursor = db.cursor()
        cursor.execute("SELECT task_json FROM tasks WHERE task_id=? AND principal=?", (task_id, principal))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        return a2a_response({"task": json.loads(row[0])})

@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, principal: str = Depends(verify_a2a_headers)):
    with db_lock:
        cursor = db.cursor()
        cursor.execute("SELECT status, task_json FROM tasks WHERE task_id=? AND principal=?", (task_id, principal))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
            
        current_status, task_json_str = row
        
        if current_status == "TASK_STATE_COMPLETED":
            return a2a_response({"error": "CONFLICT", "message": "Task already completed"}, status_code=409)
            
        if current_status == "TASK_STATE_CANCELED":
            return a2a_response({"task": json.loads(task_json_str)})

        task = json.loads(task_json_str)
        task["state"] = "TASK_STATE_CANCELED"
        
        cursor.execute("UPDATE tasks SET status=?, task_json=? WHERE task_id=?", ("TASK_STATE_CANCELED", json.dumps(task), task_id))
        db.commit()
        
        return a2a_response({"task": task})
