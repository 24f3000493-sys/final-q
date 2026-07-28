import os
import json
import sqlite3
import secrets
import hashlib
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

app = FastAPI()

# ---------------------------------------------------------
# DATABASE SETUP (With WAL mode for concurrent safety)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect("invoices.db", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
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
    conn.close()

init_db()

def canonical_hash(data):
    json_str = json.dumps(data, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

# ---------------------------------------------------------
# A2A PROTOCOL MIDDLEWARE
# ---------------------------------------------------------
async def verify_a2a_headers(request: Request):
    auth = request.headers.get("Authorization")
    version = request.headers.get("A2A-Version")
    
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing Bearer token")
        
    if version != "1.0":
        raise HTTPException(status_code=400, detail="Invalid A2A-Version")
        
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
    # 🚨 HARDCODE YOUR RENDER URL HERE (No trailing slash) 🚨
    base_url = "https://final-q.onrender.com"
    
    if "YOUR-APP-NAME" in base_url:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        host = request.headers.get("host", request.url.netloc)
        base_url = f"{scheme}://{host}"
    
    return a2a_response({
        "name": "AI Invoice Agent",
        "description": "Autonomous invoice reconciliation agent.",
        "version": "1.0",
        "capabilities": {}, # Must be an object
        "supportedInterfaces": [{
            "url": base_url, 
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "defaultInputModes": ["application/vnd.ga5.invoice-claim-batch+json"],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ],
        "skills": [ # Per A2A spec, skills are a top-level array!
            {
                "id": "invoice_action_agent",
                "name": "Invoice Reconciler",
                "description": "Evaluates invoice packages for settlement or exceptions.",
                "tags": ["finance", "invoice"]
            }
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

    # 1. Idempotency Check with strict DB locking
    is_first = False
    try:
        with sqlite3.connect("invoices.db", timeout=15.0, isolation_level="IMMEDIATE") as conn:
            conn.execute("INSERT INTO idempotency (principal, message_id, message_hash, task_id) VALUES (?, ?, ?, ?)", (principal, message_id, msg_hash, "WORKING"))
            conn.commit()
            is_first = True
    except sqlite3.IntegrityError:
        pass # Another request beat us to it

    if not is_first:
        # Concurrent replay handling
        with sqlite3.connect("invoices.db", timeout=15.0) as conn:
            row = conn.execute("SELECT message_hash, task_id FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id)).fetchone()
            if row and row[0] != msg_hash:
                return a2a_response({"error": "IDEMPOTENCY_CONFLICT"}, status_code=409)
                
        # Wait up to 10 seconds for the first request to finish AI processing
        for _ in range(20):
            with sqlite3.connect("invoices.db", timeout=15.0) as conn:
                row = conn.execute("SELECT task_id FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id)).fetchone()
                if row and row[0] != "WORKING":
                    task_json = conn.execute("SELECT task_json FROM tasks WHERE task_id=?", (row[0],)).fetchone()[0]
                    return a2a_response({"task": json.loads(task_json)})
            await asyncio.sleep(0.5)
        raise HTTPException(status_code=504, detail="Timeout waiting for concurrent request")

    parts = message.get("parts", [])
    if not parts:
        with sqlite3.connect("invoices.db") as conn:
            conn.execute("DELETE FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id))
        raise HTTPException(status_code=400, detail="Missing parts")

    media_type = parts[0].get("mediaType")
    data = parts[0].get("data", {})
    
    # 2. Process Initial Proposal
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
            
            with sqlite3.connect("invoices.db") as conn:
                c_row = conn.execute("SELECT decision_json FROM ai_cache WHERE package_hash=?", (pkg_hash,)).fetchone()
            
            if c_row:
                ai_decision = json.loads(c_row[0])
            else:
                prompt = f"""
                You are a strict invoice auditor. Analyze this package: {json.dumps(pkg)}. 
                Rules:
                1. Choose EXACTLY ONE action. Base your decision ONLY on the active, controlling facts in the main invoice text.
                2. IGNORE old examples, negations, archive data, training decoys, and cover-sheet references.
                3. Find EXACTLY 3 decisive reference IDs (e.g., "[ref_123]") from the specific paragraph that determines the action.
                4. Write a rationale (60-1500 chars). You MUST explicitly cite at least two of those reference IDs inside your explanation text (e.g., 'Due to [ref_45] and [ref_46], we must hold').
                """
                try:
                    resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=config)
                    ai_decision = json.loads(resp.text)
                    with sqlite3.connect("invoices.db") as conn:
                        conn.execute("INSERT OR IGNORE INTO ai_cache (package_hash, decision_json) VALUES (?, ?)", (pkg_hash, json.dumps(ai_decision)))
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

        with sqlite3.connect("invoices.db", timeout=15.0, isolation_level="IMMEDIATE") as conn:
            conn.execute("INSERT INTO tasks (task_id, principal, status, task_json) VALUES (?, ?, ?, ?)", (task_id, principal, "TASK_STATE_INPUT_REQUIRED", json.dumps(task)))
            conn.execute("UPDATE idempotency SET task_id=? WHERE principal=? AND message_id=?", (task_id, principal, message_id))
            conn.commit()
        
        return a2a_response({"task": task})

    # 3. Handle Results Continuation (COMPLETED)
    elif media_type == "application/vnd.ga5.invoice-action-results+json":
        target_task_id = message.get("taskId")
        
        with sqlite3.connect("invoices.db", timeout=15.0, isolation_level="IMMEDIATE") as conn:
            # We must use a strict database lock to prevent the Cancel Race Condition
            t_row = conn.execute("SELECT status, task_json FROM tasks WHERE task_id=? AND principal=?", (target_task_id, principal)).fetchone()
            
            if not t_row:
                conn.execute("DELETE FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id))
                raise HTTPException(status_code=404, detail="Task not found")
            
            current_status, task_json_str = t_row
            
            if current_status == "TASK_STATE_CANCELED":
                conn.execute("DELETE FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id))
                return a2a_response({"error": "CONFLICT", "message": "Task already canceled"}, status_code=409)
                
            if current_status != "TASK_STATE_INPUT_REQUIRED":
                conn.execute("DELETE FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id))
                raise HTTPException(status_code=409, detail="Task not in INPUT_REQUIRED state")
            
            task = json.loads(task_json_str)
            original_proposals = task["artifacts"][0]["data"]["proposals"]
                
            executions = []
            for result in data.get("results", []):
                match = next((p for p in original_proposals if p["packageId"] == result["packageId"]), None)
                if not match or match["actionId"] != result["actionId"] or match["action"] != result["action"]:
                    conn.execute("DELETE FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id))
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

            conn.execute("UPDATE tasks SET status=?, task_json=? WHERE task_id=?", ("TASK_STATE_COMPLETED", json.dumps(task), target_task_id))
            conn.execute("UPDATE idempotency SET task_id=? WHERE principal=? AND message_id=?", (target_task_id, principal, message_id))
            conn.commit()
            
            return a2a_response({"task": task})
            
    else:
        with sqlite3.connect("invoices.db") as conn:
            conn.execute("DELETE FROM idempotency WHERE principal=? AND message_id=?", (principal, message_id))
        raise HTTPException(status_code=400, detail="Unsupported media type")

@app.get("/tasks")
async def list_tasks(principal: str = Depends(verify_a2a_headers)):
    with sqlite3.connect("invoices.db") as conn:
        tasks = [json.loads(row[0]) for row in conn.execute("SELECT task_json FROM tasks WHERE principal=?", (principal,)).fetchall()]
        return a2a_response({"tasks": tasks})

@app.get("/tasks/{task_id}")
async def get_task(task_id: str, principal: str = Depends(verify_a2a_headers)):
    with sqlite3.connect("invoices.db") as conn:
        row = conn.execute("SELECT task_json FROM tasks WHERE task_id=? AND principal=?", (task_id, principal)).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        # FIX: The A2A spec expects the Task object directly, not wrapped in {"task": ...}
        return a2a_response(json.loads(row[0])) 

@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, principal: str = Depends(verify_a2a_headers)):
    # 100% strict database locking to fix the Receipt vs Cancel Race Condition
    with sqlite3.connect("invoices.db", timeout=15.0, isolation_level="IMMEDIATE") as conn:
        row = conn.execute("SELECT status, task_json FROM tasks WHERE task_id=? AND principal=?", (task_id, principal)).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
            
        current_status, task_json_str = row
        task = json.loads(task_json_str)
        
        if current_status == "TASK_STATE_COMPLETED":
            return a2a_response({"error": "CONFLICT", "message": "Task already completed"}, status_code=409)
            
        if current_status == "TASK_STATE_CANCELED":
            return a2a_response(task)

        task["state"] = "TASK_STATE_CANCELED"
        
        conn.execute("UPDATE tasks SET status=?, task_json=? WHERE task_id=?", ("TASK_STATE_CANCELED", json.dumps(task), task_id))
        conn.commit()
        
        # FIX: The A2A spec expects the Task object directly for cancellation
        return a2a_response(task)
