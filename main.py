import os
import json
import sqlite3
import secrets
import hashlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

app = FastAPI()

# ---------------------------------------------------------
# DATABASE & HASHING (The Agent's Durable Memory)
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect("incidents.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            content_hash TEXT,
            state JSON
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            receipt_id TEXT PRIMARY KEY,
            content_hash TEXT
        )
    """)
    conn.commit()
    return conn

db = init_db()

def canonical_hash(data):
    """Creates a SHA-256 hash of recursively key-sorted, compact JSON."""
    json_str = json.dumps(data, separators=(',', ':'), sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

# ---------------------------------------------------------
# OTLP TRACE GENERATOR
# ---------------------------------------------------------
def build_otlp_trace(state):
    def create_span(span_id, parent_id, name, kind):
        span = {
            "traceId": state["traceId"],
            "spanId": span_id,
            "name": name,
            "kind": kind,
            "attributes": [
                {"key": "ga5.run.id", "value": {"stringValue": state["runId"]}},
                {"key": "ga5.public.marker", "value": {"stringValue": state["publicMarker"]}}
            ]
        }
        if parent_id:
            span["parentSpanId"] = parent_id
        return span

    spans = []
    
    # 1. SERVER Span
    spans.append(create_span(state["serverSpanId"], state.get("incomingParentSpanId"), "POST /v2/incidents", 2))
    
    # 2. INTERNAL Agent Span
    spans.append(create_span(state["agentSpanId"], state["serverSpanId"], "invoke_agent incident-response", 1))
    
    # 3. CLIENT Model Span
    model_span = create_span(state["modelSpanId"], state["agentSpanId"], "chat incident-plan", 3)
    model_span["attributes"].extend([
        {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
        {"key": "gen_ai.request.model", "value": {"stringValue": "gemini-2.5-flash"}}
    ])
    spans.append(model_span)
    
    # 4. LOGICAL ACTIONS & ATTEMPTS
    logical_map = {}
    for dispatch in state["actionLog"]:
        act_id = dispatch["actionId"]
        if act_id not in logical_map:
            logical_span_id = secrets.token_hex(8)
            logical_map[act_id] = logical_span_id
            
            tool_span = create_span(logical_span_id, state["agentSpanId"], f"execute_tool {dispatch['toolName']}", 1)
            tool_span["attributes"].extend([
                {"key": "ga5.action.id", "value": {"stringValue": act_id}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": dispatch['toolName']}},
                {"key": "gen_ai.tool.call.id", "value": {"stringValue": dispatch['callId']}},
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}}
            ])
            spans.append(tool_span)
            
        # PHYSICAL ATTEMPT
        attempt_span_id = dispatch["traceparent"].split("-")[2]
        attempt_span = create_span(attempt_span_id, logical_map[act_id], f"POST tool/{dispatch['toolName']}", 3)
        
        attrs = [
            {"key": "ga5.action.id", "value": {"stringValue": act_id}},
            {"key": "ga5.attempt", "value": {"intValue": dispatch['attempt']}},
            {"key": "http.request.method", "value": {"stringValue": "POST"}},
            {"key": "http.request.resend_count", "value": {"intValue": dispatch['attempt'] - 1}}
        ]
        
        # Match receipt
        receipt = next((r for r in state["receiptLog"] if r.get("callId") == dispatch["callId"] and r.get("attempt") == dispatch["attempt"]), None)
        if receipt:
            attrs.extend([
                {"key": "ga5.receipt.id", "value": {"stringValue": receipt["receiptId"]}},
                {"key": "ga5.receipt.nonce", "value": {"stringValue": receipt["nonce"]}}
            ])
            
            if receipt.get("status") == 503:
                attempt_span["status"] = {"code": 2}
                attrs.append({"key": "error.type", "value": {"stringValue": "503"}})
            elif receipt.get("errorType") == "timeout":
                attempt_span["status"] = {"code": 2}
                attrs.append({"key": "error.type", "value": {"stringValue": "timeout"}})
                
        attempt_span["attributes"].extend(attrs)
        spans.append(attempt_span)
        
    # 5. FAN OUT JOIN
    diag_count = len(set(d["actionId"] for d in state["actionLog"] if d.get("phase") == "diagnostic"))
    if diag_count > 1:
        join_span = create_span(secrets.token_hex(8), state["agentSpanId"], "incident.join", 1)
        # FIX: The join span MUST link to every independent diagnostic tool span
        links = []
        for d in state["actionLog"]:
            if d.get("phase") == "diagnostic":
                l_id = logical_map[d["actionId"]]
                if not any(lnk["spanId"] == l_id for lnk in links):
                    links.append({"traceId": state["traceId"], "spanId": l_id})
        if links:
            join_span["links"] = links
        spans.append(join_span)
        
    # 6. APPROVAL GATE
    approval_receipt = next((r for r in state["receiptLog"] if "approvalId" in r), None)
    if approval_receipt:
        app_span = create_span(secrets.token_hex(8), state["agentSpanId"], "approval_gate", 1)
        app_span["attributes"].extend([
            {"key": "ga5.approval.id", "value": {"stringValue": approval_receipt["approvalId"]}},
            {"key": "ga5.approval.nonce", "value": {"stringValue": approval_receipt["nonce"]}}
        ])
        spans.append(app_span)
        
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}

def format_response(state):
    resp = {"runId": state["runId"], "status": state["status"]}
    if state["status"] == "waiting":
        resp["diagnosis"] = state["diagnosis"]
        resp["dispatches"] = state["dispatches"]
        resp["approvals"] = state["approvals"]
    else:
        resp["diagnosis"] = state["diagnosis"]
        resp["chosenEffect"] = state["chosenEffect"]
        resp["suppressed"] = []
        resp["actionLog"] = state["actionLog"]
        resp["receiptLog"] = state["receiptLog"]
        resp["otlp"] = build_otlp_trace(state)
    return JSONResponse(content=resp)

# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------
@app.post("/v2/incidents")
async def create_incident(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if body.get("profile") != "ga5-incident-agent/v2":
        raise HTTPException(status_code=400, detail="Unsupported profile")
        
    run_id = body.get("runId")
    content_hash = canonical_hash(body)
    
    # 1. Replay Protection
    cursor = db.execute("SELECT content_hash, state FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    if row:
        if row[0] != content_hash:
            return JSONResponse(status_code=409, content={"error": "Conflict"})
        return format_response(json.loads(row[1]))
        
    # FIX: Check for incoming traceparent header to preserve global trace continuity
    incoming_traceparent = request.headers.get("traceparent")
    incoming_tracestate = request.headers.get("tracestate")
    
    if incoming_traceparent:
        parts = incoming_traceparent.split("-")
        trace_id = parts[1]
        server_parent_span_id = parts[2]
    else:
        trace_id = secrets.token_hex(16)
        server_parent_span_id = None
        incoming_tracestate = None
        
    # 2. AI Processing via Gemini API
    safe_body = body.copy()
    safe_body.pop("sensitive", None) 
    
    prompt = f"""
    You are an SRE incident agent. Read this incident carefully:
    {json.dumps(safe_body)}

    Rules:
    1. Pick exactly 1 rootCause from allowedRootCauses.
    2. Pick 2 to 4 evidence IDs (e.g., "[ev_123]") from the transcript that explicitly prove it. Do not duplicate IDs.
    3. Pick 1 to 3 diagnostic tools from the toolCatalog to confirm it. Provide EXACT arguments based on the incident text.
    4. Pick exactly 1 effect tool from policy.effectTools for recovery. Provide EXACT arguments.
    
    Output strictly JSON matching this structure:
    {{
      "rootCause": "string",
      "evidence": ["string"],
      "diagnostics": [
        {{"toolName": "string", "arguments": {{"key": "value"}}, "evidence": ["string"]}}
      ],
      "effectToolName": "string",
      "effectArguments": {{"key": "value"}}
    }}
    """
    
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set")
    
    try:
        client = genai.Client(api_key=gemini_key)
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            safety_settings=[
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE)
            ]
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config
        )
        
        ai_text = response.text.strip()
        if ai_text.startswith("```json"):
            ai_text = ai_text[7:-3].strip()
        elif ai_text.startswith("```"):
            ai_text = ai_text[3:-3].strip()
            
        ai_plan = json.loads(ai_text)
            
    except Exception as e:
        print(f"CRITICAL AI ERROR: {str(e)}") 
        raise HTTPException(status_code=502, detail=f"AI service failed: {str(e)}")
    
    state = {
        "runId": run_id, "status": "waiting", "publicMarker": body.get("publicMarker"),
        "traceId": trace_id, "serverSpanId": secrets.token_hex(8),
        "incomingParentSpanId": server_parent_span_id,
        "incomingTracestate": incoming_tracestate,
        "agentSpanId": secrets.token_hex(8), "modelSpanId": secrets.token_hex(8),
        "diagnosis": {"rootCause": ai_plan.get("rootCause"), "evidence": ai_plan.get("evidence", [])},
        "dispatches": [], "approvals": [], "actionLog": [], "receiptLog": [],
        "chosenEffect": ai_plan.get("effectToolName"), "effectArguments": ai_plan.get("effectArguments", {}),
        "policy": body.get("policy", {})
    }
    
    # 3. Generate Diagnostic Dispatches
    for diag in ai_plan.get("diagnostics", []):
        span_id = secrets.token_hex(8)
        
        # Ensure exact formatting of the diagnostic evidence
        diag_evidence = diag.get("evidence", [])
        if not isinstance(diag_evidence, list) or len(diag_evidence) == 0:
            diag_evidence = [state["diagnosis"]["evidence"][0]] if state["diagnosis"]["evidence"] else []
            
        dispatch = {
            "actionId": f"act_{secrets.token_hex(4)}",
            "callId": f"call_{secrets.token_hex(4)}",
            "phase": "diagnostic",
            "toolName": diag.get("toolName"),
            "arguments": diag.get("arguments", {}),
            "evidence": diag_evidence, 
            "attempt": 1,
            "traceparent": f"00-{trace_id}-{span_id}-01"
        }
        
        if incoming_tracestate:
            dispatch["tracestate"] = incoming_tracestate
            
        state["dispatches"].append(dispatch)
        state["actionLog"].append(dispatch)
        
    db.execute("INSERT INTO runs (run_id, content_hash, state) VALUES (?, ?, ?)", (run_id, content_hash, json.dumps(state)))
    db.commit()
    return format_response(state)

@app.post("/v2/incidents/{run_id}/receipts")
async def handle_receipt(run_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    receipt_id = body.get("receiptId")
    content_hash = canonical_hash(body)
    
    cursor = db.execute("SELECT content_hash FROM receipts WHERE receipt_id = ?", (receipt_id,))
    row = cursor.fetchone()
    if row:
        if row[0] != content_hash: return JSONResponse(status_code=409, content={"error": "Conflict"})
        cursor = db.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,))
        return format_response(json.loads(cursor.fetchone()[0]))
        
    cursor = db.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    db_row = cursor.fetchone()
    if not db_row:
        raise HTTPException(status_code=404, detail="Run ID not found")
        
    state = json.loads(db_row[0])
    
    # 1. Process Tool Outcomes
    if "outcomes" in body:
        for outcome in body["outcomes"]:
            active = next((d for d in state["dispatches"] if d["callId"] == outcome["callId"]), None)
            if not active: continue
                
            state["receiptLog"].append(outcome)
            state["dispatches"].remove(active)
            
            if outcome.get("status") == 503: 
                # FIX: Keep EXACT SAME callId for retries! Only generate new span ID.
                span_id = secrets.token_hex(8)
                retry = active.copy()
                retry["attempt"] += 1
                retry["traceparent"] = f"00-{state['traceId']}-{span_id}-01"
                state["dispatches"].append(retry)
                state["actionLog"].append(retry)
                
            elif outcome.get("errorType") == "timeout": 
                state["status"] = "failed"
                state["dispatches"] = []
                break
                
            elif outcome.get("status") == 200:
                if active["phase"] == "effect":
                    state["status"] = "completed"
                    state["dispatches"] = []
                
    # 2. Process Approval Receipts
    if "approvals" in body:
        for app_receipt in body["approvals"]:
            active_app = next((a for a in state["approvals"] if a["approvalId"] == app_receipt["approvalId"]), None)
            if active_app and app_receipt.get("decision") == "approved":
                state["receiptLog"].append({"receiptId": receipt_id, "approvalId": app_receipt["approvalId"], "decision": "approved", "nonce": app_receipt["nonce"]})
                state["approvals"].remove(active_app)
                
                span_id = secrets.token_hex(8)
                effect_dispatch = {
                    "actionId": active_app["actionId"],
                    "callId": f"call_{secrets.token_hex(4)}",
                    "phase": "effect", "toolName": active_app["toolName"],
                    "arguments": state.get("effectArguments", {}),
                    "evidence": state["diagnosis"]["evidence"],
                    "attempt": 1,
                    "traceparent": f"00-{state['traceId']}-{span_id}-01"
                }
                effect_dispatch["arguments"]["approvalId"] = app_receipt["approvalId"]
                effect_dispatch["arguments"]["approvalNonce"] = app_receipt["nonce"]
                
                if state.get("incomingTracestate"):
                    effect_dispatch["tracestate"] = state["incomingTracestate"]
                
                state["dispatches"].append(effect_dispatch)
                state["actionLog"].append(effect_dispatch)
                
    # 3. Transition to Effect Phase
    if state["status"] == "waiting" and not state["dispatches"] and not state["approvals"]:
        effect_tool = state["chosenEffect"]
        if effect_tool in state["policy"].get("approvalRequiredFor", []):
            digest = canonical_hash(state.get("effectArguments", {}))
            state["approvals"].append({
                "approvalId": f"app_{secrets.token_hex(4)}",
                "actionId": f"act_{secrets.token_hex(4)}",
                "toolName": effect_tool,
                "argumentsDigest": digest
            })
        else:
            span_id = secrets.token_hex(8)
            effect_dispatch = {
                "actionId": f"act_{secrets.token_hex(4)}",
                "callId": f"call_{secrets.token_hex(4)}",
                "phase": "effect", "toolName": effect_tool,
                "arguments": state.get("effectArguments", {}), 
                "evidence": state["diagnosis"]["evidence"],
                "attempt": 1, 
                "traceparent": f"00-{state['traceId']}-{span_id}-01"
            }
            if state.get("incomingTracestate"):
                effect_dispatch["tracestate"] = state["incomingTracestate"]
                
            state["dispatches"].append(effect_dispatch)
            state["actionLog"].append(effect_dispatch)
            
    db.execute("INSERT INTO receipts (receipt_id, content_hash) VALUES (?, ?)", (receipt_id, content_hash))
    db.execute("UPDATE runs SET state = ? WHERE run_id = ?", (json.dumps(state), run_id))
    db.commit()
    return format_response(state)

@app.get("/v2/incidents/{run_id}")
async def get_incident(run_id: str):
    cursor = db.execute("SELECT state FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    if not row: raise HTTPException(status_code=404)
    return format_response(json.loads(row[0]))
