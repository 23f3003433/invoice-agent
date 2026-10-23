
import os
import json
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from storage import (
    init_db,
    canonical_message_hash,
    get_saved_message,
    get_task,
    list_tasks,
    create_task_if_new,
    update_task,
)

app = FastAPI(title="A2A Invoice Agent")

BASE_URL = os.getenv(
    "A2A_BASE_URL",
    "http://127.0.0.1:8000/"
).rstrip("/") + "/"

EXPECTED_TOKEN = os.getenv("A2A_BEARER_TOKEN")

BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
A2A_MEDIA = "application/a2a+json"


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": "Invoice Action Agent",
        "description": (
            "Evaluates invoice packages and proposes "
            "evidence-based business actions."
        ),
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "capabilities": {},
        "defaultInputModes": [BATCH_MEDIA],
        "defaultOutputModes": [PROPOSALS_MEDIA, RECEIPTS_MEDIA],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Reviews invoice evidence and proposes "
                    "safe actions with supporting references."
                ),
                "tags": [
                    "invoices",
                    "evidence",
                    "approval",
                    "receipts",
                ],
            }
        ],
    }


def error(status_code, message):
    return JSONResponse(
        status_code=status_code,
        content={"error": message},
        media_type=A2A_MEDIA,
    )


def authenticate(request):
    authorization = request.headers.get("Authorization", "")
    parts = authorization.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None, error(401, "Authentication required")

    token = parts[1]

    if not EXPECTED_TOKEN or token != EXPECTED_TOKEN:
        return None, error(401, "Authentication failed")

    # The token identifies the principal. Never accept a principal
    # supplied by the request body.
    return token, None


def validate_protocol(request):
    if request.headers.get("A2A-Version") != "1.0":
        return error(400, "Unsupported A2A version")

    content_type = request.headers.get(
        "Content-Type", ""
    ).split(";")[0].strip().lower()

    if content_type != A2A_MEDIA:
        return error(415, "Unsupported media type")

    return None


def parse_json_body(raw):
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    return value if isinstance(value, dict) else None


def find_part(message, media_type):
    for part in message.get("parts", []):
        if isinstance(part, dict) and part.get("mediaType") == media_type:
            return part

    return None


@app.post("/message:send")
async def send_message(request: Request):
    principal, auth_error = authenticate(request)
    if auth_error:
        return auth_error

    protocol_error = validate_protocol(request)
    if protocol_error:
        return protocol_error

    raw = await request.body()
    body = parse_json_body(raw)

    if body is None:
        return error(400, "Invalid JSON request")

    message = body.get("message")
    if not isinstance(message, dict):
        return error(400, "Message is required")

    message_id = message.get("messageId")
    role = message.get("role")

    if not isinstance(message_id, str) or not message_id:
        return error(400, "Message ID is required")

    if role != "ROLE_USER":
        return error(400, "Invalid message role")

    # Continuations must reference an existing task owned by this user.
    task_id = message.get("taskId")
    if task_id:
        task = get_task(principal, task_id)

        if task is None:
            return error(404, "Task not found")

        if message.get("contextId") != task.get("contextId"):
            return error(400, "Context mismatch")

        return error(
            501,
            "Result continuation is not implemented yet"
        )

    # Initial invoice-batch message.
    part = find_part(message, BATCH_MEDIA)
    if part is None or not isinstance(part.get("data"), dict):
        return error(400, "Invoice batch data is required")

    batch = part["data"]
    batch_id = batch.get("batchId")
    packages = batch.get("packages")

    if (
        not isinstance(batch_id, str)
        or not batch_id
        or not isinstance(packages, list)
        or not packages
    ):
        return error(400, "Invalid invoice batch")

    # The actual evidence-based decision engine must be connected
    # before creating an INPUT_REQUIRED Task with proposals.
    return error(
        501,
        "Invoice decision engine is not configured yet"
    )


@app.get("/tasks/{task_id}")
def read_task(task_id: str, request: Request):
    principal, auth_error = authenticate(request)
    if auth_error:
        return auth_error

    task = get_task(principal, task_id)

    if task is None:
        return error(404, "Task not found")

    return JSONResponse(content=task, media_type=A2A_MEDIA)


@app.get("/tasks")
def read_tasks(request: Request):
    principal, auth_error = authenticate(request)
    if auth_error:
        return auth_error

    return JSONResponse(
        content={"tasks": list_tasks(principal)},
        media_type=A2A_MEDIA,
    )


@app.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal, auth_error = authenticate(request)
    if auth_error:
        return auth_error

    protocol_error = validate_protocol(request)
    if protocol_error:
        return protocol_error

    task = get_task(principal, task_id)

    if task is None:
        return error(404, "Task not found")

    state = task.get("status", {}).get("state")

    if state in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
        return error(409, "Task is already terminal")

    # The final implementation must make cancellation atomic with
    # receipt processing. Do not use this simplified transition
    # as the final assignment implementation.
    task["status"]["state"] = "TASK_STATE_CANCELED"

    if not update_task(principal, task):
        return error(409, "Task state changed; retry")

    return JSONResponse(content=task, media_type=A2A_MEDIA)
