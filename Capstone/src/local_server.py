# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from vaccess_agent import VAccessOrchestrator
from config import CONFIG

app = FastAPI(title="VAccess Agent API", version="1.0.0")

# Initialize orchestrator
orchestrator = VAccessOrchestrator(CONFIG)

# In-memory session storage (use Redis in production)
sessions = {}


class QueryRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    message: str


class QueryResponse(BaseModel):
    response: str
    session_state: str
    session_id: str


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the VAccess orchestrator."""
    try:
        session_id = request.session_id or f"session_{request.user_id}"

        result = await orchestrator.stream_query(
            session_id=session_id,
            message=request.message,
            user_id=request.user_id
        )

        return QueryResponse(
            response=result.response,
            session_state=result.state_delta.get("current_state", "unknown"),
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/find-clinics")
async def find_clinics(user_id: str, location: str):
    """Find vaccination clinics."""
    try:
        session = orchestrator.start_session(user_id, f"Find clinics in {location}")
        result = await orchestrator.find_and_schedule(session, location)

        return {
            "confirmed": result.get("confirmed", False),
            "clinic_id": result.get("clinic_id"),
            "appointment_time": result.get("appointment_time"),
            "error": result.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "VAccess Orchestrator"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "VAccess Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Query the agent with a message",
            "POST /find-clinics": "Find and book vaccination clinics",
            "GET /health": "Health check",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }


if __name__ == "__main__":
    print("Starting VAccess Agent API Server...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)