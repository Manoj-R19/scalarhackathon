"""
server/app.py — FastAPI server for EmailTriage OpenEnv
Endpoints: /reset /step /grader /tasks /state /health /docs
"""
from __future__ import annotations

import sys
import os
import json
from pathlib import Path
from typing import Optional

# Ensure root directory is in sys.path when running from server/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from environment import EmailTriageEnv
from models import Action

# ─────────────────────────── App Init ─────────────────────────

app = FastAPI(
    title="EmailTriage OpenEnv",
    description=(
        "🏆 Hackathon OpenEnv: Real-world email support triage benchmark. "
        "Agents classify, prioritize, draft replies, and escalate critical issues. "
        "3 tasks (easy → hard), deterministic graders, shaped rewards."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (stateful, single session)
env = EmailTriageEnv()


# ─────────────────────────── Request Models ────────────────────

class ResetRequest(BaseModel):
    task: str = Field(default="easy", description="Task ID: easy | medium | hard")


class StepRequest(BaseModel):
    action: str = Field(..., description='Action JSON')


class GraderRequest(BaseModel):
    task: Optional[str] = Field(default=None, description="Optional: override task for grading")


# ─────────────────────────── Endpoints ────────────────────────

@app.get("/", tags=["Meta"])
def root():
    return {
        "name": "EmailTriage OpenEnv",
        "version": "1.0.0",
        "description": "Real-world email support triage benchmark for training AI agents.",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/grader", "/tasks", "/state", "/health", "/docs"],
        "hackathon": "OpenEnv Scalar Hackathon 2025",
    }


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "environment": "EmailTriageEnv", "version": "1.0.0"}


@app.get("/tasks", tags=["Environment"])
def get_tasks():
    """List all available tasks with descriptions and grader weights."""
    return {"tasks": env.get_tasks()}


@app.post("/reset", tags=["Environment"])
def reset(req: ResetRequest = ResetRequest()):
    """Reset environment to a fresh task inbox."""
    try:
        obs = env.reset(task=req.task)
        return {
            "observation": obs.model_dump(),
            "message": f"Environment reset for task '{req.task}'. {obs.stats.total} emails loaded.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", tags=["Environment"])
def step(req: StepRequest):
    """Take one action in the environment."""
    if env.state is None or not env.state.inbox:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    result = env.step(req.action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/grader", tags=["Environment"])
def grader(req: GraderRequest = GraderRequest()):
    """Score the current episode."""
    if env.state is None or not env.state.inbox:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    result = env.grader()
    return result.model_dump()


@app.get("/state", tags=["Environment"])
def get_state():
    """Return full internal state (for debugging)."""
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return env.get_state().model_dump()


@app.get("/action-schema", tags=["Meta"])
def action_schema():
    return Action.model_json_schema()


# ─────────────────────────── Main Server ──────────────────────

def main():
    """Main entry point for starting the OpenEnv server."""
    import uvicorn
    # Use the server.app:app name since it's now in the server/ package
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)), reload=False)

if __name__ == "__main__":
    main()
