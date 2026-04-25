"""
server/app.py — Standardized OpenEnv Server (Theme 3.1)
Implements Gym-like HTTP interfaces for Post-Training post-Gradio era.
"""
from __future__ import annotations

import sys
import os
import json
from typing import Optional

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from environment import EmailTriageEnv
from models import Action
from server.ui_builder import create_ui

# ─────────────────────────── App Init ─────────────────────────

app = FastAPI(
    title="EmailTriage Enterprise Simulator",
    description=(
        "Standardized OpenEnv Benchmark for Enterprise Workflow Agents. "
        "Implements RLVR/RLVE patterns to resolve fragmented LLM RL post-training. "
        "Features: Isolated Docker execution, Causal dependencies, and P0 Crisis Injection."
    ),
    version="4.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = EmailTriageEnv()


# ─────────────────────────── Request Models ────────────────────

class ResetRequest(BaseModel):
    task: str = Field(default="expert", description="Task ID: expert (RLVE Mode)")


class StepRequest(BaseModel):
    action: str = Field(..., description='Action JSON conforming to Action Schema')


class GraderRequest(BaseModel):
    task: Optional[str] = Field(default=None)


# ─────────────────────────── Endpoints ────────────────────────

@app.get("/", tags=["Meta"])
def root():
    """Root metadata discovery for OpenEnv Clients."""
    return {
        "name": "EmailTriage Enterprise Max",
        "version": "4.0.0",
        "motive": "Democratize RL post-training via isolated, verifiable enterprise workflows.",
        "ui_dashboard": "/ui",
        "tasks": ["expert"],
        "openenv_compliance": "v0.3.0",
        "references": [
            "https://arxiv.org/abs/2408.10215",
            "https://arxiv.org/abs/2601.19100"
        ]
    }


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok", "environment": "EmailTriageRLVE", "version": "4.0.0"}


@app.get("/tasks", tags=["Environment"])
def get_tasks():
    return {"tasks": env.get_tasks()}


@app.post("/reset", tags=["Environment"])
def reset(req: ResetRequest = ResetRequest()):
    try:
        obs = env.reset(difficulty=req.task)
        return {
            "observation": obs.model_dump(),
            "message": f"Environment reset (RLVE Expert Mode). {obs.stats.total} P0-aware emails loaded.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", tags=["Environment"])
def step(req: StepRequest):
    if not env.get_state():
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
    if not env.get_state():
        raise HTTPException(status_code=400, detail="Environment not initialized.")

    result = env.grader()
    return result.model_dump()


@app.get("/state", tags=["Environment"])
def get_state():
    if not env.get_state():
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return env.get_state().model_dump()


@app.get("/action-schema", tags=["Meta"])
def action_schema():
    return Action.model_json_schema()


# ─────────────────────────── UI Mount ────────────────────────
app = gr.mount_gradio_app(app, create_ui(env), path="/ui")


# ─────────────────────────── Main Server ──────────────────────

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)), reload=False)

if __name__ == "__main__":
    main()
