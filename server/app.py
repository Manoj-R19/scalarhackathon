"""
server/app.py — Global Standardized OpenEnv Server (v4.0.5)
Refactored for ROOT UI landing page to fix HF 404 errors.
"""
from __future__ import annotations
import sys
import os
import gradio as gr
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from environment import EmailTriageEnv
from models import Action
from server.ui_builder import create_ui

# 1. Initialize Core App
app = FastAPI(title="EmailTriage Enterprise Simulator", version="4.0.5")
env = EmailTriageEnv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Discovery Metadata Endpoint
@app.get("/meta")
def get_metadata():
    """Discovery Metadata for OpenEnv Agents."""
    return {
        "name": "EmailTriage Enterprise Max",
        "version": "4.0.0",
        "motive": "Democratize RL post-training via isolated, verifiable enterprise workflows.",
        "ui_dashboard": "/",
        "tasks": ["expert"],
        "openenv_compliance": "v0.3.0",
        "references": [
            "https://arxiv.org/abs/2408.10215",
            "https://arxiv.org/abs/2601.19100"
        ]
    }

# 3. Environment Endpoints
@app.post("/reset")
def reset(task: str = "expert"):
    obs = env.reset(difficulty=task)
    return {"observation": obs.model_dump(), "message": "Reset success."}

@app.post("/step")
def step(action: str):
    res = env.step(action)
    return {"observation": res.observation.model_dump(), "reward": res.reward, "done": res.done, "info": res.info}

@app.post("/grader")
def grader():
    return env.grader().model_dump()

@app.get("/state")
def get_state():
    return env.get_state().model_dump()

# 4. Gradio Dashboard Mount at ROOT
# HF Spaces expect the main UI to be at / for the landing page
ui_app = create_ui(env)
app = gr.mount_gradio_app(app, ui_app, path="/")

# Note: Any requests to / will now be caught by Gradio. 
# FastAPI endpoints like /meta, /reset, /step still work as they are defined BEFORE the mount.

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)))

if __name__ == "__main__":
    main()
