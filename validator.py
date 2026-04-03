import os
import sys
import json
import yaml  # PyYAML is needed, usually in these environments
from pathlib import Path

def validate():
    cwd = Path.cwd()
    errors = []
    warnings = []

    print(f"🔍 Validating OpenEnv Submission at {cwd}\n")

    # 1. README.md Meta Check
    readme_path = cwd / "README.md"
    if not readme_path.exists():
        errors.append("❌ Missing README.md")
    else:
        content = readme_path.read_text(encoding="utf-8")
        if "---" not in content or "sdk: docker" not in content:
            errors.append("❌ README.md's YAML frontmatter is missing or incorrect (needs 'sdk: docker')")
        else:
            print("✅ README.md metadata found.")

    # 2. openenv.yaml Check
    openenv_path = cwd / "openenv.yaml"
    if not openenv_path.exists():
        errors.append("❌ Missing openenv.yaml")
    else:
        try:
            with open(openenv_path, "r") as f:
                cfg = yaml.safe_load(f)
                required_keys = ["name", "version", "tasks", "action_space", "observation_space", "deploy"]
                for k in required_keys:
                    if k not in cfg:
                        errors.append(f"❌ openenv.yaml is missing required key: {k}")
                if len(cfg.get("tasks", [])) < 3:
                    warnings.append("⚠️ OpenEnv recommends at least 3 tasks (easy, medium, hard).")
            print("✅ openenv.yaml follows spec.")
        except Exception as e:
            errors.append(f"❌ Failed to parse openenv.yaml: {e}")

    # 3. Dockerfile Check
    docker_path = cwd / "Dockerfile"
    if not docker_path.exists():
        errors.append("❌ Missing Dockerfile (Mandatory for HF Space docker SDK)")
    else:
        print("✅ Dockerfile exists.")

    # 4. inference.py Check
    inference_path = cwd / "inference.py"
    if not inference_path.exists():
        errors.append("❌ Missing inference.py in root directory.")
    else:
        content = inference_path.read_text(encoding="utf-8")
        # Mandatory logs check
        for tag in ["[START]", "[STEP]", "[END]"]:
            if tag not in content:
                errors.append(f"❌ Structured logging tag '{tag}' not found in inference.py")
        
        # Mandatory variables check
        for var in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]:
            if var not in content:
                errors.append(f"❌ Mandatory variable/env check for '{var}' not found in inference.py")
        
        # OpenAI client check
        if "OpenAI" not in content or "client.chat.completions.create" not in content:
            errors.append("❌ inference.py must use OpenAI client for LLM calls.")
            
        print("✅ inference.py follows mandatory rules.")

    # 5. Requirement Check
    req_path = cwd / "requirements.txt"
    if not req_path.exists():
        errors.append("❌ Missing requirements.txt")
    else:
        print("✅ requirements.txt exists.")

    # Result Summary
    print("\n--- Validation Summary ---")
    if errors:
        for e in errors:
            print(e)
        print("\n❌ FAILED: Please fix errors before submitting.")
        sys.exit(1)
    else:
        if warnings:
            for w in warnings:
                print(w)
        print("\n✅ PASSED: Project matches Scalar Hackathon OpenEnv requirements.")
        print("🚀 Ready for submission!")

if __name__ == "__main__":
    validate()
