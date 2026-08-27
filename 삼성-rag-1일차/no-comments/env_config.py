import sys
from pathlib import Path


ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))

from project_env import load_project_env, optional_env, require_env


def load_environment():
    return load_project_env()
