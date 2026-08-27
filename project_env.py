import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


def find_project_root(start):
    path = Path(start).resolve()
    directory = path if path.is_dir() else path.parent

    for candidate in (directory, *directory.parents):
        if (candidate / "project_env.py").is_file():
            return candidate

    raise RuntimeError("project_env.py 파일을 기준으로 프로젝트 루트를 찾을 수 없습니다.")


def load_project_env(env_file=None):
    selected_file = Path(env_file) if env_file else PROJECT_ROOT / ".env"
    load_dotenv(selected_file, override=True)
    return selected_file


def require_env(name):
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(
            f"프로젝트 root .env 파일에 {name} 값을 입력한 뒤 다시 실행하세요. "
            f"예시는 {PROJECT_ROOT / '.env.example'} 파일을 참고하세요."
        )
    return value


def optional_env(name, default=None):
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default
