# Python 환경 구축

이 프로젝트의 기존 `.venv311` 메타데이터를 기준으로 환경을 고정했습니다.

- Python: **3.11.9 (CPython)**
- pip: **26.2.1**
- setuptools: **84.0.0**
- 패키지: `requirements.txt`에 정확한 버전으로 고정

가상환경 폴더 자체는 PC별 절대 경로를 포함하므로 공유하지 말고, 아래 방법으로 각 PC에서 새로 생성하세요.

## Windows: 자동 설치

1. Python 3.11.9 또는 Ollama가 없으면 스크립트가 `winget`으로 자동 설치합니다. `winget`을 사용할 수 없으면 공식 다운로드 페이지를 자동으로 엽니다.
2. 프로젝트 루트에서 PowerShell을 열고 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup-env.ps1
.\.venv311\Scripts\Activate.ps1
```

스크립트는 Python 3.11.9 및 Ollama 준비, 가상환경 생성, 패키지 설치, 의존성 충돌 검사를 수행합니다. 자동 설치 후에는 새 PowerShell 창에서 한 번 더 실행하세요.

## Windows: 수동 설치

```powershell
py -3.11 --version
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade "pip==26.2.1" "setuptools==84.0.0"
python -m pip install -r requirements.txt
python -m pip check
```

첫 명령의 결과가 `Python 3.11.9`인지 확인하세요.

## macOS / Linux

Python 3.11.9가 설치되어 있고 `python3.11`로 실행된다는 전제입니다.

```bash
python3.11 --version
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade 'pip==26.2.1' 'setuptools==84.0.0'
python -m pip install -r requirements.txt
python -m pip check
```

## 설치 확인

```powershell
python --version
python -m pip check
python -m pip list
```

정상 기준은 Python 버전이 `3.11.9`이고, `pip check` 결과가 `No broken requirements found.`인 것입니다.

## 주의

- 현재 로컬의 기존 `.venv311`은 생성 당시 사용한 원본 Python 경로가 사라져 실행되지 않습니다. 새 환경은 위 절차로 다시 만들어야 합니다.
- 기존 `.venv311`이 있으면 `setup-env.ps1`은 덮어쓰지 않고 중단합니다. 필요한 파일이 없다면 폴더를 삭제하거나 다른 이름으로 백업한 뒤 재실행하세요.
- 일부 패키지(`torch`, `faiss-cpu`, `onnxruntime` 등)는 용량이 크므로 설치에 시간이 걸릴 수 있습니다.
- API 키와 같은 비밀값은 `requirements.txt`에 포함되지 않습니다. 필요한 경우 별도의 `.env` 파일을 각자 준비하세요.
