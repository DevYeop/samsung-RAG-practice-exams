# 삼성 RAG 실습 안내

이 저장소는 LangChain, LangGraph, LlamaIndex를 이용해 RAG(Retrieval-Augmented Generation)를 단계적으로 익히는 실습 자료입니다. 처음 실행하는 분은 이 문서의 순서대로 환경을 준비한 뒤 `삼성-rag-1일차/no-comments`의 실습을 실행하세요.

> **가장 먼저 기억할 점**: 실습 1-2를 포함한 다수의 실습은 OpenAI API가 아니라 **내 컴퓨터의 Ollama 모델**을 사용합니다. 따라서 API 키가 없어도 되지만, Ollama 앱(또는 서버)과 모델은 준비되어 있어야 합니다.

## 1. 실습 자료 구성

| 위치 | 내용 |
| --- | --- |
| `삼성-rag-1일차` | LangChain 기반 RAG, Self-RAG, reranker, GraphRAG 실습 |
| `삼성-rag-1일차/no-comments` | 주석 없이 직접 읽고 해설하는 실습 코드 |
| `삼성-rag-2일차` | LlamaIndex, Workflow, Modular RAG 심화 실습 |
| `ENVIRONMENT_SETUP.md` | 프로젝트 루트 Python 환경을 새로 만드는 상세 안내 |

## 2. 준비물

- Windows 10 이상 권장
- **Python 3.11.9**
- [Ollama for Windows](https://ollama.com/download/windows)
- 패키지와 모델 다운로드를 위한 인터넷 연결
- 약 15GB 이상의 여유 디스크 공간 권장

Python 설치 시에는 **Add python.exe to PATH**와 **Python Launcher**를 선택하세요.

## 3. 가장 쉬운 환경 설정: no-comments 실습

PowerShell을 열고 아래를 차례로 실행합니다. 첫 실행에서는 패키지와 모델을 내려받으므로 시간이 걸릴 수 있습니다.

```powershell
cd "프로젝트경로\삼성-rag-1일차\no-comments"
Set-ExecutionPolicy -Scope Process Bypass
.\setup-windows.ps1
```

이 스크립트는 Python 버전 확인, 가상환경 생성, 패키지 설치, `.env` 준비, Ollama 모델 다운로드를 처리합니다. reranker 모델 다운로드를 나중에 하려면 다음을 사용하세요.

```powershell
.\setup-windows.ps1 -SkipHuggingFaceModel
```

### 수동으로 설정할 때

```powershell
cd "프로젝트경로\삼성-rag-1일차\no-comments"
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade "pip==26.2.1" "setuptools==84.0.0"
python -m pip install -r requirements.txt
python -m pip check
ollama pull llama3.1
ollama pull nomic-embed-text
```

Git Bash에서는 가상환경을 아래처럼 활성화합니다.

```bash
source .venv311/Scripts/activate
```

## 4. Ollama: 로컬 LLM 준비

다음 역할을 로컬 모델이 수행합니다.

| 모델 | 역할 |
| --- | --- |
| `llama3.1` | 답변 생성, 질문 라우팅, 관련성·할루시네이션 판정 |
| `nomic-embed-text` | 문서를 숫자 벡터로 바꿔 Chroma에서 검색할 수 있게 하는 임베딩 모델 |

설치·연결 상태는 다음으로 확인합니다.

```powershell
ollama list
```

`llama3.1`과 `nomic-embed-text`가 목록에 보이면 준비된 것입니다. 목록을 볼 수 없거나 실행 중 `WinError 10061`이 발생하면 Ollama 앱을 시작 메뉴에서 실행하세요. 앱으로 해결되지 않으면 별도 터미널에서 아래 명령을 실행한 채로 두고, 새 터미널에서 실습을 다시 실행합니다.

```powershell
ollama serve
```

## 5. API 키가 필요한 실습과 필요 없는 실습

모든 실습에 API 키가 필요한 것은 아닙니다.

| 실습 | 필요한 설정 |
| --- | --- |
| 1일차 실습 1-1, **1-2(Self-RAG)**, 2-1, 2-2 | Ollama만 필요, API 키 불필요 |
| 1일차 실습 1-3, 1-4, 3-1 | `TAVILY_API_KEY` 필요 |
| 1일차 실습 3-2 GraphRAG | `OPENAI_API_KEY` 필요 |

키가 필요한 실습을 하기 전에는 프로젝트 최상위에서 `.env.example`을 `.env`로 복사한 뒤 값을 입력합니다.

```powershell
cd "프로젝트경로"
Copy-Item .env.example .env
```

`.env`는 개인 비밀값 파일이므로 Git에 올리지 마세요. 이미 `.gitignore`에 제외 규칙이 있습니다.

## 6. 실습 실행

가상환경을 활성화한 상태에서 실행합니다. 예를 들어 실습 1-2는 다음과 같습니다.

```powershell
cd "프로젝트경로\삼성-rag-1일차\no-comments"
.\.venv311\Scripts\Activate.ps1
python ".\실습1-2-할루시네이션 자가 검증 기능을 활용한 Self-RAG 챗봇 구현.py"
```

실습을 마친 뒤에는 다음으로 가상환경을 종료합니다.

```text
deactivate
```

## 7. 자주 발생하는 문제

### `httpx.ConnectError: [WinError 10061]`

**원인:** Python 코드가 `ChatOllama`를 통해 로컬 Ollama 서버에 연결하려 했지만, Ollama가 실행 중이 아닙니다. API 키 문제가 아닙니다.

**해결:** Ollama 앱을 실행한 뒤 `ollama list`로 확인합니다. 연결되지 않으면 `ollama serve`를 실행하고 실습을 다시 시작하세요.

### `model ... not found`

**원인:** 필요한 로컬 모델이 아직 내려받아지지 않았습니다.

**해결:** 아래 두 명령을 실행합니다.

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

### `LangChainDeprecationWarning: PyPDFLoader ...`

**의미:** 오래된 import 방식에 대한 경고이며, 이번 실습의 실행 중단 원인은 아닙니다. PDF 로더를 실제로 사용할 때는 아래처럼 최신 import를 사용합니다.

```python
from langchain_community.document_loaders import PyPDFLoader
```

### `py`를 찾을 수 없음

Python 3.11.9를 다시 설치하고 설치 화면에서 Python Launcher를 포함하세요.

### PowerShell에서 스크립트를 실행할 수 없음

현재 창에서만 아래 명령을 실행한 뒤 다시 시도합니다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 8. 복습 순서 제안

1. 실습 1-1: 로컬 LLM과 벡터 검색으로 기본 RAG 이해
2. 실습 1-2: 검색 문서 관련성, 답변 유용성, 할루시네이션 검증을 연결한 Self-RAG 이해
3. 실습 1-3 ~ 1-4: 웹 검색과 Adaptive RAG 확장
4. 실습 2-1 ~ 2-2: reranker와 hybrid search로 검색 품질 개선
5. 실습 3-1 ~ 3-2: Modular RAG와 GraphRAG로 구조 확장

`no-comments` 버전을 먼저 읽고 막히는 지점에서 주석이 있는 같은 이름의 실습 파일을 비교하면, 코드 흐름을 더 잘 이해할 수 있습니다.
