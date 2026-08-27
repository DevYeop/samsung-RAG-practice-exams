# Root Environment Configuration Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store all project API settings in one root `.env` and load it reliably from every practice script.

**Architecture:** A root `project_env.py` finds its sibling `.env`, loads it with override enabled, and exposes `require_env` and `optional_env`. Script-local bootstrap code adds the root to `sys.path` before importing the helper, avoiding assumptions about the terminal's current directory.

**Tech Stack:** Python 3.11, python-dotenv, unittest, PowerShell, Streamlit.

**Spec:** `docs/superpowers/specs/2026-08-28-root-env-consolidation-design.md`

## Global Constraints

- Use only the root `.env`; never commit real credentials.
- Root `.env` overrides inherited shell environment values.
- Preserve existing Neo4j optional fallback behavior.
- Validate all changed Python source with `compile()` and run tests before committing.

---

### Task 1: Root environment helper

**Files:**
- Create: `project_env.py`
- Create: `tests/test_project_env.py`
- Create: `.env.example`
- Create: `.gitignore`

**Interfaces:**
- Produces: `load_project_env() -> Path`, `require_env(name: str) -> str`, `optional_env(name: str, default=None) -> str | None`.

- [ ] **Step 1: Write failing tests**

```python
load_project_env(env_file)
assert require_env("OPENAI_API_KEY") == "root-value"
with self.assertRaisesRegex(RuntimeError, r"root.*\.env.*TAVILY_API_KEY"):
    require_env("TAVILY_API_KEY")
```

- [ ] **Step 2: Run tests and verify they fail because `project_env` is absent**

- [ ] **Step 3: Implement the three helper functions and root `.env` templates**

```python
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(env_file or ROOT_DIR / ".env", override=True)
```

- [ ] **Step 4: Run tests and verify they pass**

### Task 2: Replace direct credentials

**Files:**
- Modify: Python scripts with OpenAI, Tavily, or Neo4j assignments under `번외-랭체인과 RAG 기초`, `삼성-rag-1일차`, and `삼성-rag-2일차`
- Modify: `삼성-rag-2일차/최종프로젝트-삼성-RAG-커스텀-챗봇/samsung-rag-custom_chatbot-exam/main.py`
- Delete: `삼성-rag-1일차/no-comments/.env`
- Delete: `삼성-rag-1일차/no-comments/.env.example`
- Delete: `삼성-rag-2일차/최종프로젝트-삼성-RAG-커스텀-챗봇/samsung-rag-custom_chatbot-exam/.streamlit/secrets.toml`

**Interfaces:**
- Consumes: `project_env.load_project_env`, `project_env.require_env`, `project_env.optional_env`.

- [ ] **Step 1: Add a root-path bootstrap before each helper import**

```python
ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "project_env.py").is_file())
sys.path.insert(0, str(ROOT_DIR))
```

- [ ] **Step 2: Replace each hardcoded key with `load_project_env()` and the appropriate getter**

- [ ] **Step 3: Preserve optional Neo4j and existing Streamlit behavior**

- [ ] **Step 4: Run syntax checks for every changed Python file**

### Task 3: Documentation and public-repository verification

**Files:**
- Modify: `삼성-rag-1일차/no-comments/환경설정_가이드.md`
- Modify: `삼성-rag-1일차/no-comments/setup-windows.ps1`
- Test: `tests/test_project_env.py`

- [ ] **Step 1: Update the beginner guide to edit the root `.env`**
- [ ] **Step 2: Ensure setup creates the root `.env` from its example when missing**
- [ ] **Step 3: Run all tests, source syntax checks, and root secret scan**
- [ ] **Step 4: Initialize Git, inspect the staging set, and make the first commit**
