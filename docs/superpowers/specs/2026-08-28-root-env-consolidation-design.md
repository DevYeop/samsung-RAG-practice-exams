# Root Environment Configuration Consolidation

## Goal

프로젝트 최상위 `.env` 하나에 OpenAI, Tavily, Neo4j 설정을 보관하고, 모든 실습과 Streamlit 앱이 실행 위치와 관계없이 그 파일을 읽도록 한다.

## Architecture

루트에 `project_env.py`를 두고, 각 스크립트는 자신의 파일 위치에서 상위 폴더를 탐색해 이 모듈을 로드한다. `project_env.py`는 프로젝트 루트의 `.env`만 읽고 필수·선택 환경변수 접근 함수를 제공한다. 실행 파일은 하드코딩된 키를 제거하고 이 공통 함수를 사용한다.

## Scope

- 루트 `.env`, `.env.example`, `.gitignore` 추가
- 기존 `no-comments/.env`, `no-comments/.env.example` 제거
- OpenAI, Tavily, Neo4j 키가 직접 입력된 실습 코드 교체
- Streamlit 앱 시작 시 루트 환경을 읽도록 연결하고 `.streamlit/secrets.toml` 제거
- `no-comments` 가이드를 루트 `.env` 기준으로 수정
- API 키 또는 토큰처럼 보이는 문자열을 전체 프로젝트에서 검사

## Constraints

- 실제 키는 어떤 파일에도 기록하지 않는다.
- `.env`는 Git에서 제외하고 `.env.example`만 추적한다.
- `.env`는 기존 셸 환경변수보다 우선한다.
- 키가 없는 실습은 실행 초기에 키 이름과 루트 `.env` 위치를 안내하며 중단한다.
- Neo4j는 선택값이며, 값이 없으면 기존 인메모리 대체 흐름을 유지한다.
- `no-comments` 가상환경과 requirements 설정은 유지하되 `.env` 안내만 루트 기준으로 바꾼다.

## Validation

- 루트 탐색, `.env` 우선 적용, 필수 키 오류를 자동 테스트한다.
- 변경된 Python 파일을 문법 검사한다.
- 루트 전체에서 실제 키 패턴이 남지 않았는지 검사한다.
- 공개 전 Git 스테이징 목록에 `.env`, 가상환경, 캐시·벡터 저장소가 없는지 확인한다.
