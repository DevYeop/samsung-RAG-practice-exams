[CmdletBinding()]
param(
    [switch] $SkipHuggingFaceModel
)

$ErrorActionPreference = 'Stop'
$RequiredPython = '3.11.9'
$RequiredOllamaModels = @('llama3.1', 'nomic-embed-text')
$HuggingFaceModel = 'BAAI/bge-reranker-v2-m3'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvPath = Join-Path $PSScriptRoot '.venv311'
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'
$RequirementsPath = Join-Path $PSScriptRoot 'requirements.txt'
$EnvPath = Join-Path $ProjectRoot '.env'
$EnvExamplePath = Join-Path $ProjectRoot '.env.example'

Import-Module (Join-Path $PSScriptRoot 'SetupTools.psm1') -Force

Write-Host '[1/6] Python 버전을 확인합니다.'
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python Launcher(py)를 찾을 수 없습니다. Python 3.11.9 설치 시 py launcher를 포함한 뒤 다시 실행하세요: https://www.python.org/downloads/release/python-3119/'
}

$DetectedPython = (& py -3.11 -c "import platform; print(platform.python_version())").Trim()
if ($LASTEXITCODE -ne 0 -or -not (Test-ExactPythonVersion -DetectedVersion $DetectedPython -RequiredVersion $RequiredPython)) {
    throw "Python $RequiredPython 버전이 필요합니다. 현재 py -3.11 버전: $DetectedPython"
}

Write-Host '[2/6] 가상환경을 준비합니다.'
if (Test-Path -LiteralPath $VenvPath) {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "기존 $VenvPath 폴더가 정상적인 Windows 가상환경이 아닙니다. 폴더를 백업하거나 삭제한 뒤 다시 실행하세요."
    }

    $VenvVersion = (& $VenvPython -c "import platform; print(platform.python_version())").Trim()
    if ($LASTEXITCODE -ne 0 -or -not (Test-ExactPythonVersion -DetectedVersion $VenvVersion -RequiredVersion $RequiredPython)) {
        throw "기존 .venv311을 실행할 수 없거나 Python $RequiredPython 환경이 아닙니다. 폴더를 백업하거나 삭제한 뒤 다시 실행하세요."
    }
    Write-Host '기존 .venv311을 그대로 사용합니다.'
} else {
    & py -3.11 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw '가상환경 생성에 실패했습니다.' }
}

Write-Host '[3/6] Python 패키지를 설치합니다.'
& $VenvPython -m pip install --upgrade 'pip==26.2.1' 'setuptools==84.0.0'
if ($LASTEXITCODE -ne 0) { throw 'pip 또는 setuptools 설치에 실패했습니다.' }
& $VenvPython -m pip install --requirement $RequirementsPath
if ($LASTEXITCODE -ne 0) { throw 'requirements.txt 설치에 실패했습니다.' }
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Python 패키지 의존성 검사에 실패했습니다.' }

Write-Host '[4/6] .env 파일을 준비합니다.'
if (-not (Test-Path -LiteralPath $EnvPath)) {
    Copy-Item -LiteralPath $EnvExamplePath -Destination $EnvPath
    Write-Host '프로젝트 루트에 .env 파일을 만들었습니다. 필요한 API 키를 입력하세요.'
} else {
    Write-Host '기존 .env 파일을 그대로 사용합니다.'
}

Write-Host '[5/6] Ollama 모델을 확인합니다.'
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw 'Ollama가 설치되어 있지 않습니다. https://ollama.com/download/windows 에서 설치하고 새 PowerShell 창에서 이 스크립트를 다시 실행하세요.'
}

$OllamaList = & ollama list 2>&1
if ($LASTEXITCODE -ne 0) {
    throw 'Ollama에 연결할 수 없습니다. Windows 시작 메뉴에서 Ollama를 실행한 뒤 다시 시도하세요.'
}

$InstalledModels = @(
    $OllamaList |
        Select-Object -Skip 1 |
        ForEach-Object { ($_ -split '\s+')[0] } |
        Where-Object { $_ }
)
$MissingModels = @(Get-MissingOllamaModels -RequiredModels $RequiredOllamaModels -InstalledModels $InstalledModels)

if ($MissingModels.Count -eq 0) {
    Write-Host '필요한 Ollama 모델이 이미 모두 설치되어 있습니다.'
} else {
    foreach ($Model in $MissingModels) {
        Write-Host "없는 모델을 다운로드합니다: $Model"
        & ollama pull $Model
        if ($LASTEXITCODE -ne 0) { throw "Ollama 모델 다운로드에 실패했습니다: $Model" }
    }
}

Write-Host '[6/6] Hugging Face reranker 모델을 확인합니다.'
if ($SkipHuggingFaceModel) {
    Write-Host '요청에 따라 Hugging Face 모델 다운로드를 건너뜁니다.'
} else {
    $HuggingFaceDownloadCode = @"
from huggingface_hub import snapshot_download
model = "$HuggingFaceModel"
try:
    snapshot_download(repo_id=model, local_files_only=True)
    print(f"이미 설치된 모델을 사용합니다: {model}")
except Exception:
    print(f"없는 모델을 다운로드합니다: {model}")
    snapshot_download(repo_id=model)
"@
    & $VenvPython -c $HuggingFaceDownloadCode
    if ($LASTEXITCODE -ne 0) { throw "Hugging Face 모델 준비에 실패했습니다: $HuggingFaceModel" }
}

Write-Host ''
Write-Host '환경 설정이 완료되었습니다.'
Write-Host 'PowerShell 활성화: .\.venv311\Scripts\Activate.ps1'
Write-Host 'Git Bash 활성화: source .venv311/Scripts/activate'
Write-Host '프로젝트 루트의 .env 파일에 실습에 필요한 API 키를 입력한 뒤 Python 파일을 실행하세요.'
