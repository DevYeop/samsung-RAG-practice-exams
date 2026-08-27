$ErrorActionPreference = 'Stop'

$RequiredPython = '3.11.9'
$VenvPath = Join-Path $PSScriptRoot '.venv311'
$RequirementsPath = Join-Path $PSScriptRoot 'requirements.txt'

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python Launcher(py)가 없습니다. Python 3.11.9를 설치한 뒤 다시 실행하세요.'
}

$DetectedPython = (& py -3.11 -c "import platform; print(platform.python_version())").Trim()
if ($DetectedPython -ne $RequiredPython) {
    throw "Python $RequiredPython 버전이 필요하지만 $DetectedPython 버전이 감지되었습니다."
}

if (Test-Path -LiteralPath $VenvPath) {
    throw "이미 $VenvPath 폴더가 있습니다. 기존 폴더를 백업하거나 삭제한 뒤 다시 실행하세요."
}

& py -3.11 -m venv $VenvPath
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'

& $VenvPython -m pip install --upgrade 'pip==26.2.1' 'setuptools==84.0.0'
& $VenvPython -m pip install --requirement $RequirementsPath
& $VenvPython -m pip check

Write-Host ''
Write-Host '환경 구축 완료'
& $VenvPython --version
Write-Host '활성화: .\.venv311\Scripts\Activate.ps1'
