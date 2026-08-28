$ErrorActionPreference = 'Stop'

$RequiredPython = '3.11.9'
$VenvPath = Join-Path $PSScriptRoot '.venv311'
$RequirementsPath = Join-Path $PSScriptRoot 'requirements.txt'

$PythonReady = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    $DetectedPython = (& py -3.11 -c "import platform; print(platform.python_version())" 2>$null).Trim()
    $PythonReady = ($LASTEXITCODE -eq 0 -and $DetectedPython -eq $RequiredPython)
}
if (-not $PythonReady) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host 'Python 3.11.9가 없어 winget으로 설치합니다.'
        & winget install --id Python.Python.3.11 --exact --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11.9 설치에 실패했습니다.' }
        throw 'Python 3.11.9 설치가 완료되었습니다. 새 PowerShell 창에서 이 스크립트를 다시 실행하세요.'
    }
    Start-Process 'https://www.python.org/downloads/release/python-3119/'
    throw 'Python 3.11.9가 필요합니다. 공식 다운로드 페이지를 열었습니다. 설치 후 새 PowerShell 창에서 다시 실행하세요.'
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
