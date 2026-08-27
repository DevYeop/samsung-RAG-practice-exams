$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Import-Module (Join-Path $ProjectRoot 'SetupTools.psm1') -Force

function Assert-Equal {
    param(
        [Parameter(Mandatory)] [AllowEmptyCollection()] $Expected,
        [Parameter(Mandatory)] [AllowNull()] [AllowEmptyCollection()] $Actual,
        [Parameter(Mandatory)] [string] $Message
    )

    if ((@($Expected) -join '|') -ne (@($Actual) -join '|')) {
        throw "$Message`nExpected: $($Expected -join ', ')`nActual: $($Actual -join ', ')"
    }
}

if (-not (Test-ExactPythonVersion -DetectedVersion '3.11.9' -RequiredVersion '3.11.9')) {
    throw '정확한 Python 버전을 허용해야 합니다.'
}

if (Test-ExactPythonVersion -DetectedVersion '3.11.8' -RequiredVersion '3.11.9') {
    throw '다른 Python 패치 버전을 허용하면 안 됩니다.'
}

$requiredModels = @('llama3.1', 'nomic-embed-text')
$installedModels = @('llama3.1:latest', 'other-model:latest')
$missingModels = Get-MissingOllamaModels -RequiredModels $requiredModels -InstalledModels $installedModels

Assert-Equal -Expected @('nomic-embed-text') -Actual $missingModels -Message '설치되지 않은 모델만 반환해야 합니다.'

$noneMissing = Get-MissingOllamaModels `
    -RequiredModels $requiredModels `
    -InstalledModels @('llama3.1:latest', 'nomic-embed-text:latest')

Assert-Equal -Expected @() -Actual $noneMissing -Message '모든 모델이 있으면 다운로드 대상이 없어야 합니다.'

Write-Output 'setup tools tests: PASS'
