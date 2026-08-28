function Test-ExactPythonVersion {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string] $DetectedVersion,
        [Parameter(Mandatory)] [string] $RequiredVersion
    )

    return $DetectedVersion.Trim() -eq $RequiredVersion.Trim()
}

function Get-MissingOllamaModels {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)] [string[]] $RequiredModels,
        [Parameter()] [string[]] $InstalledModels = @()
    )

    $installedLookup = @{}
    foreach ($model in $InstalledModels) {
        $installedLookup[$model.Trim().ToLowerInvariant()] = $true
    }

    foreach ($requiredModel in $RequiredModels) {
        $normalized = $requiredModel.Trim().ToLowerInvariant()
        $withDefaultTag = if ($normalized.Contains(':')) { $normalized } else { "${normalized}:latest" }

        if (-not $installedLookup.ContainsKey($normalized) -and -not $installedLookup.ContainsKey($withDefaultTag)) {
            $requiredModel
        }
    }
}


function Get-DependencyInstallPlan {
    [CmdletBinding()]
    param([Parameter(Mandatory)] [ValidateSet('Python','Ollama')] [string] $Dependency)

    if ($Dependency -eq 'Python') {
        return [pscustomobject]@{ WingetId = 'Python.Python.3.11'; DownloadUrl = 'https://www.python.org/downloads/release/python-3119/' }
    }

    return [pscustomobject]@{ WingetId = 'Ollama.Ollama'; DownloadUrl = 'https://ollama.com/download/windows' }
}
Export-ModuleMember -Function Test-ExactPythonVersion, Get-MissingOllamaModels, Get-DependencyInstallPlan
