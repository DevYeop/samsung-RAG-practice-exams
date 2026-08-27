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

Export-ModuleMember -Function Test-ExactPythonVersion, Get-MissingOllamaModels
