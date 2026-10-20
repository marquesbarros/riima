param(
    [string]$Python = 'python',
    [string]$OutputDirectory,
    [string[]]$TestArgs = @()
)

$ErrorActionPreference = 'Stop'
# Keep native pytest exit codes available even when the caller opts into PS7 exceptions.
$PSNativeCommandUseErrorActionPreference = $false
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$temporaryRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
$automatic = -not $OutputDirectory
if ($automatic) {
    $runRoot = Join-Path $temporaryRoot ('riima-tests-' + [guid]::NewGuid().ToString('N'))
}
else {
    $runRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
}
$repoPrefix = $repoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if ($runRoot -eq $repoRoot -or $runRoot.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Choose a test output directory outside the source tree.'
}
if (Test-Path -LiteralPath $runRoot) {
    throw 'Choose a new test output directory.'
}
New-Item -ItemType Directory -Path $runRoot, (Join-Path $runRoot 'tmp') | Out-Null
$variables = @('PYTHONDONTWRITEBYTECODE', 'MPLCONFIGDIR', 'TMPDIR', 'TEMP', 'TMP')
$previous = @{}
foreach ($name in $variables) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$previousLocation = Get-Location
try {
    Set-Location -LiteralPath $repoRoot
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:MPLCONFIGDIR = Join-Path $runRoot 'matplotlib'
    $env:TMPDIR = Join-Path $runRoot 'tmp'
    $env:TEMP = $env:TMPDIR
    $env:TMP = $env:TMPDIR
    & $Python -B -m pytest tests -q -p no:cacheprovider "--basetemp=$(Join-Path $runRoot 'pytest')" @TestArgs
    $testExit = $LASTEXITCODE
}
finally {
    Set-Location -LiteralPath $previousLocation.Path
    foreach ($name in $variables) {
        if ($null -eq $previous[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
        }
    }
    if ($automatic) {
        $resolvedRun = [System.IO.Path]::GetFullPath($runRoot)
        $tempPrefix = $temporaryRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        if (-not $resolvedRun.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not ([System.IO.Path]::GetFileName($resolvedRun)).StartsWith('riima-tests-')) {
            throw 'Unexpected temporary directory; refusing recursive removal.'
        }
        Remove-Item -LiteralPath $resolvedRun -Recurse -Force
    }
}
exit $testExit
