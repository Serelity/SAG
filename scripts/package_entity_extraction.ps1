$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$RepoParent = Resolve-Path "$ProjectRoot\.."
$PackageDir = Join-Path $RepoParent "packages"
$PackagePath = Join-Path $PackageDir "ragflow-learning-plan-entity-extraction.zip"
$StagingDir = Join-Path $PackageDir "_entity_extraction_staging"

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null

if (Test-Path $PackagePath) {
  Remove-Item -LiteralPath $PackagePath -Force
}
if (Test-Path $StagingDir) {
  Remove-Item -LiteralPath $StagingDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

foreach ($Name in @("src", "tests", "configs", "scripts", "docs")) {
  $SourceDir = Join-Path $ProjectRoot $Name
  $DestDir = Join-Path $StagingDir $Name
  New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
  Copy-Item -Path (Join-Path $SourceDir "*") -Destination $DestDir -Recurse -Force
}
foreach ($Name in @("requirements.sag.txt", "requirements.entity.txt")) {
  Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) -Destination (Join-Path $StagingDir $Name) -Force
}

Get-ChildItem -LiteralPath $StagingDir -Recurse -Directory -Filter "__pycache__" |
  Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $StagingDir -Recurse -File |
  Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
  Remove-Item -Force

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
  $StagingDir,
  $PackagePath,
  [System.IO.Compression.CompressionLevel]::Optimal,
  $false
)
Remove-Item -LiteralPath $StagingDir -Recurse -Force
Write-Host "Wrote $PackagePath"
