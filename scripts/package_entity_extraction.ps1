$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
$RepoParent = Split-Path -Parent $ProjectRoot
$PackageDir = if ($env:PACKAGE_DIR) { $env:PACKAGE_DIR } else { Join-Path $ProjectRoot "packages" }
$PackagePath = Join-Path $PackageDir "sag-qwen3-4b-semantic-extraction.zip"
$StagingDir = Join-Path $PackageDir "_semantic_extraction_staging"

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
Remove-Item -LiteralPath $PackagePath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $StagingDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

try {
  foreach ($Name in @("src", "tests", "configs", "scripts")) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) -Destination (Join-Path $StagingDir $Name) -Recurse -Force
  }
  Remove-Item -LiteralPath (Join-Path $StagingDir "tests\fixtures") -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path (Join-Path $StagingDir "docs") | Out-Null
  Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs\13-Qwen4B工单级语义抽取.md") -Destination (Join-Path $StagingDir "docs") -Force
  foreach ($Name in @("requirements.sag.txt", "requirements.entity.txt", "requirements.vllm.txt")) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) -Destination $StagingDir -Force
  }

  Get-ChildItem $StagingDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
  Get-ChildItem $StagingDir -Recurse -File | Where-Object { $_.Extension -in @(".pyc", ".pyo") } | Remove-Item -Force

  $Forbidden = Get-ChildItem $StagingDir -Recurse -Force | Where-Object {
    $_.FullName -match '[\\/](data|outputs|models|packages|\.superpowers|\.pi-subagents)([\\/]|$)' -or
    (-not $_.PSIsContainer -and ($_.Name -like '.env*' -or $_.Extension -in @('.jsonl', '.duckdb', '.zip', '.npy', '.key', '.pem', '.tsv', '.csv')))
  }
  if ($Forbidden) {
    throw "Forbidden package content: $($Forbidden.FullName -join ', ')"
  }

  $Commit = (git -C $ProjectRoot rev-parse HEAD).Trim()
  $Files = Get-ChildItem $StagingDir -Recurse -File | Sort-Object FullName
  $Manifest = @{
    package_commit = $Commit
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    files = @($Files | ForEach-Object {
      @{
        path = $_.FullName.Substring($StagingDir.Length + 1).Replace('\', '/')
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
      }
    })
  }
  $Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $StagingDir "MANIFEST.json") -Encoding UTF8

  Compress-Archive -Path (Join-Path $StagingDir "*") -DestinationPath $PackagePath -CompressionLevel Optimal
  $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PackagePath).Hash.ToLowerInvariant()
  Write-Host "Wrote $PackagePath"
  Write-Host "SHA256 $Hash"
}
finally {
  Remove-Item -LiteralPath $StagingDir -Recurse -Force -ErrorAction SilentlyContinue
}
