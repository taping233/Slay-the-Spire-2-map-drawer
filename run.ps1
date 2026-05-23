param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$localPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (Test-Path $localPython) {
  & $localPython -c "import PIL" 2>$null
  if ($LASTEXITCODE -eq 0) {
    & $localPython -m sts2_drawer.cli @Args
    exit $LASTEXITCODE
  }
}

if (Test-Path $bundledPython) {
  & $bundledPython -m sts2_drawer.cli @Args
  exit $LASTEXITCODE
}

python -m sts2_drawer.cli @Args
exit $LASTEXITCODE
