# Выгрузка списка Q-A в читаемый файл (UTF-8 с BOM).
# Запуск из корня проекта:
#   powershell -ExecutionPolicy Bypass -File scripts\qa_list_to_file.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root "venv_py311\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$Out = Join-Path $Root "reports\qa\list.txt"
& $Python (Join-Path $Root "scripts\qa_list_to_file.py") $Out
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Откройте файл: $Out"
