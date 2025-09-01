param(
    [switch]$NoInstall
)

Write-Host "[setup] Ensuring Poetry environment and pre-commit hooks..."

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Warning "Poetry not found. Install via 'winget install Python.Poetry' or 'pipx install poetry'"
    if (-not $NoInstall) { exit 1 }
}

poetry install
if ($LASTEXITCODE -ne 0) { throw "Poetry install failed" }

poetry run pre-commit install
if ($LASTEXITCODE -ne 0) { throw "pre-commit install failed" }

Write-Host "[setup] Done. Use 'poetry run news-insights --help' to see commands."

