# Quick script to check if Git is available

Write-Host "Checking for Git installation..." -ForegroundColor Cyan
Write-Host ""

$gitCheck = Get-Command git -ErrorAction SilentlyContinue
if ($gitCheck) {
    $gitVersion = git --version
    Write-Host "Git is installed!" -ForegroundColor Green
    Write-Host "  $gitVersion" -ForegroundColor Gray
    Write-Host ""
    Write-Host "You can proceed with uploading to GitHub." -ForegroundColor Green
    Write-Host "See UPLOAD_TO_GITHUB.md for instructions." -ForegroundColor Cyan
} else {
    Write-Host "Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Git:" -ForegroundColor Yellow
    Write-Host "  1. Download from: https://git-scm.com/download/win" -ForegroundColor Cyan
    Write-Host "  2. Run the installer with default settings" -ForegroundColor Cyan
    Write-Host "  3. Restart PowerShell after installation" -ForegroundColor Cyan
    Write-Host "  4. Run this script again to verify" -ForegroundColor Cyan
}

