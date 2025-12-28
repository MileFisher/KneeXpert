# PowerShell script to upload KneeXpert to GitHub
# Run this script after installing Git and creating a GitHub repository

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "KneeXpert"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "KneeXpert GitHub Upload Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Navigate to project directory
$projectPath = "C:\Users\Administrator\Downloads\KneeXpert"
if (-not (Test-Path $projectPath)) {
    Write-Host "✗ Project directory not found: $projectPath" -ForegroundColor Red
    exit 1
}

Set-Location $projectPath
Write-Host "✓ Working directory: $projectPath" -ForegroundColor Green
Write-Host ""

# Check if git repository is initialized
if (-not (Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
}

Write-Host ""

# Check git status
Write-Host "Checking git status..." -ForegroundColor Yellow
git status --short
Write-Host ""

# Ask for confirmation
$response = Read-Host "Do you want to stage all changes and commit? (y/n)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "Cancelled by user" -ForegroundColor Yellow
    exit 0
}

# Stage all changes
Write-Host ""
Write-Host "Staging all changes..." -ForegroundColor Yellow
git add .
Write-Host "✓ Changes staged" -ForegroundColor Green

# Commit
Write-Host ""
$commitMessage = Read-Host "Enter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMessage)) {
    $commitMessage = "Initial commit: KneeXpert training system with fixes

- Fixed CPU device mapping for model loading
- Removed deprecated verbose parameter from scheduler
- Updated training pipeline to handle CPU-only environments
- Added proper error handling for device compatibility"
}

Write-Host "Committing changes..." -ForegroundColor Yellow
git commit -m $commitMessage
Write-Host "✓ Changes committed" -ForegroundColor Green

# Check if remote exists
Write-Host ""
$remoteExists = git remote | Select-String -Pattern "origin"
if ($remoteExists) {
    Write-Host "Remote 'origin' already exists" -ForegroundColor Yellow
    $remoteUrl = git remote get-url origin
    Write-Host "Current remote URL: $remoteUrl" -ForegroundColor Cyan
    
    $changeRemote = Read-Host "Do you want to change the remote URL? (y/n)"
    if ($changeRemote -eq "y" -or $changeRemote -eq "Y") {
        git remote remove origin
        $remoteExists = $false
    }
}

# Add remote if it doesn't exist
if (-not $remoteExists) {
    Write-Host ""
    Write-Host "Adding remote repository..." -ForegroundColor Yellow
    $remoteUrl = "https://github.com/$GitHubUsername/$RepositoryName.git"
    git remote add origin $remoteUrl
    Write-Host "✓ Remote added: $remoteUrl" -ForegroundColor Green
}

# Rename branch to main
Write-Host ""
Write-Host "Setting branch to 'main'..." -ForegroundColor Yellow
git branch -M main
Write-Host "✓ Branch set to 'main'" -ForegroundColor Green

# Push to GitHub
Write-Host ""
Write-Host "Ready to push to GitHub!" -ForegroundColor Cyan
Write-Host "You will be prompted for credentials:" -ForegroundColor Yellow
Write-Host "  - Username: $GitHubUsername" -ForegroundColor Yellow
Write-Host "  - Password: Use a Personal Access Token (not your GitHub password)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Create token at: https://github.com/settings/tokens" -ForegroundColor Cyan
Write-Host ""

$push = Read-Host "Push to GitHub now? (y/n)"
if ($push -eq "y" -or $push -eq "Y") {
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "✓ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "Repository URL: https://github.com/$GitHubUsername/$RepositoryName" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Note: Trained model files (.pth) are excluded from git." -ForegroundColor Yellow
        Write-Host "Consider uploading them via GitHub Releases or Git LFS." -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "✗ Push failed. Check error messages above." -ForegroundColor Red
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "  - Authentication failed: Use Personal Access Token" -ForegroundColor Yellow
        Write-Host "  - Repository doesn't exist: Create it on GitHub first" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "Skipped push. Run manually with:" -ForegroundColor Yellow
    Write-Host "  git push -u origin main" -ForegroundColor Cyan
}

Write-Host ""

