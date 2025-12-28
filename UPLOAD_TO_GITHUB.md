# Upload KneeXpert to GitHub - Step by Step Guide

## Prerequisites

### 1. Install Git (if not already installed)

**Option A: Download Git for Windows**
- Download from: https://git-scm.com/download/win
- Run the installer with default settings
- Restart your terminal/PowerShell after installation

**Option B: Check if Git is installed but not in PATH**
```powershell
# Try to find git
where.exe git
```

### 2. Create a GitHub Account
- Go to https://github.com and sign up (if you don't have an account)

## Step 1: Initialize Git Repository

Open PowerShell in the project directory and run:

```powershell
cd C:\Users\Administrator\Downloads\KneeXpert

# Initialize git repository
git init

# Configure git (if first time, replace with your info)
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com and sign in
2. Click the **"+"** icon → **"New repository"**
3. Settings:
   - **Name**: `KneeXpert`
   - **Description**: "AI System for Knee Joint Analysis and Diagnosis"
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we already have them)
4. Click **"Create repository"**

## Step 3: Stage and Commit Changes

```powershell
# Check what will be committed
git status

# Add all files (respects .gitignore)
git add .

# Commit changes
git commit -m "Initial commit: KneeXpert training system with fixes

- Fixed CPU device mapping for model loading
- Removed deprecated verbose parameter from scheduler
- Updated training pipeline to handle CPU-only environments
- Added proper error handling for device compatibility"
```

## Step 4: Connect to GitHub and Push

```powershell
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/KneeXpert.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note**: You'll be prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your GitHub password)
  - Create one at: https://github.com/settings/tokens
  - Select scope: `repo` (full control of private repositories)

## Step 5: Share the Trained Model

The trained model files (`.pth`) are **excluded** from git because they're large. Here are options to share them:

### Option A: GitHub Releases (Recommended)
1. Go to your repository on GitHub
2. Click **"Releases"** → **"Create a new release"**
3. Tag version: `v1.0.0`
4. Title: `Initial Trained Model - ResNet50`
5. Upload `models/pretrained/resnet50/best_model.pth` as a release asset
6. Add description with model details

### Option B: Git LFS (for large files)
```powershell
# Install Git LFS (if not installed)
# Download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track .pth files
git lfs track "*.pth"

# Add the model file
git add models/pretrained/resnet50/best_model.pth
git add .gitattributes

# Commit and push
git commit -m "Add trained ResNet50 model"
git push
```

### Option C: External Storage
- Upload to Google Drive, Dropbox, or OneDrive
- Share the link in README.md
- Or use cloud storage services like AWS S3, Azure Blob Storage

## Step 6: Update README with Model Info

Add a section to `README.md` with:
- Model download link (if using releases)
- How to load and use the trained model
- Model performance metrics

## Quick Reference Commands

```powershell
# Check status
git status

# Add changes
git add .

# Commit
git commit -m "Your commit message"

# Push
git push

# View remote
git remote -v

# Pull latest changes
git pull
```

## Troubleshooting

### "git is not recognized"
- Install Git: https://git-scm.com/download/win
- Restart PowerShell after installation

### Authentication failed
- Use Personal Access Token instead of password
- Create token: https://github.com/settings/tokens

### "remote origin already exists"
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/KneeXpert.git
```

### Large file rejected
- Use Git LFS (see Option B above)
- Or exclude from git and use releases/external storage

---

**Ready to upload!** Follow the steps above in order.

