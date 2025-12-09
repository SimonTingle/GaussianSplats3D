#!/bin/bash

# --- CONFIGURATION ---
REPO_URL="https://github.com/SimonTingle/GaussianSplats3D.git"
BRANCH="main"

echo "========================================================"
echo "🚀 INITIALIZING BACKEND GIT DEPLOYMENT"
echo "========================================================"

# 1. Initialize Git (Safe to run even if already initialized)
if [ -d ".git" ]; then
    echo "ℹ️  Git already initialized. Re-initializing..."
fi
git init -b $BRANCH

# 2. Create .gitignore (CRITICAL STEP)
# We MUST ignore 'ckpts' because GitHub rejects files larger than 100MB.
# Your ckpts folder is ~20GB. Users will download it via the README instructions.
echo "📄 Creating .gitignore to exclude massive model files..."
cat > .gitignore <<EOF
# Ignore massive model weights (Users download these via script)
ckpts/
backend/ckpts/

# Ignore Python cache and compiled files
__pycache__/
*.pyc
*.pyo
*.pyd

# Ignore environment folders
.env
venv/
env/
.venv/

# Ignore IDE settings
.vscode/
.idea/

# Ignore system files
.DS_Store
EOF

# 3. Add the Remote Repository
# Remove origin if it exists to ensure we use the correct URL
if git remote | grep -q "^origin$"; then
    echo "🔄 Removing existing remote 'origin'..."
    git remote remove origin
fi
echo "🔗 Adding remote origin: $REPO_URL"
git remote add origin $REPO_URL

# 4. Stage and Commit Files
echo "📦 Staging files..."
git add .

echo "📝 Committing files..."
git commit -m "Deploy: Full backend setup for Lightning AI (Fixed nvdiffrast/Kaolin/FlashAttn/Server)"

# 5. Push to GitHub
echo "========================================================"
echo "🚀 READY TO PUSH TO GITHUB"
echo "========================================================"
echo "You will now be asked for your GitHub credentials."
echo "Username: SimonTingle"
echo "Password: [YOUR_PERSONAL_ACCESS_TOKEN] (Not your login password)"
echo "--------------------------------------------------------"

# Attempt push
git push -u origin $BRANCH --force

echo ""
echo "✅ DONE! Check your repo at: $REPO_URL"