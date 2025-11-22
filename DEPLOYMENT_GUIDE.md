# Deployment Guide - Air Quality Monitoring System

This guide covers multiple deployment options for your RTAI Lab 5 project, from simple GitHub hosting to full cloud deployment.

---

## Table of Contents
1. [GitHub Repository Setup](#1-github-repository-setup)
2. [Frontend-Only Deployment (GitHub Pages)](#2-frontend-only-deployment-github-pages)
3. [Full Stack Deployment Options](#3-full-stack-deployment-options)
4. [Recommended: Render.com (Free)](#4-recommended-rendercom-free)
5. [Alternative: Railway.app](#5-alternative-railwayapp)
6. [Alternative: Heroku](#6-alternative-heroku)
7. [Alternative: Vercel + Backend](#7-alternative-vercel--backend)
8. [Professional: AWS/GCP/Azure](#8-professional-awsgcpazure)
9. [Docker Deployment](#9-docker-deployment)

---

## 1. GitHub Repository Setup

### Step 1: Initialize Git Repository

```bash
cd "/Users/chhaviverma/Desktop/Semester_7/Recent Trends in AI/RTAI Lab 5"

# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Models (optional - can be large)
# models/*.h5
# models/*.pkl

# Data (optional - might be large)
# data/*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
lstm_training.log

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.env.local
EOF

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Air Quality Monitoring System"
```

### Step 2: Create GitHub Repository

**Option A: Using GitHub CLI**
```bash
# Install GitHub CLI (if not installed)
brew install gh  # macOS

# Login
gh auth login

# Create repository
gh repo create rtai-air-quality --public --source=. --remote=origin --push

# View on GitHub
gh repo view --web
```

**Option B: Manual (Web Interface)**
1. Go to https://github.com/new
2. Repository name: `rtai-air-quality`
3. Description: "Smart City Air Quality Monitoring System with ML/DL models"
4. Public/Private: Choose based on preference
5. Don't initialize with README (you already have one)
6. Click "Create repository"

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/rtai-air-quality.git

# Push code
git branch -M main
git push -u origin main
```

### Step 3: Add README Badges (Optional)

Add to top of README.md:
```markdown
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
```

---

## 2. Frontend-Only Deployment (GitHub Pages)

**Best for:** Static demo, no backend functionality

### Step 1: Modify Frontend for Static Hosting

Create `frontend/config.js`:
```javascript
// Configuration for different environments
const config = {
    development: {
        apiUrl: 'http://localhost:5001'
    },
    production: {
        apiUrl: 'https://your-backend.render.com'  // Update after backend deployment
    }
};

const ENV = window.location.hostname === 'localhost' ? 'development' : 'production';
export const API_URL = config[ENV].apiUrl;
```

Update `frontend/js/app.js` to use config:
```javascript
import { API_URL } from './config.js';

// Replace all 'http://localhost:5001' with API_URL
fetch(`${API_URL}/api/overview`)
```

### Step 2: Deploy to GitHub Pages

```bash
# Create gh-pages branch
git checkout -b gh-pages

# Copy frontend files to root (GitHub Pages serves from root)
cp -r frontend/* .

# Commit
git add .
git commit -m "Deploy to GitHub Pages"

# Push
git push origin gh-pages

# Switch back to main
git checkout main
```

### Step 3: Enable GitHub Pages

1. Go to repository Settings
2. Scroll to "Pages" section
3. Source: `gh-pages` branch
4. Click Save
5. Your site will be at: `https://YOUR_USERNAME.github.io/rtai-air-quality/`

**Note:** Frontend-only deployment won't have backend functionality (predictions, data fetching). You'll need to deploy the backend separately.

---

## 3. Full Stack Deployment Options

For complete functionality (backend + frontend), choose one of these options:

| Platform | Free Tier | Ease | Best For |
|----------|-----------|------|----------|
| **Render.com** | ✅ Yes | ⭐⭐⭐⭐⭐ | Recommended for students |
| **Railway.app** | ✅ 500 hrs/month | ⭐⭐⭐⭐⭐ | Simple, fast deployment |
| **Heroku** | ❌ Paid only | ⭐⭐⭐⭐ | Classic choice |
| **Vercel** | ✅ Frontend only | ⭐⭐⭐⭐ | Great for frontend |
| **PythonAnywhere** | ✅ Limited | ⭐⭐⭐ | Python-focused |
| **AWS/GCP/Azure** | ⚠️ Complex | ⭐⭐ | Professional |

---

## 4. Recommended: Render.com (Free)

**Best option for students - completely free, no credit card required!**

### Prerequisites

1. GitHub account with your code pushed
2. Render.com account (sign up at https://render.com)

### Step 1: Prepare for Render

Create `render.yaml` in project root:
```yaml
services:
  - type: web
    name: rtai-air-quality-api
    env: python
    region: oregon
    plan: free
    branch: main
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: FLASK_ENV
        value: production
```

Update `requirements.txt` - add gunicorn:
```bash
echo "gunicorn==21.2.0" >> requirements.txt
git add requirements.txt render.yaml
git commit -m "Add Render configuration"
git push
```

### Step 2: Deploy on Render

1. **Login to Render:** https://dashboard.render.com
2. **Click "New +" → "Web Service"**
3. **Connect GitHub repository:**
   - Authorize Render to access GitHub
   - Select `rtai-air-quality` repository
4. **Configure Service:**
   - Name: `rtai-air-quality`
   - Region: Oregon (US West)
   - Branch: `main`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -b 0.0.0.0:$PORT app:app`
5. **Advanced Settings:**
   - Instance Type: Free
   - Environment Variables:
     ```
     PYTHON_VERSION=3.11.0
     FLASK_ENV=production
     ```
6. **Click "Create Web Service"**

### Step 3: Wait for Deployment

- Render will build your app (5-10 minutes first time)
- You'll get a URL like: `https://rtai-air-quality.onrender.com`
- Check logs for any errors

### Step 4: Update Frontend

Update `frontend/js/app.js`:
```javascript
// Change API URL
const API_BASE_URL = 'https://rtai-air-quality.onrender.com';
```

**Or** deploy frontend separately on Render:
1. New Static Site
2. Build Command: (none)
3. Publish Directory: `frontend`

### Step 5: Test

Visit: `https://rtai-air-quality.onrender.com`

**⚠️ Important Notes:**
- Free tier spins down after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- Storage is ephemeral (uploaded files/data lost on restart)
- 750 hours/month limit

---

## 5. Alternative: Railway.app

**Pros:** Very simple, automatic deployments, 500 hours/month free  
**Cons:** Requires credit card (won't charge without permission)

### Deployment Steps

1. **Install Railway CLI:**
```bash
npm install -g @railway/cli
# OR
brew install railway
```

2. **Login and Initialize:**
```bash
railway login
railway init
```

3. **Deploy:**
```bash
railway up
```

4. **Add Domain:**
```bash
railway domain
```

That's it! Railway automatically detects Python and Flask.

**Configuration (optional)** - Create `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "gunicorn app:app",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

---

## 6. Alternative: Heroku

**⚠️ Note:** Heroku no longer has free tier (starts at $5/month)

### Prerequisites

1. Heroku account: https://heroku.com
2. Heroku CLI: `brew install heroku/brew/heroku`

### Deployment Steps

```bash
# Login
heroku login

# Create app
heroku create rtai-air-quality

# Add Procfile
cat > Procfile << 'EOF'
web: gunicorn app:app
EOF

# Add runtime
cat > runtime.txt << 'EOF'
python-3.11.5
EOF

# Commit
git add Procfile runtime.txt
git commit -m "Add Heroku configuration"

# Deploy
git push heroku main

# Open app
heroku open

# View logs
heroku logs --tail
```

### Heroku Add-ons (Optional)

```bash
# PostgreSQL database
heroku addons:create heroku-postgresql:mini

# Redis caching
heroku addons:create heroku-redis:mini

# Scheduler for periodic tasks
heroku addons:create scheduler:standard
```

---

## 7. Alternative: Vercel + Backend

**Best for:** Frontend deployment (Vercel) + separate backend

### Frontend on Vercel

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Deploy Frontend:**
```bash
cd frontend
vercel

# Follow prompts:
# - Project name: rtai-air-quality-frontend
# - Framework: Other
# - Build command: (none)
# - Output directory: .
```

3. **Get URL:** `https://rtai-air-quality-frontend.vercel.app`

### Backend on Render/Railway

Deploy backend separately (see sections above) and update frontend API URL.

---

## 8. Professional: AWS/GCP/Azure

**For production-grade deployment with scalability**

### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.11 rtai-air-quality

# Create environment
eb create rtai-air-quality-env

# Deploy
eb deploy

# Open
eb open
```

### Google Cloud Run

```bash
# Install gcloud
brew install google-cloud-sdk

# Login
gcloud auth login

# Deploy
gcloud run deploy rtai-air-quality \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure App Service

```bash
# Install Azure CLI
brew install azure-cli

# Login
az login

# Create resource group
az group create --name rtai-rg --location eastus

# Create app service plan
az appservice plan create --name rtai-plan --resource-group rtai-rg --sku FREE

# Create web app
az webapp up --name rtai-air-quality --resource-group rtai-rg --plan rtai-plan
```

---

## 9. Docker Deployment

**For consistent deployment across any platform**

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5001

# Environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["gunicorn", "-b", "0.0.0.0:5001", "-w", "4", "app:app"]
```

### Step 2: Create .dockerignore

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.git
.gitignore
*.log
.DS_Store
```

### Step 3: Build and Run

```bash
# Build image
docker build -t rtai-air-quality .

# Run container
docker run -p 5001:5001 rtai-air-quality

# Test
curl http://localhost:5001/api/status
```

### Step 4: Push to Docker Hub

```bash
# Login
docker login

# Tag image
docker tag rtai-air-quality YOUR_USERNAME/rtai-air-quality:latest

# Push
docker push YOUR_USERNAME/rtai-air-quality:latest
```

### Deploy Docker Container

**On any platform that supports Docker:**

```bash
# Pull and run
docker pull YOUR_USERNAME/rtai-air-quality:latest
docker run -d -p 80:5001 YOUR_USERNAME/rtai-air-quality:latest
```

---

## 10. Configuration for Production

### Environment Variables

Create `.env` file (don't commit this):
```bash
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Update app.py for Production

```python
import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# Configuration
app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False') == 'True'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# CORS configuration
if app.config['DEBUG']:
    CORS(app)  # Allow all origins in development
else:
    CORS(app, origins=['https://your-frontend-domain.com'])  # Restrict in production

# ... rest of your code ...

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=app.config['DEBUG'])
```

---

## 11. Post-Deployment Checklist

### Testing

- [ ] Test all API endpoints
- [ ] Test frontend functionality
- [ ] Test predictions
- [ ] Test visualizations loading
- [ ] Test on mobile devices
- [ ] Test in different browsers

### Security

- [ ] Add rate limiting
- [ ] Add API authentication (if needed)
- [ ] Set up HTTPS (most platforms do this automatically)
- [ ] Sanitize user inputs
- [ ] Add CORS restrictions

### Monitoring

- [ ] Set up error tracking (Sentry)
- [ ] Monitor uptime (UptimeRobot)
- [ ] Check logs regularly
- [ ] Set up alerts for failures

### Performance

- [ ] Enable caching
- [ ] Compress responses
- [ ] Optimize images
- [ ] Minify JavaScript/CSS
- [ ] Use CDN for static assets

---

## 12. Recommended Deployment Strategy

### For Lab Submission (Quick & Free):

**Option 1: Render.com (Recommended)**
```bash
# 1. Push to GitHub
git push origin main

# 2. Deploy on Render (web interface)
# - Connect GitHub
# - Auto-deploy from main branch
# - Get URL in 5 minutes

# 3. Share the URL
https://rtai-air-quality.onrender.com
```

**Option 2: GitHub + Static Demo**
```bash
# 1. Deploy frontend to GitHub Pages
git subtree push --prefix frontend origin gh-pages

# 2. Note: Backend features won't work
# 3. Good for visual demonstration
```

### For Production (Scalable):

**AWS/GCP/Azure + Docker**
- Use Docker for consistency
- Set up CI/CD pipeline
- Add monitoring and logging
- Configure auto-scaling

---

## 13. Troubleshooting

### Common Issues

**1. Port Binding Error**
```bash
# Solution: Use environment variable
port = int(os.getenv('PORT', 5001))
app.run(host='0.0.0.0', port=port)
```

**2. TensorFlow Too Large**
```bash
# Solution: Use smaller model or CPU-only TensorFlow
pip install tensorflow-cpu
```

**3. Build Timeout**
```bash
# Solution: Reduce model sizes or use pre-trained models
# Or exclude models/ from deployment and load from S3/GCS
```

**4. CORS Errors**
```bash
# Solution: Install flask-cors
pip install flask-cors
# Add CORS(app) in app.py
```

**5. Static Files Not Found**
```bash
# Solution: Check paths in app.py
app = Flask(__name__, static_folder='frontend', static_url_path='')
```

---

## 14. Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| **Render** | 750 hrs/month | $7/month | Students, portfolios |
| **Railway** | 500 hrs/month | $5/month | Simple projects |
| **Vercel** | Unlimited | $20/month | Frontend apps |
| **Heroku** | None | $5-7/month | Established apps |
| **AWS** | 750 hrs/month (1 year) | Pay-as-you-go | Scalable apps |
| **GCP** | $300 credit (90 days) | Pay-as-you-go | Data-heavy apps |
| **Azure** | $200 credit (30 days) | Pay-as-you-go | Enterprise apps |

---

## 15. Quick Start Commands

### GitHub Only (Code Hosting)
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/rtai-air-quality.git
git push -u origin main
```

### Render Deployment (Full Stack)
```bash
# 1. Push to GitHub (above)
# 2. Go to https://render.com
# 3. New Web Service → Connect GitHub repo
# 4. Deploy (automatic)
```

### Railway Deployment
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### Docker Deployment
```bash
docker build -t rtai-air-quality .
docker run -p 5001:5001 rtai-air-quality
```

---

## 16. Final Recommendations

### For RTAI Lab Submission:

**Best Choice: Render.com**
- ✅ Completely free
- ✅ No credit card required
- ✅ Easy setup (5 minutes)
- ✅ Automatic HTTPS
- ✅ Auto-deploy from GitHub
- ✅ Suitable for portfolios

**Steps:**
1. Push code to GitHub
2. Sign up on Render.com
3. Connect GitHub repository
4. Click "Deploy"
5. Share the URL in submission

### For Future Professional Use:

**Best Choice: AWS/GCP with Docker**
- Scalable
- Professional-grade
- CI/CD integration
- Monitoring tools
- High availability

---

## Need Help?

**Common Questions:**

1. **Q: Which deployment should I use?**  
   A: Render.com for free hosting, GitHub for code submission

2. **Q: Do I need to change code for deployment?**  
   A: Minimal changes - mostly environment variables and gunicorn

3. **Q: Will my models work on free tier?**  
   A: Yes, but they'll be slow on first request (cold start)

4. **Q: Can I deploy frontend and backend separately?**  
   A: Yes! Frontend on Vercel/GitHub Pages, Backend on Render

5. **Q: How to update after deployment?**  
   A: Push to GitHub, Render auto-deploys (if connected)

---

**End of Deployment Guide**

Good luck with your deployment! 🚀
