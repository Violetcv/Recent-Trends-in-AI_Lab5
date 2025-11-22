# 🚀 Render Deployment Instructions

Your project is ready to deploy! Follow these steps:

## Step 1: Push to GitHub

```bash
git push -u origin main
```

If you get an authentication error, you have two options:

**Option A: Personal Access Token (Recommended)**
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (full control)
4. Copy the token
5. When prompted for password, paste the token

**Option B: GitHub CLI**
```bash
# Install GitHub CLI (if not installed)
brew install gh

# Authenticate
gh auth login

# Push again
git push -u origin main
```

## Step 2: Create GitHub Repository (if needed)

If the repository doesn't exist yet on GitHub:

```bash
# Using GitHub CLI
gh repo create rtai-air-quality --public --source=. --remote=origin --push

# Or manually:
# 1. Go to https://github.com/new
# 2. Repository name: rtai-air-quality
# 3. Click "Create repository"
# 4. Don't initialize with README
# 5. Run: git push -u origin main
```

## Step 3: Deploy on Render

### 3.1 Sign Up/Login to Render
1. Go to https://render.com
2. Click "Get Started" or "Sign In"
3. Sign in with GitHub (recommended)

### 3.2 Create New Web Service
1. Click "New +" button (top right)
2. Select "Web Service"
3. Click "Connect GitHub" if not already connected
4. Find and select your repository: `rtai-air-quality`
5. Click "Connect"

### 3.3 Configure Service

**Basic Settings:**
- **Name:** `rtai-air-quality` (or any name you prefer)
- **Region:** Oregon (US West) or closest to you
- **Branch:** `main`
- **Root Directory:** (leave empty)
- **Runtime:** Python 3

**Build Settings:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn -b 0.0.0.0:$PORT app:app --timeout 120 --workers 2`

**Instance Type:**
- Select: **Free** (0.1 CPU, 512 MB RAM)

**Advanced Settings (Optional):**
- **Environment Variables:** (click "Add Environment Variable")
  ```
  FLASK_ENV=production
  PYTHON_VERSION=3.11.5
  ```

### 3.4 Deploy!
1. Scroll down and click "Create Web Service"
2. Wait for deployment (5-10 minutes first time)
3. Watch the logs for any errors

## Step 4: Get Your URL

Once deployed, you'll get a URL like:
```
https://rtai-air-quality.onrender.com
```

**Test it:**
1. Open the URL in browser
2. Check the frontend loads
3. Test the Map tab
4. Try making a prediction in ML Model tab

## Step 5: Monitor and Debug

### View Logs
1. Go to your service dashboard on Render
2. Click "Logs" tab
3. Watch for errors

### Common Issues

**Issue 1: Build fails - TensorFlow too large**
```bash
# Solution: Use CPU-only TensorFlow
# Update requirements.txt:
# Replace: tensorflow-macos==2.15.0
# With: tensorflow-cpu==2.15.0
```

**Issue 2: Out of memory**
```bash
# Solution: Reduce model sizes or upgrade plan
# Free tier has only 512 MB RAM
```

**Issue 3: Cold starts (first request slow)**
```bash
# This is normal on free tier
# Service spins down after 15 min inactivity
# First request takes 30-60 seconds to wake up
```

**Issue 4: Models not loading**
```bash
# Check logs for file paths
# Ensure models/ and data/ are in git
# Run: git add -f models/*.pkl models/*.h5 data/*.csv
```

## Step 6: Update Frontend URL (Optional)

If you want the frontend to use the Render URL instead of localhost:

1. Edit `frontend/js/app.js`
2. Change line 1:
```javascript
const API_BASE_URL = 'https://rtai-air-quality.onrender.com';
```
3. Commit and push:
```bash
git add frontend/js/app.js
git commit -m "Update API URL for production"
git push
```
4. Render will auto-deploy the update

## Important Notes

### Free Tier Limitations:
- ⏰ Spins down after 15 min of inactivity
- 🐌 First request after sleep: 30-60 seconds
- 💾 750 hours/month free
- 📦 Storage is ephemeral (resets on restart)

### Keeping Service Awake (Optional):
Use a service like UptimeRobot to ping your URL every 5 minutes:
1. Sign up at https://uptimerobot.com
2. Add new monitor (HTTP)
3. URL: Your Render URL
4. Interval: 5 minutes

## Troubleshooting

### If deployment fails:

1. **Check Render Logs:**
   - Look for Python errors
   - Check if models loaded correctly
   - Verify all dependencies installed

2. **Test Locally:**
```bash
# Test with gunicorn locally
gunicorn -b 0.0.0.0:5001 app:app
```

3. **Verify Files:**
```bash
# Check models and data exist
ls -lh models/
ls -lh data/

# Check git tracked them
git ls-files models/
git ls-files data/
```

4. **Re-deploy:**
   - Go to Render dashboard
   - Click "Manual Deploy" → "Deploy latest commit"

## Success Checklist

- [ ] Code pushed to GitHub
- [ ] Repository is public (or Render has access)
- [ ] Render service created
- [ ] Build completed successfully
- [ ] Service is live (green status)
- [ ] Frontend loads at Render URL
- [ ] API endpoints work
- [ ] Predictions work in ML Model tab
- [ ] Visualizations load correctly

## Share Your Work

Once deployed, share:
1. **GitHub Repository:** https://github.com/Violetcv/rtai-air-quality
2. **Live Demo:** https://rtai-air-quality.onrender.com
3. **Documentation:** Your comprehensive report

## Need Help?

**Render Documentation:**
- https://render.com/docs/web-services
- https://render.com/docs/deploy-flask

**Contact:**
- Render Support: https://render.com/support
- Check logs in Render dashboard
- Review DEPLOYMENT_GUIDE.md for more options

---

**Ready to deploy? Run:** `git push -u origin main` and then follow Step 3! 🚀
