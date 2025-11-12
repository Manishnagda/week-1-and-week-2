# 🚀 Streamlit Cloud Deployment - Step by Step

## ✅ Pre-Deployment Checklist

All files are ready:
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - All dependencies listed
- ✅ `class_names.txt` - Class labels
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ Repository: https://github.com/Manishnagda/week-1-and-week-2

## 📋 Deployment Steps

### Step 1: Open Streamlit Cloud
👉 Go to: **https://share.streamlit.io/**

### Step 2: Sign In
- Click **"Sign in"** button
- Select **"Continue with GitHub"**
- Authorize Streamlit Cloud to access your GitHub account

### Step 3: Create New App
- Click **"New app"** button (top right or center)

### Step 4: Configure Deployment
Fill in the form:
- **Repository**: Select `Manishnagda/week-1-and-week-2`
- **Branch**: `main`
- **Main file path**: `app.py`
- **App URL**: (Optional - leave default or enter custom name like `plant-disease-detector`)

### Step 5: Deploy
- Click **"Deploy!"** button
- Wait for build to complete (5-10 minutes for first deployment)

### Step 6: Access Your App
Once deployment is complete, you'll see:
- ✅ Status: "Running"
- 🌐 Your app URL: `https://your-app-name.streamlit.app`

## 🎯 Expected Result

After successful deployment:
- App will be live and accessible via the URL
- You can upload plant leaf images
- Model will make predictions
- Results will display with confidence scores

## ⚠️ Important Notes

### Model File
- If `model.h5` is not in the repository, you may see a warning
- To include it, run:
  ```bash
  git add -f model.h5
  git commit -m "Add model file"
  git push origin main
  ```
- Then redeploy on Streamlit Cloud (it auto-redeploys on push)

### Build Time
- First deployment: 5-10 minutes
- Subsequent updates: 2-5 minutes (auto-redeploy on git push)

### Troubleshooting
If deployment fails:
1. Check build logs in Streamlit Cloud dashboard
2. Verify all dependencies in `requirements.txt`
3. Ensure `app.py` has no syntax errors
4. Check if model.h5 is accessible

## 🎉 Success!

Once deployed, your Plant Disease Detection app will be live and shareable!

**Your app URL will be**: `https://your-app-name.streamlit.app`

---

**Need help?** Check the build logs in Streamlit Cloud dashboard for specific errors.

