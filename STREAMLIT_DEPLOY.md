# Streamlit Cloud Deployment - Quick Guide

## ✅ Repository Ready for Deployment

Your repository is now ready to deploy on Streamlit Cloud!

## 🚀 Deployment Steps

### Step 1: Go to Streamlit Cloud
Visit: **https://share.streamlit.io/**

### Step 2: Sign In
- Click "Sign in" 
- Use your GitHub account to authenticate

### Step 3: Deploy Your App
1. Click **"New app"** button
2. Fill in the details:
   - **Repository**: `Manishnagda/week-1-and-week-2`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: (optional - auto-generated or custom name)
3. Click **"Deploy!"**

### Step 4: Wait for Build
- First deployment takes 5-10 minutes
- You'll see build logs in real-time
- Once complete, your app will be live!

## 📍 Your App URL
After deployment, your app will be available at:
`https://your-app-name.streamlit.app`

## ✅ What's Included
- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - All dependencies
- ✅ `class_names.txt` - Class labels
- ✅ `model.h5` - Trained CNN model (if included)
- ✅ `.streamlit/config.toml` - Streamlit configuration

## 🔧 Troubleshooting

### If Model File Not Found:
- Check if `model.h5` is in the repository
- If not, you may need to add it manually:
  ```bash
  git add model.h5
  git commit -m "Add model file"
  git push origin main
  ```

### If Build Fails:
- Check the build logs in Streamlit Cloud
- Verify all dependencies in `requirements.txt`
- Ensure Python version is compatible (3.8+)

## 📝 Notes
- Model file size: If > 100MB, GitHub may reject it. Use Git LFS or cloud storage.
- Free tier: Streamlit Cloud free tier has memory limits
- Updates: Any push to main branch will auto-redeploy the app

## 🎉 Success!
Once deployed, share your app URL with anyone to use your Plant Disease Detection system!

