# Streamlit Cloud Deployment Guide

## Prerequisites

1. GitHub account
2. Streamlit Cloud account (free at https://streamlit.io/cloud)
3. Your repository pushed to GitHub: https://github.com/Manishnagda/week-1-and-week-2

## Step-by-Step Deployment

### Step 1: Prepare Your Repository

Make sure your repository has:
- ✅ `app.py` (main Streamlit app)
- ✅ `requirements.txt` (all dependencies)
- ✅ `class_names.txt` (class labels)
- ✅ `model.h5` or `model/plant_disease_model.h5` (trained model)

**Important:** Since model files are large, you have two options:

**Option A: Include model in repository**
- Remove `model.h5` from `.gitignore` temporarily
- Commit and push the model file
- Note: GitHub has a 100MB file limit, so use Git LFS if model is larger

**Option B: Use cloud storage (Recommended)**
- Upload model to Google Drive, Dropbox, or similar
- Modify `app.py` to download model on first run
- This avoids repository size issues

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **New App**
   - Click "New app" button
   - Select your GitHub repository: `Manishnagda/week-1-and-week-2`
   - Set main file path: `app.py`
   - Choose branch: `main`

3. **Configure App**
   - App URL: Will be auto-generated (e.g., `your-app-name.streamlit.app`)
   - Python version: 3.9 or 3.10 (auto-detected)

4. **Deploy**
   - Click "Deploy!"
   - Wait for build to complete (5-10 minutes first time)

### Step 3: Verify Deployment

- Once deployed, your app will be live at: `https://your-app-name.streamlit.app`
- Test by uploading a plant leaf image
- Check logs if there are any errors

## Troubleshooting

### Model Not Found Error

If you see "Model file not found":
- Ensure `model.h5` is in the repository root
- Or update `app.py` to download from cloud storage

### Build Fails

- Check `requirements.txt` has all dependencies
- Verify Python version compatibility
- Check Streamlit Cloud logs for specific errors

### Memory Issues

- Streamlit Cloud free tier has memory limits
- Consider optimizing model size or using model quantization
- Upgrade to paid tier for more resources

## Alternative: Deploy Model Separately

If model file is too large, you can:

1. Upload model to cloud storage (Google Drive, Dropbox, etc.)
2. Get a direct download link
3. Modify `app.py` to download model on startup:

```python
import urllib.request

@st.cache_resource
def download_model():
    model_url = "YOUR_MODEL_DOWNLOAD_LINK"
    model_path = "model.h5"
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return model_path
```

## Support

For issues, check:
- Streamlit Cloud documentation: https://docs.streamlit.io/streamlit-community-cloud
- GitHub repository issues
- Streamlit community forum

