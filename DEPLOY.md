# Deploying OpenLift to the Web 🚀

The easiest way to get this on the open internet is **Streamlit Community Cloud** (it's free and connects directly to GitHub).

## 1. Prerequisites
- [x] Your code is on GitHub.
- [ ] You need a `requirements.txt` file (I will create this for you).

## 2. Limitations (Important!)
**⚠️ Ollama (Local) will NOT work on the cloud.**
- Since the cloud server doesn't have your local Llama 3 model or Ollama installed, the "Ollama" provider will fail.
- **Solution:** You must use **Gemini** (or other API-based providers) for the public version.

## 3. Steps to Deploy
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"New App"**.
3.  Select your GitHub repo (`daramolaworks-create/openlift`).
4.  Set the Main file path to `app.py`.
5.  **Secrets:**
    - If you want to make it easy for users, you can add your generic Google API Key in the "Secrets" settings on Streamlit Cloud, but usually, it's safer to let users enter their own.

6.  Click **Deploy!**

## 4. Updates
Whenever you `git push` to main, the website updates automatically!
