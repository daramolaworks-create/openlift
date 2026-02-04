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

Whenever you `git push` to main, the website updates automatically!

## 5. Option 2: Self-Hosting (VPS / Docker)

If you have a personal hosting plan (like DigitalOcean, AWS, Linode, or a dedicated server), you can use **Docker**.

### Steps:
1.  **Clone the repo** on your server:
    ```bash
    git clone https://github.com/daramolaworks-create/openlift.git
    cd openlift
    ```
2.  **Run with Docker Compose**:
    ```bash
    docker compose up -d
    ```
    The app will start on port `8501`.

### Custom Domain (Nginx Proxy)
To connect a "proper domain" (e.g., `openlift.ai`):
1.  Point your domain's **A Record** to your server's IP address.
2.  Use **Nginx** as a reverse proxy.
    Example Nginx Config:
    ```nginx
    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://localhost:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
    ```
3.  Use **Certbot** for free HTTPS: `sudo certbot --nginx`
