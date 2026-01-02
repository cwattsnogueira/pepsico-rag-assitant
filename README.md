# PepsiCo RAG Assistant (Cloud Run Deployment)

![Cloud Run](https://img.shields.io/badge/Deployed%20to-Google%20Cloud%20Run-blue?logo=googlecloud)

This project deploys a Retrieval-Augmented Generation (RAG) assistant
based on the Pepsi Bottling Group Worldwide Code of Conduct.

##  Deployment Steps

### 1. Upload these files to GitHub:
- app.py
- requirements.txt
- Dockerfile
- .gcloudignore
- PepsiCo_Global_Code_of_Conduct.pdf

### 2. Create a Cloud Build Trigger
- Go to Google Cloud Console → Cloud Build → Triggers
- Connect your GitHub repo
- Select Dockerfile build

### 3. Deploy to Cloud Run
- After Cloud Build finishes, go to Cloud Run
- Click “Create Service”
- Choose the built container image
- Set:
  - Port: 8080
  - Allow unauthenticated access
  - Add environment variable:
    - OPENAI_API_KEY = your_key_here

### 4. Done
Cloud Run will give you a public URL.
