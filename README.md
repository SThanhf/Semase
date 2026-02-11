# Semase 
Semase is an application that uses semantic search and Azure AI Search service to search relevant document chunks.

# Project Structure
```

Semase/
├── src/            # FastAPI backend connection and index setup for Azure AI search
├── app_ui.py       # Streamlit frontend
├── requirements.txt
└── README.md
```

---
## 1) Prerequisites
- Python 3.10+
- Download package requirements through pip
- Create `.env` from template and fill keys, values and connections.
- Azure Supcription with: 
    + Azure AI Search
    + Azure AI Service with text embedding model
    + Azure Blob Storage

## 2) Run app
Start FastAPI backend with Uvicorn 
- uvicorn main:app --host 0.0.0.0 --port 8000

Then open another shell/cmd and start Streamlit frontend
- streamlit run app_ui.py


