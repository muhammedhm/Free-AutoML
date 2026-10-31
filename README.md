# Free AutoML

## Install
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Run (two terminals, both from this folder)
    uvicorn api.main:app --port 8000          # backend  -> http://localhost:8000/docs
    streamlit run ui/app.py                   # UI       -> http://localhost:8501


Data and models are stored in ./storage (override with AUTOML_HOME).
The UI looks for the backend at http://localhost:8000 (override with AUTOML_API_URL).
