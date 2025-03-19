#!/bin/bash
python3 app.py &  # Run FastAPI/Flask app in background
streamlit run streamlit-ui.py  --server.port 8501 --server.address 0.0.0.0 # Run Streamlit app
