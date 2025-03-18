#!/bin/bash
python3 app.py &  # Run FastAPI/Flask app in background
streamlit run streamlit-ui.py  # Run Streamlit app
