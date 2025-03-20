import pytest
import subprocess
import time
import requests

FASTAPI_PORT = 8080  # Change as per your app
STREAMLIT_PORT = 8501


@pytest.fixture(scope="session", autouse=True)
def start_servers():
    """Start the FastAPI backend and Streamlit UI before running tests."""
    print("came to fixtures")
    fastapi_process = subprocess.Popen(["python", "app.py"])
    streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit-ui.py"])

    # Wait for the servers to be ready
    time.sleep(10)  # Adjust based on app startup time

    # Check if servers are up
    try:
        print("installing playwright")
        subprocess.run(["playwright", "install", "--with-deps"], check=True)
        print("trying to start servers")
        requests.get(f"http://localhost:{FASTAPI_PORT}", timeout=10)
        requests.get(f"http://localhost:{STREAMLIT_PORT}", timeout=10)
        print("started servers")
    except requests.exceptions.ConnectionError:
        fastapi_process.terminate()
        streamlit_process.terminate()
        pytest.exit("Servers failed to start", returncode=1)

    yield  # Run tests after servers start

    # Teardown: Stop the servers after tests
    fastapi_process.terminate()
    streamlit_process.terminate()