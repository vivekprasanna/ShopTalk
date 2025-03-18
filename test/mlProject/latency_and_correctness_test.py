from playwright.sync_api import sync_playwright
import pytest
from test import test_logger
import time

@pytest.mark.e2e
def test_chatbot_ui_and_latency():
    with sync_playwright() as p:
        try:
            test_logger.info("Starting test_chatbot_ui")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:8501")  # Your Streamlit URL

            page.wait_for_selector("input")
            page.locator("input").fill("find red shoes")

            # Wait for button and click it
            page.wait_for_selector("button:has-text('Search')")
            start_time = time.time()
            page.click("button:has-text('Search')")

            # Wait for the response
            page.wait_for_selector("div.stMarkdown")
            end_time = time.time()
            response_text = page.inner_text("div.stMarkdown")
            test_logger.info(f"Response from chat: {response_text}")
            latency = end_time - start_time
            test_logger.info(f"Latency: {latency}")

            assert "shoes" in response_text
            assert latency < 3.0

            browser.close()
        except Exception as e:
            test_logger.exception(e)