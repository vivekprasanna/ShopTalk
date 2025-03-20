from playwright.sync_api import sync_playwright
import pytest
from test import test_logger
import time

test_cases = [
    ("find red shoes", "shoes", 3.0),
    ("find blue jeans", "jeans", 3.0),
    ("find black jackets", "jackets", 3.0),
    ("find phone cases for iphone", "phone case", 3.0),
    ("find white sneakers", "sneakers", 3.0),
    ("find groceries", "groceries", 3.0),
]

@pytest.mark.e2e
@pytest.mark.parametrize("input_query, expected_text, max_latency", test_cases)
def test_chatbot_ui_and_latency(input_query, expected_text, max_latency):
    with sync_playwright() as p:
        try:
            test_logger.info(f"Starting UI test for query: {input_query}")
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("http://localhost:8501")  # Your Streamlit URL

            page.wait_for_selector("input")
            page.locator("input").fill(input_query)

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

            # Assertions
            assert expected_text in response_text, f"Expected '{expected_text}' in response but got '{response_text}'"
            assert latency < max_latency, f"Latency exceeded {max_latency} seconds"


            browser.close()
        except Exception as e:
            test_logger.exception(e)