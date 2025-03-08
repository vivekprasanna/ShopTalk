import streamlit as st
import json
import os

# Debugging: Show current directory & files
#st.write("Current Directory:", os.getcwd())
#st.write("Files:", os.listdir())

def fetch_data():
    # TBD
    return

# Load product data from a local JSON file
def load_data():
    try: 
        with open("products.json", "r") as file:
            st.success("Loaded JSON Successfully!")
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        

# UI
# Streamlit App Title
st.title("ShopTalk - Your Product Finder Assistant")

# User Input
user_input = st.text_input("Describe what you're looking for:", "")

# Process and Display Results
if user_input:
    # Load Data
    data = load_data()
    st.write(f"**Searching for:** {user_input}")

    # Extract Description and Products
    description = data.get("description", "No description available.")
    products = data.get("products", [])

    # Display Description
    st.subheader("Product Suggestions")
    st.write(description)

    # Display Product List
    if products:
        for product in products:
            title = product.get("title", "No Title")
            image_url = product.get("image", None)
            link = product.get("url", "#")
            details = product.get("details", "No details available.")

            st.subheader(title)
            if image_url:
                st.image(image_url, use_column_width=True)
            st.write(details)
            st.markdown(f"[View Product]({link})")
            st.write("---")
    else:
        st.warning("No matching products found.")
