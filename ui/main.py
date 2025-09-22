import sys
import os
import requests
from PIL import Image
from io import BytesIO
import asyncio
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poster_generator.instagram_agent.graph_builder import build_graph # type: ignore
from poster_generator.instagram_agent.services.poster_image_service import poster_generation
from poster_generator.instagram_agent.state import State
from poster_generator.post_scheduler.suggest import suggest_posting_windows
import streamlit as st # type: ignore
from PIL import Image
import requests
from io import BytesIO
from poster_generator.post_scheduler.model import load_model

# Set up Streamlit interface
st.title("Product Marketing Automation")

# Input fields
brand_name = st.text_input("Brand Name")
product_desc = st.text_area("Product Description")
image_url = st.text_input("Product Image URL")

def load_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

async def generate_caption(brand: str, desc: str, image_url: str) -> str:
    graph = build_graph()
    state: State = State(
            product_description=desc,
            keywords=[],
            generated_caption="",
            final_output="",
            image_url=image_url,
            brand=brand,
        )

    final = await graph.ainvoke(state)
    return final["final_output"]

# Display the product image
if image_url:
    try:
        product_image = load_image_from_url(image_url)
        st.image(product_image, caption="Product Image", use_column_width=True)
    except Exception as e:
        st.error(f"Could not load image: {e}")

# Display the results if input is provided
if st.button("Generate Poster and Caption"):
    if brand_name and product_desc and image_url:
        # Generate poster and caption
        poster_generation(product_desc, brand_name, image_url)

        caption = asyncio.run(generate_caption(brand_name, product_desc, image_url))

        matches = re.findall(r'"([^"]*)"', caption)
        if matches:
            caption = matches[0]
        
        # Display generated poster and caption
        st.image("poster_generator/poster_0.png", caption="Generated Poster", use_column_width=True)
        st.subheader("Generated Caption:")
        st.write(caption)
        
        # Forecast posting time
        model = load_model(path="poster_generator/post_scheduler/posting_model.pkl")
        top_windows = suggest_posting_windows(model, product_desc, top_n=1)
        st.subheader("Suggested Posting Time:")
        st.write(top_windows[0]['window'])
    else:
        st.error("Please provide all inputs (Brand Name, Product Description, and Product Image URL).")