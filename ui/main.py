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
from poster_generator.post_scheduler.model import load_model

from qna.data_loader import load_data
from qna.embedder import build_index
from qna.pipeline import generate_final_reply

import streamlit as st # type: ignore

# -----------------------------
# Theme Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #28a745;
        color: #FFFFFF;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helper Functions
# -----------------------------
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

# -----------------------------
# Sidebar Navigation
# -----------------------------
with st.sidebar:
    st.header("Menu")
    menu = st.radio("Choose Option", ["Poster Generation", "QandA"])

# -----------------------------
# Poster Generation Interface
# -----------------------------
if menu == "Poster Generation":
    st.title("Product Marketing Automation")

    # Inputs
    brand_name = st.text_input("Brand Name")
    product_desc = st.text_area("Product Description")
    image_url = st.text_input("Product Image URL")

    # Show product image preview if URL provided
    if image_url:
        try:
            product_image = load_image_from_url(image_url)
            resized_preview = product_image.resize((300, 400))
            st.image(resized_preview, caption="Product Image Preview")
        except Exception as e:
            st.error(f"Could not load image: {e}")

    # Generate poster button
    if st.button("Generate Poster and Caption"):
        if brand_name and product_desc and image_url:
            with st.spinner("Generating poster and caption..."):
                # Generate poster
                poster_generation(product_desc, brand_name, image_url)
                
                # Generate caption
                caption = asyncio.run(generate_caption(brand_name, product_desc, image_url))
            
            matches = re.findall(r'"([^"]*)"', caption)
            if matches:
                caption = matches[0]

            poster_image = Image.open("poster_generator/poster_0.png")
            resized_poster = poster_image.resize((400, 500))
            st.image(resized_poster, caption="Generated Poster")

            st.subheader("Generated Caption:")
            st.write(caption)

            model = load_model(path="poster_generator/post_scheduler/posting_model.pkl")
            top_windows = suggest_posting_windows(model, product_desc, top_n=1)
            st.subheader("Suggested Posting Time:")
            st.write(top_windows[0]['window'])
        else:
            st.error("Please provide all inputs (Brand Name, Product Description, and Product Image URL).")

# -----------------------------
# QandA Interface
# -----------------------------
elif menu == "QandA":
    st.title("QandA Interface")
    st.markdown("### 1️⃣ Upload Product Catalog")

    csv_path = "products_with_description.csv"

    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "embedder" not in st.session_state:
        st.session_state.embedder = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "uploaded_csv_name" not in st.session_state:
        st.session_state.uploaded_csv_name = None

    # Upload CSV
    uploaded_file = st.file_uploader("", type=["csv"])

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_csv_name:
            # Save file
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ File saved to {csv_path} and ready for indexing.")

            # Build FAISS index
            with st.spinner("Building FAISS index..."):
                df = load_data(csv_path)
                embedder, index = build_index(df)

            # Save to session
            st.session_state.df = df
            st.session_state.embedder = embedder
            st.session_state.index = index
            st.session_state.uploaded_csv_name = uploaded_file.name
        else:
            st.info("CSV already uploaded and indexed. You can ask questions below.")

    # -----------------------------
    # Display first 5 rows always if dataframe exists
    # -----------------------------
    if st.session_state.df is not None:
        st.markdown("### Preview: First 5 Rows of the Uploaded CSV")
        st.dataframe(st.session_state.df.head())

    # Question answering
    st.markdown("### 2️⃣ Ask a question")
    user_question = st.text_input("Your question about the uploaded CSV:")

    if st.button("Get Answer"):
        if st.session_state.df is None:
            st.error("Please upload the product CSV first.")
        elif not user_question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Retrieving answer from RAG model..."):
                answer = generate_final_reply(
                    user_question,
                    st.session_state.embedder,
                    st.session_state.index,
                    st.session_state.df
                )
            st.subheader("Answer:")
            st.write(answer)