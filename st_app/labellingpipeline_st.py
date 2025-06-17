import os
import tempfile
import requests
import streamlit as st

GROUNDING_DINO_URL = "http://groundingdino:8001/detect"
GEMINI_URL = "http://gemini2:8002/detect"

def st_app():
    st.set_page_config(page_title="Gemini 2.0 YOLO pipeline")
    st.header("🖼️Gemini 2.0 spatial generation pipeline")
    user_req = st.text_input("Enter your YOLO requests")
    run = st.button("Generate")

    with st.sidebar:
        uploads = st.file_uploader(
            accept_multiple_files=False,
            label="Upload image here:",
            type=["jpg", "jpeg", "png"]   
        )

        if uploads:
            with st.expander("View image"):
                st.image(uploads)

    if uploads and run and user_req:
        temp_file = tempfile.NamedTemporaryFile(
            "wb", suffix=f".{uploads.type.split('/')[1]}", delete=False
        )
        temp_file.write(uploads.getbuffer())
        image_path = temp_file.name

        with open(image_path, "rb") as f:
            files = {"image": (image_path, f, uploads.type)}
            data = {"prompt": user_req}

        with st.spinner("Running..."):
            plotted_img = requests.post(GEMINI_URL, files=files, data=data)
        
        os.unlink(image_path)
        st.image(plotted_img)

if __name__ == "__main__":
    st_app()