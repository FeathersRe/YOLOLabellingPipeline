import os
import tempfile
import requests
import streamlit as st
from PIL import Image
from io import BytesIO
from storage import upload_to_storage

GROUNDING_DINO_URL = "http://groundingdino:8001/detect"
GEMINI_URL = "http://gemini2:8002/detect"

MODEL_LINKS = {
    "GroundingDino":GROUNDING_DINO_URL,
    "Gemini2":GEMINI_URL
}

def st_app():
    st.set_page_config(page_title="YOLO Labelling pipeline")
    st.header("🖼️YOLO Labelling pipeline")
    
    option = st.selectbox(
        "Select your labelling base model:",
        ("Gemini2", "GroundingDino")
    )

    user_req = st.text_input("Enter your YOLO requests:")
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
        """ 
            temp_file = tempfile.NamedTemporaryFile(
            "wb", suffix=f".{uploads.type.split('/')[1]}", delete=False
        )
        temp_file.write(uploads.getbuffer())
        image_path = temp_file.name 
        """
        filename = uploads.name
        image_url = upload_to_storage(uploads, filename)


        with st.spinner("Running..."):
            """             
            with open(image_path, "rb") as f:
                files = {"image": (image_path, f, uploads.type)}
                data = {"prompt": user_req}
                response = requests.post(MODEL_LINKS[option], files=files, data=data) 
            """
            data = {
                "prompt" : user_req,
                "image_url" : image_url
            }
            response = requests.post(MODEL_LINKS[option], data=data)

            if response.status_code == 200:
                plotted_img = Image.open(BytesIO(response.content))
                st.image(plotted_img)
            else:
                st.error("Failed to get image from backend.")
                
if __name__ == "__main__":
    st_app()