import io
import tempfile
import requests
from io import BytesIO
from flask import Flask, request, send_file
from langchain_client_backend import config_client, get_response
from image_utils import resize_img, plot_bounding_boxes

app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect():
    image_url = request.form.get("image_url")
    prompt = request.form.get("prompt")

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = response.content
    except:
        return "Error: Failed to download image!"
    
    im = resize_img(image_bytes)
    
    llm_client = config_client()
    bounding_boxes = get_response(llm_client, im, prompt)
    plotted_img = plot_bounding_boxes(im, bounding_boxes)

    img_io = BytesIO()
    plotted_img.save(img_io, format='PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)