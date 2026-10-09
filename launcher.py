import os
import requests
import streamlit as st

MODEL_URL = (
    "https://github.com/saivaishnavidevi/weed-management/"
    "releases/download/v1.0.0/improved_weed_unet_final.keras"
)

MODEL_PATH = "improved_weed_unet_final.keras"

def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return

    try:
        with st.spinner("Downloading the trained weed detection model..."):
            response = requests.get(MODEL_URL, stream=True, timeout=(30, 300))
            response.raise_for_status()

            temp_path = MODEL_PATH + ".download"
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

            os.replace(temp_path, MODEL_PATH)

    except Exception as error:
        st.error("Unable to download the trained model.")
        st.exception(error)
        st.stop()

download_model()

# Run the existing application without editing app.py.
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
with open(app_path, "r", encoding="utf-8") as file:
    app_code = compile(file.read(), app_path, "exec")

exec(app_code, {"__name__": "__main__", "__file__": app_path})
