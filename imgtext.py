from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import streamlit as st

st.title("Image Caption Generator")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32
    ).to("cpu")  
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        result = processor.decode(out[0], skip_special_tokens=True)
        st.success(result)
