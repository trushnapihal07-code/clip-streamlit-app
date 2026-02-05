import torch
import clip
from PIL import Image
import streamlit as st

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

CATEGORIES = [
    "a photo of an animal",
    "a photo of a person",
    "a photo of food",
    "a photo of a plant or flower",
    "a photo of a vehicle",
    "a photo of a building",
    "a photo of furniture",
    "a photo of an electronic device",
    "a photo of clothing",
    "a photo of a document or text",
    "a photo of nature or landscape",
    "a photo of an indoor scene",
    "a photo of an outdoor scene"
]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(CATEGORIES).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        logits = image_features @ text_features.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    st.subheader("Classification Result")

    top_idx = probs.argmax()
    st.success(f"🧠 **Predicted Category:** {CATEGORIES[top_idx].replace('a photo of ', '').title()}")

    st.subheader("Confidence Scores")
    for label, prob in sorted(zip(CATEGORIES, probs), key=lambda x: x[1], reverse=True):
        st.write(f"{label.replace('a photo of ', '').title()}: **{prob*100:.2f}%**")
