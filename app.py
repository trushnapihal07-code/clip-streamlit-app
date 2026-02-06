import torch
import clip
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Universal Image Classifier")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

CATEGORIES = [
    "an animal",
    "a person",
    "food",
    "a plant or flower",
    "a vehicle",
    "a building",
    "furniture",
    "an electronic device",
    "clothing",
    "a document or text",
    "nature or landscape",
    "an indoor scene",
    "an outdoor scene"
]

st.title("🧠 Universal Image Classifier (Deep Learning)")
st.write("Upload **any image** and get the best classification.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(CATEGORIES).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    best_index = similarity.argmax().item()
    best_label = CATEGORIES[best_index]
    confidence = similarity[0][best_index].item() * 100

    st.subheader("Prediction")
    st.success(f"🖼️ This image is most likely **{best_label.upper()}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
