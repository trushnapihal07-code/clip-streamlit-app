import streamlit as st
import clip
import torch
from PIL import Image

st.set_page_config(page_title="CLIP Image Classifier")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

st.title("🖼️ CLIP Zero-Shot Image Classifier")
st.write("Upload any image and classify it using custom labels.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

labels_text = st.text_input(
    "Enter labels (comma separated)",
    "cat, dog, car, person, food"
)

if uploaded_file and labels_text:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    labels = [l.strip() for l in labels_text.split(",")]

    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    st.subheader("Predictions")
    results = {
        labels[i]: float(similarity[0][i])
        for i in range(len(labels))
    }

    for label, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        st.write(f"**{label}**: {score*100:.2f}%")
