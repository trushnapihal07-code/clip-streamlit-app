# CLIP Zero-Shot Image Classifier

This web app uses OpenAI's **CLIP** model to classify images into **user-defined categories** without retraining.

## Features
- Upload any image
- Enter custom labels (comma separated)
- Zero-shot classification using CLIP
- Free deployment using Streamlit Community Cloud

## Tech Stack
- Python
- PyTorch
- CLIP (from OpenAI)
- Streamlit

## How to Use
1. Open the app on Streamlit Community Cloud.
2. Upload an image (jpg, png, jpeg).
3. Enter the labels you want to classify the image into.
4. View the predictions with confidence scores.

## Limitations
- Depends on quality of input labels
- CPU-only free tier may be slow for large images
- Works best for general objects, not very fine-grained categories
