import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Step 1: Function to extract images from PDF
def extract_images_from_pdf(pdf_bytes):
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    
    # Iterate through pages to find and extract images
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images_info = page.get_images(full=True)
        
        # Extract each image
        for img_index, img_info in enumerate(images_info):
            xref = img_info[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    
    return images

# Step 2: Function to generate captions for images
def generate_caption(image, processor, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Preprocess the image
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    
    # Decode the caption from the model's output
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption

# Step 3: Streamlit setup
st.title("PDF Image Extraction & Captioning")
st.write("Upload a PDF file, and we will extract the images and generate captions for them!")

# Upload file through Streamlit
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

# Load the captioning model
@st.cache_resource
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_captioning_model()

if uploaded_pdf is not None:
    st.write("PDF uploaded successfully! Extracting images and generating captions...")

    # Extract images from the uploaded PDF
    pdf_bytes = uploaded_pdf.read()
    images = extract_images_from_pdf(pdf_bytes)
    
    if images:
        st.write(f"Found {len(images)} image(s) in the PDF.")
        
        # Display extracted images and generate captions
        for idx, image in enumerate(images):
            st.image(image, caption=f"Image {idx + 1}", use_column_width=True)
            
            # Generate a caption for the image
            caption = generate_caption(image, processor, model)
            st.write(f"Caption for Image {idx + 1}: {caption}")
    else:
        st.write("No images found in the PDF.")
else:
    st.write("Please upload a PDF file.")
