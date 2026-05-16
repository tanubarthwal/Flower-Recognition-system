# utils.py

import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import textwrap

def load_model(path, num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_image(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        return (
            class_names[predicted_idx],
            confidence.item(),
            probabilities.tolist()
        )

def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def create_pdf(predicted_class, confidence, description, image):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 50, "🌸 Flower Classification Report")

    # Prediction
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Prediction: {predicted_class}")
    c.drawString(50, height - 120, f"Confidence: {confidence:.2f}%")

    # Description
    c.setFont("Helvetica", 12)
    wrapped_description = textwrap.wrap(description, width=90)  
    y_position = height - 160
    c.drawString(50, y_position, "Description:")
    y_position -= 20
    for line in wrapped_description:
        c.drawString(60, y_position, line)
        y_position -= 15

    image_y_position = y_position - 220  # add some space
    if image_y_position < 50:  # Ensure image doesn't go below page margin
        c.showPage()
        image_y_position = height - 100
    # Insert image
    if image:
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        img_reader = ImageReader(img_buffer)

        img_width = 200
        img_height = 200
        c.drawImage(img_reader, 50, image_y_position, width=img_width, height=img_height)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer