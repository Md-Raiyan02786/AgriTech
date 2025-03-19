import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import requests

# Set up OpenRouter API Key
OPENROUTER_API_KEY = "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"  # Replace with actual key

# Load the CNN Model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

# Load Class Names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Disease
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to Get Treatment Recommendation using OpenRouter AI
def get_treatment_recommendation(disease_name):
    # Convert to lowercase for a reliable comparison
    healthy_keywords = ["healthy", "no disease", "normal leaf", "healthy leaf", "no issues"]
    
    if any(keyword in disease_name.lower() for keyword in healthy_keywords):
        return "✅ The leaf is healthy. No treatment is needed. Keep monitoring for any future issues."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourwebsite.com",  # Replace with your actual site
        "X-Title": "Plant Disease Detector"
    }
    
    data = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",  # Choose LLaMA 3 or another model like GPT-4
        "messages": [
            {"role": "system", "content": "You are an expert in plant disease treatment."},
            {"role": "user", "content": f"How do I treat {disease_name} in plants?"}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        print("API Response:", response_json)  # Debugging: Print full response

        if "choices" in response_json:
            return response_json['choices'][0]['message']['content']
        else:
            return f"API Error: {response_json}"  # Show full error response
    except Exception as e:
        return f"Error: {str(e)}"


# Streamlit App UI
st.title('🌿 AgriTech Solution: Plant Disease Detector')

uploaded_image = st.file_uploader("📤 Upload a plant image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button('🔍 Predict Disease'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'🌱 **Prediction:** {str(prediction)}')

            # Fetch treatment recommendation from OpenRouter AI
            treatment = get_treatment_recommendation(prediction)
            st.info(f'💡 **Treatment Recommendation:** {treatment}')
