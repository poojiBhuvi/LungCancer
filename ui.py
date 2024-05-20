import streamlit as st
from keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your own model
model_path = "ct_effnet_best_model.hdf5"  # Update with your model path
model = load_model(model_path)

# Define class labels
class_labels = {
    0: 'adenocarcinoma',
    1: 'large.cell.carcinoma',
    2: 'normal',
    3: 'squamous.cell.carcinoma'
}

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.convert("RGB")  # Convert to RGB format
    img = img.resize((350, 350))  # Adjust the target size to match your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to the range [0, 1]
    return img_array

# Function to make predictions
def predict(image):
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels.get(predicted_class_index, "Unknown")
    return predicted_class

# Streamlit app
def main():
    st.title("Lung Cancer Image Classification")
    st.write("Upload an image to classify the type of lung cancer.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image",width=200)
        image_array = preprocess_image(uploaded_file)
        prediction = predict(image_array)
        st.warning(prediction)

if __name__ == "__main__":
    main()
