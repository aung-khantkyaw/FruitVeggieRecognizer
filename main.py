import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import json

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Set page title and icon
st.set_page_config(page_title="Fruit & Veggie Recognizer", page_icon="🍎")

# Sidebar for additional options
with st.sidebar:
    st.title("Fruit & Veggie Recognizer")
    st.write("""
    Upload an image of a fruit or vegetable, and this app will recognize it for you!
    Powered by Machine Learning.
    """)
    st.markdown("---")
    st.header("About")
    st.write("""
    This app uses a pre-trained deep learning model to classify **10 fruits** and **26 vegetables**.
    Simply upload an image, and the app will predict the type of fruit or vegetable.
    """)
    st.markdown("---")
    st.write("© 2025 Fruit & Veggie Recognizer. All rights reserved. Developed by **Group-7**.")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a fruit or vegetable...", type=["jpg", "jpeg", "png"])

# Display image and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        # Predict the class using the model
        result_index = model_prediction(uploaded_file)

        # Load labels
        with open("labels.txt") as f:
            content = f.readlines()
            label = [i.strip() for i in content]  # Remove newline characters

        # Display the predicted label
        st.success("Model is Predicting it's a **{}**".format(label[result_index]))

        # Make API request to OpenRouter for additional information
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": "Bearer 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",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "deepseek/deepseek-r1:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""You are a botanical expert. Provide detailed biological information about {label[result_index]}. Include the following:
                                        Scientific Name (Binomial Nomenclature):
                                        Plant Family:
                                        Botanical Classification (Fruit or Vegetable):
                                        Edible Part(s) of the Plant:
                                        Typical Plant Size and Growth Habit:
                                        Optimal Growing Conditions (Climate, Soil, Seasonality):
                                        Key Nutritional Components and Benefits:
                                        Brief Description of Seed/Reproductive Biology (if applicable):
                                        Any Interesting or Unique Biological Facts:
                                        Ensure the information is accurate and scientifically sound."""
                        }
                    ],
                })
            )

            # Debugging: Print the API response status code and content
            st.write("API Response Status Code:", response.status_code)

            # Check if the API request was successful
            if response.status_code == 200:
                response_json = response.json()  # Parse JSON response
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0]["message"]["content"]  # Extract content
                    st.write(content)  # Display the content
                else:
                    st.error("No valid response from the API. Check the API response format.")
            else:
                st.error(f"Failed to fetch additional information. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"An error occurred while making the API request: {e}")