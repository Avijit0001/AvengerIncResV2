import streamlit as st 
import tensorflow as tf
from PIL import Image
import numpy as np
import pyttsx3

engine = pyttsx3.init()

model = tf.keras.models.load_model("C:\Code_EveryThing\Git_Project\AvengerMFaceNet\AvengerClass.h5")


def preprocess(image):
    image = image.resize((112, 112))  
    image = np.array(image)
    image = image.astype('float32') / 255.0 
    image = np.expand_dims(image, axis=0)   
    return image

st.title("Avenger Classification")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    processed_image = preprocess(image)
    prediction = model.predict(processed_image)
    class_names = ['BlackWidow', 'CaptainAmerica', 'Hulk', 'Ironman', 'Thor']  
    predicted_class = class_names[np.argmax(prediction)]
    x = st.button("Predict The Avenger")
    if x:
        st.write('Prediction:', predicted_class)
    y = st.button("Say My Name")
    if y:
        engine.say(predicted_class)
        engine.runAndWait()