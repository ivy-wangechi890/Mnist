import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#load_model
model=tf.keras.models.load_model('mnist_cnn_model.h5')

#model labels
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def process_image(image):
    image = image.convert("L")
    image=image.resize((28,28))
    image=np.array(image)/255.0
    image=np.expand_dims(image,axis=-1)
    image = np.expand_dims(image, axis=0)
    return image
def predict(image):
    processed_image=process_image(image)
    predictions=model.predict(processed_image)
    return labels[np.argmax(predictions)]
#streamlit app
st.title("Image Prediction with the Mnist Model")
st.write("Upload an image to make some predictions with the model.")

uploaded_file=st.file_uploader("Choose an image for upload",type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image,caption="uploaded_image",use_column_width=True)

    st.write("Predicting")
    prediction=predict(image)
    st.write(f"Prediction: {prediction}")