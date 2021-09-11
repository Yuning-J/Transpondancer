#Library imports
import numpy as np
import streamlit as st
from keras.models import load_model
import cv2
model = load_model('dance.h5')
st.title("Indian Classical Dance Image Classification")
CLASS_NAMES = ['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi', 'manipuri', 'mohiniyattam', 'odissi', 'sattriya']
st.write("Classes",CLASS_NAMES)
st.markdown("Upload an image of the indian dance")
dance_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

st.write("Submit",submit)

#Setting Title of App

if submit:
    if dance_image is not None:

        file_bytes = np.asarray(bytearray(dance_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (180,180))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,180,180,3)
        #Make Prediction
        pred = model.predict(opencv_image)

        st.title(str("Dance image is of "+CLASS_NAMES[np.argmax(pred)]))
