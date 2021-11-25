import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import cv2
class_names=['0','1','2','3','4','5','6','7','8','9']

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('default_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

html_temp = """
    <div style="background-color:#C0DC44 ;padding:15px">
    <h2 style="color:yellow;text-align:center;">Hand Written Digit Classification Web Application </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)


file = st.file_uploader("Please upload a hand written digit", type=["jpg","png"])
import cv2
from PIL import Image, ImageOps
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
        size = (28,28)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        
        #img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28,28),interpolation = cv2.INTER_AREA)
        newimg = tf.keras.utils.normalize(resized, axis=1)
        newimg = np.array(newimg).reshape(-1,28,28,1)

        #img_reshape = img[np.newaxis,...]
        #prediction = model.predict(img_reshape)
        
        prediction = model.predict(newimg)
        return prediction  

#def check(image):
        if st.button("White"):
         # import numpy
           image_inv = np.invert(image)
           st.image(image_inv, use_column_width=True)

        elif st.button("Black"):
           image_inv = image

 #       return image_inv
          

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    st.text("If background has White colour, press White button Else press black button") 
    st.text("Then press predict")

   # image_inv = image
    if st.button("White"):
         # import numpy
       image = np.invert(image)
       st.image(image, use_column_width=True)

    elif st.button("Black"):
       image = image   

    if st.button("Predict"):
     #  image_inv = check(image)
     #  st.image(image, use_column_width=True)
       predictions = import_and_predict(image, model)
       score=np.array(predictions[0])
       st.title(
         "This image most likely belongs to {} with a {:.2f} percent confidence."
         .format(class_names[np.argmax(score)], 100 * np.max(score))
             )
