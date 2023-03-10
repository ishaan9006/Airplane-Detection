import cv2
from PIL import Image
import requests
from patchify import patchify
import numpy as np
import streamlit as st 
from keras.models import load_model
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt

cnn = load_model('model.h5')

def process_data(img):
    arr = np.array(img)
    print("Curr shape",arr.shape)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    arr = cv2.resize(arr,(20,20))
    print("New shape",arr.shape)

    # Code to improve accuracy
    
    # newImg = cv2.resize(arr,(400,400))
    # patches = patchify(newImg, (80,80,3), step=80)

    # print("Patch Shape",patches.shape)
    # for i in range(patches.shape[0]):
    #     for j in range(patches.shape[1]):
    #         single_patch_img = patches[i,j,:,:,:]
    #         single_patch_img = np.array(single_patch_img[0])
    #         print(single_patch_img.shape)
    #         plt.imsave('./images/' + 'image_' + '_'+ str(i)+str(j)+'.png', single_patch_img)

    arr = arr.reshape(-1, 20,20,3)
    arr = arr/255
    return arr

def predict_value(img):
    data = process_data(img)
    result = cnn.predict(data)
    result = np.argmax(result)
    print("Result", result)
    return result

def load_image(image_file):
	img = Image.open(image_file)
	return img


def load_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_file = load_url("https://assets4.lottiefiles.com/packages/lf20_8Trbef.json")

st.set_page_config(page_title="My page", page_icon=":tada:", layout="wide")

with st.container():
    st.subheader("Hi i'm Ishaan Khullar")

    st.title("Aircraft Detection System")
    st.write(
        """
        A ML model capable of detecting aircrafts in an airbase.
        It takes satellite images as an input and predicts whether
        an aircraft is present or not.
        Currently i'm using cnn just to predict its presence but
        i'm planning to use YOLO Algorithm soon to identify as well
        as mark the target.
        Comes with an accuracy of 98.43%.
        
        """
        )


with st.container():
    st.write("---")
    left_col, right_col = st.columns(2)
    with left_col:
        st.header("Need of this product?")
        st.write("##")
        st.write(
            """
            Well this product of made with a long term vision
            in mind. They say AI is the future of warfare. If
            that's the case then why dont we just sit back and 
            let AI identify enemy aircrafts in an airbase and
            do things necessary to neutralize the threat.
            """
        )
    with right_col:
        st_lottie(lottie_file, height=300, key="coding")

with st.container():
    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        img = load_image(image_file)
        st.image(img,width=250)

        res = predict_value(img)
        myMap = ['No Aircraft Detected','Aircraft Detected']
        st.header(myMap[res])

