import streamlit as st

import imutils
import numpy as np
import skimage
import time

from PIL import Image
from transform import four_point_transform
from skimage.filters import threshold_local
from scipy.signal import gaussian
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d

import cv2

def find_cnts(cnts):
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    screenCnt = 0
    #loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    return screenCnt

def noisy(noise_typ,image):
   if noise_typ == "Gaussian":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   
   elif noise_typ == "Salt and Pepper":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   
   elif noise_typ == "Poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
  
   elif noise_typ =="Speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

def Filter_Selector():
    SelectedFilter = st.multiselect('Choose a Filter or more',('Median Filter','Gaussian Blur','Weiner Filter','Bilateral Filter','Unsharp Masking'))
    st.markdown("""
    <style> label, .st-d8{
        font-size:2rem!important;
    } 
    </style>
    """,unsafe_allow_html=True)
    return SelectedFilter

def Filter_Function(warped, SelectedFilter):
    st.write(SelectedFilter)
    Output_Filter= warped
    i=0
    while i != len(SelectedFilter):
        if SelectedFilter[i] == 'Median Filter':
            Output_Filter = skimage.filters.median(Output_Filter)
            i+=1
            
        elif SelectedFilter[i] == 'Gaussian Blur':
            Output_Filter = skimage.filters.gaussian(Output_Filter,(1,1),0)
            i+=1
            
        elif SelectedFilter[i] == 'Weiner Filter':
            psf = np.ones((5, 5)) / 25
            Output_Filter = convolve2d(Output_Filter, psf, 'same')
            Output_Filter += 0.1 * Output_Filter.std() * np.random.standard_normal(Output_Filter.shape)
            Output_Filter = skimage.restoration.wiener(Output_Filter,psf,0.1)
            i+=1
        
        elif SelectedFilter[i] == 'Bilateral Filter':
            Output_Filter = skimage.restoration.denoise_bilateral(Output_Filter,multichannel=False)
            i+=1
            
        elif SelectedFilter[i] == 'Unsharp Masking':
            Output_Filter = skimage.filters.unsharp_mask(Output_Filter)
            i+=1
        
        else:
            pass
    return Output_Filter

def main():
    "Image Pre-Processing"
    st.title('PreProcessing Framework for document Image Analysis')
    st.subheader('A python framework developed using streamlit')

    img_file_buffer = st.file_uploader("Upload Your Image", type=['png','jpg','jpeg'])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        ratio = img_array.shape[0] / 500.0
        orig = img_array.copy()
        image = imutils.resize(img_array, height = 500)
        st.subheader("STEP 1: Thresholding")
        with st.spinner('Wait for Image to Load'):
            time.sleep(1)
        st.success('Image Loaded!')
        st.image(image, caption='Original Image',width=300)
    

    submit = st.checkbox('Apply Thresholding Filters')
    if submit:        
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        st.image(th1, caption='Binarization Filter', width=300)

        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
        st.image(th2, caption='Adaptive Gaussian Binarization Filter', width=300)

        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        st.image(th3, caption='Otsu Binarization Filter', width=300)

        if submit:
            st.success('Thresholding Filter Succesfully Applied!')

    choice = st.selectbox('Which Filter Would You like to Choose?', ('','Binarization Filter','Adaptive Gaussian Binarization Filter','Otsu Binarization Filter'))
    st.write('You Selected: ', choice)

    try:
        if choice == 'Binarization Filter':
            cnts = cv2.findContours(th1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        elif choice == 'Adaptive Gaussian Binarization Filter':
            cnts = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        elif choice == 'Otsu Binarization Filter':
            cnts = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
        else:
            pass
    except UnboundLocalError:
        pass
    else:
        if choice:
            if choice=='':
                st.error('Edges Could not be found Select Different Thresholding Filter!')

    try:
        screen_count = find_cnts(cnts)
        contors = cv2.drawContours(image, [screen_count], -1, (0, 255, 0), 4)
        st.subheader("STEP 2 : Find Contours Of Paper")
        st.image(contors, caption='Outlined Image')
        if choice:
            st.success('Edges of the Image Succesfully Found!')
        warped = four_point_transform(orig, screen_count.reshape(4, 2) * ratio)
        # convert the warped image to grayscale, then threshold it
        # to give it that 'black and white' paper effect
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        T = threshold_local(warped, 11, offset = 10, method = "gaussian")
        warped = (warped > T).astype("uint8") * 255
        # apply the four point transform to obtain a top-down
        # # view of the original image

        st.subheader("STEP 3: Apply perspective transform")
        st.image(warped, caption='Applying Perspective Transform to the Image')
        if choice:
            st.success('Perspective Transform Applied!')
        Add_Noise = st.checkbox('Do you want to add Noise')
        st.markdown("""
        <style> {
            font-size:2rem!important;
            } </style>
            """,unsafe_allow_html=True)
        if Add_Noise:
            SelectNoise = st.selectbox('Choose Your type of Noise to add', ('','s&p','gaussian','poisson','speckle'))
            st.write('You Selected: ', SelectNoise, 'noise')
            if SelectNoise:
                noise = skimage.util.random_noise(warped, mode=SelectNoise)
                my_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1)
                st.image(noise)
                st.success('Noise Succesfully Applied!')
                SelectedFilter = Filter_Selector()
                Filtered_Img = Filter_Function(noise, SelectedFilter)
        else:
            SelectedFilter = Filter_Selector()
            Filtered_Img = Filter_Function(warped, SelectedFilter)
        
        if SelectedFilter:
            my_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)

        st.image(Filtered_Img, caption='Selected Filters are applied to the Image')
        if SelectedFilter:
                st.success('Filter Succesfully Applied!')
    except UnboundLocalError:
        pass
    except RuntimeError:
        if err:
            st.error('Please Select Different Filter')


if __name__ == "__main__":
    main()