# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:47:06 2022

@author: MarioPC
"""
import tensorflow as tf
import numpy as np
import streamlit as st
import os
from PIL import Image
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import time
import plotly.io as pio
import plotly.express as px
from sklearn.linear_model import LinearRegression
import pandas as pd
# @st.cache(suppress_st_warning=True)
def load_models():
    tensorflow_model = tf.saved_model.load("./Modelos/model1.pb", tags=None) #baseline model
    # model1 = tensorflow_model.signatures["serving_default"]
    
    tensorflow_model2 = tf.saved_model.load("./Modelos/model2.pb", tags=None) # model with data augmentation
    # model2 = tensorflow_model2.signatures["serving_default"]
    
    tensorflow_model3 = tf.saved_model.load("./Modelos/model3.pb", tags=None) # model trained with loss weights
    # model3 = tensorflow_model3.signatures["serving_default"]
    
    return tensorflow_model,tensorflow_model2,tensorflow_model3

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def load_images(base_path,img_paths):
    img_list = []
    for img in img_paths:
        img_list.append(np.array(Image.open(base_path+img)))
    return img_list


def prob_to_solid(img):
    pred_labels = np.argmax(img,0)
    solid_img = np.zeros([3,512,512])
    solid_img[0,pred_labels == 0] = 1
    solid_img[1,pred_labels == 1] = 1
    solid_img[2,pred_labels == 2] = 1
    return np.moveaxis(solid_img, 0, -1)

st.set_page_config(layout="wide")
st.title('Estimación de la Reducción del Area Forestal a través de DeepLearning')
st.header('Proyecto del Bootcamp de visión artificial')
st.subheader('Grupo: Hufflepuffs')

option = st.selectbox(
     'Seleccione el area de estudio',
     ('Chiquitania Boliviana', 'Alrededores de Rurrenabaque'))

st.header('Area Selectionada: '+ option)

if option == 'Chiquitania Boliviana':
    path = "./gui_data3/P1/"
    plot_names =['2011', '2013', '2016', '2019',
     '2011', '2013', '2016', '2019',
     '2020', '2022','','',
     '2020', '2022','','']
     
elif option == 'Alrededores de Rurrenabaque':
    path = "./gui_data3/P2/"
    plot_names =['2005', '2011', '2013', '2016',
                 '2005', '2011', '2013', '2016',
                 '2019', '2020','2022','',
                 '2019', '2020','2022','']
     
else: 
    st.write("Seleccione el area de estudio")
    

img_paths = os.listdir(path)
img_list = load_images(path,img_paths)

st.header("Identifique el area de estudio")

col_sliders, col_patches = st.columns([1, 3])

values_width = col_sliders.slider(
     'Selecciona los cuadrantes de estudio (Eje x)',
     0, 24, (0, 13))

values_height = col_sliders.slider(
     'Selecciona los cuadrantes de estudio (Eje y)',
     0, 13, (0, 13))




height_size = 200
width_size = 200



# great_img = great_img[:,:,:-1] 
great_img = img_list[-1].copy()
to_seg = great_img[values_height[0]*height_size:values_height [1]*height_size,values_width[0]*width_size:values_width[1]*width_size,:].copy()
great_img[values_height[0]*height_size:values_height [1]*height_size,values_width[0]*width_size:values_width[1]*width_size,0] +=75


col_patches.image(great_img , caption='Zona de Prueba: '+option)


st.header('Area de estudio')
st.image(to_seg, caption='Area de estudio')

st.markdown(":information_source:")
info = st.button('Realizar analisis')
if  info:
    st.subheader("Imagen Segmentada")
    st.write("Carga de los modelos")
    my_bar = st.progress(0)
    
    model1,model2, model3 = load_models()
    my_bar.progress(40)
    model1_ = model1.signatures["serving_default"]
    my_bar.progress(60)
    model2_ = model2.signatures["serving_default"]
    my_bar.progress(80)
    model3_ = model3.signatures["serving_default"]
    my_bar.progress(100)
  
    
    st.write("Proceso de Inferencia")
    
    # plot_names = [name[:-4] for name in img_paths]*2
    # plot_names.sort()
    fig = make_subplots(rows=4, cols=4,subplot_titles=plot_names,
                        horizontal_spacing=0.04,vertical_spacing=0.04)
    
    
    state=0
    my_bar2 = st.progress(state)
    add = 100//(3*len(img_list))
    green_coverage = []
    red_coverage = []
    blue_coverage = []
    for i,img in enumerate(img_list): 
        to_seg = img[values_height[0]*height_size:values_height [1]*height_size,values_width[0]*width_size:values_width[1]*width_size,:].copy()
        model_in =tf.keras.preprocessing.image.img_to_array(Image.fromarray(to_seg).resize((512,512)))    
        arg = tf.convert_to_tensor(np.expand_dims(np.moveaxis(model_in/255, -1, 0), axis=0), dtype=tf.float32)
        mask1 = model1_(arg)['output_0'].numpy()[0]
        state+=add
        my_bar2.progress(state)
        mask2 = model2_(arg)['output_0'].numpy()[0]
        state+=add
        my_bar2.progress(state)
        mask3 = model3_(arg)['output_0'].numpy()[0]
        state+=add
        my_bar2.progress(state)
        segmented = prob_to_solid(mask1*0.6+mask2*0.2+mask3*0.2)    
        green_coverage.append(np.sum(segmented[:,:,1])/(512*512))
        red_coverage.append(np.sum(segmented[:,:,0])/(512*512))
        blue_coverage.append(np.sum(segmented[:,:,2])/(512*512))
        if i < 4:      
            fig.add_trace(go.Image(z=Image.fromarray(to_seg).resize((512,512))), 1, i+1)
            fig.add_trace(go.Image(z=Image.fromarray(np.uint8(segmented*255),'RGB')), 2, i+1)
        else:
            fig.add_trace(go.Image(z=Image.fromarray(to_seg).resize((512,512))), 3, i+1-4)
            fig.add_trace(go.Image(z=Image.fromarray(np.uint8(segmented*255),'RGB')), 4, i+1-4)
    my_bar2.progress(100)
    
    fig.update_layout(height=1600,width=1600)
    fig.update_xaxes(visible=False)
    #y axis    
    fig.update_yaxes(visible=False)
    
    st.plotly_chart(fig)
    col1, col2 = st.columns([3, 2])
    
    
    X = np.array([name[:-4] for name in img_paths]).reshape(-1,1)
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.array(green_coverage).reshape(-1,1)
    reg = LinearRegression().fit(X, y)
    
    x_range = np.linspace(2000,2027, 100)
    y_range = reg.predict(x_range.reshape(-1, 1))
    
    
    
    fig_r = go.Figure([go.Scatter(x=x_range, y=100*y_range.reshape(-1), name='Proyección'),
        go.Scatter(x=X.reshape(-1),y=100*y.reshape(-1), opacity=0.8, name='Area Forestal Segmentada',
                   line = dict(width=4, dash='dash'))
    ])
    
    fig_r.update_layout(width=800, xaxis_title="Año",
                        yaxis_title="[%] Area Forestal",
                        legend=dict(yanchor="top",
                                    y=0.99,
                                    xanchor="left",
                                    x=0.01))

    
    col1.subheader("Comportamiento Temporal")
    col1.plotly_chart(fig_r)
    
    df = pd.DataFrame({
      'Area Forestal [%]': 100*np.array(green_coverage) ,
      'Area Deforestada [%]': 100*np.array(red_coverage) ,
      'Area Otros [%]"': 100*np.array(blue_coverage) ,
    }, index = [name[:-4] for name in img_paths])
     
    col2.subheader("Tabla Resumen")
    col2.write('Zona de Prueba: '+option)
    col2.subheader("")
    col2.write(df)
    
    if st.button('Seleccionar otra area de estudio'):
        info = False
else:
    pass



# st.image(Image.fromarray(np.uint8(segmented*255),'RGB'))
# tensorflow_model = tf.saved_model.load("./output/model1.pb", tags=None)
# infer = tensorflow_model.signatures["serving_default"]
# pic = np.zeros([1,3,512,512])

# arg = tf.convert_to_tensor(pic, dtype=tf.float32)
# infer(arg)
