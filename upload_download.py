import streamlit as st 
import pandas as pd 
import tkinter as tk
import numpy as np

import os 

from tkinter import filedialog 
from PIL import Image 

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; \
            color: grey;'>Labelling Helper</h1>", unsafe_allow_html=True)

# session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "folder_names" not in st.session_state: 
    st.session_state['folder_names'] = []

if "counter" not in st.session_state: 
    st.session_state.counter = 0

def select_folder(folder_path='.'):
    root = tk.Tk() 
    root.withdraw() 
    folder_path = filedialog.askdirectory(master=root)
    root.destroy() 

    return folder_path 

def get_data(key): 
    if len(st.session_state["folder_names"]) - 1 < key: 
        return False 
    
    return isinstance(st.session_state["folder_names"][key], str) 

def add_data(key, value):
    st.session_state[key] = value 

_, i0, _ = st.columns([0.1, 0.8, 0.1])
def show_image(selected_folder_path, img): 
    name = os.path.join(selected_folder_path, img)
    i0.image(name, caption=img, use_column_width=True)

def prev_image(): 
    st.session_state.counter -= 1
    if st.session_state.counter < 0: 
        st.session_state.counter = 0 
    
def next_image():     
    st.session_state.counter += 1
    if st.session_state.counter >= len_images: 
        st.session_state.counter = len_images - 1  
        st.session_state.disabled = True

def save_image(selected_folder_path, save_folder, img): 
    os.system('cp ' + selected_folder_path + '/' + img + ' ' + selected_folder_path + '/' + save_folder)
    st.write("Saved!") 

    next_image()

def undo_image(selected_folder_path, save_folder, img):
    name = selected_folder_path + '/' + save_folder + '/' + img
    if os.path.exists(name):
        os.system('rm ' + name)
        st.write("Deleted ", img)
    else: 
        st.write(img, " doesn't exist in ", save_folder)

    name = os.path.join(selected_folder_path, img)
    i0.image(name, caption=img, use_column_width=True)

# selecting folder 
selected_folder_path = st.session_state.get("folder_path", None)
folder_select_button = st.button("Select folder") 
if folder_select_button:
    selected_folder_path = select_folder()
    st.session_state.folder_path = selected_folder_path

st.write('Selected folder path: ', selected_folder_path)


st.markdown("""---""")


if selected_folder_path:
    files = os.listdir(selected_folder_path)
    images = [files[i] for i in range(len(files)) if files[i].endswith('.jpg')]
    len_images = len(images)

# selecting number of save buttons 
num = [2, 3, 4]
num_save_buttons = st.selectbox('How many save directories?', num)
button_names = [str(i) for i in (list(range(1, num_save_buttons+1)))]

st.write("Enter the name of each save folder")

if selected_folder_path:
    img = images[st.session_state.counter]

    with st.sidebar:
        prev_button = st.button("Prev img", on_click=prev_image, disabled=st.session_state.disabled)
        skip_button = st.button("Skip img", on_click=next_image, disabled=st.session_state.disabled)  
        finish_button = st.button("Finish")

        if st.session_state.counter < len_images:
            show_image(selected_folder_path, img)     

    bt = False
    for i, n in enumerate(button_names):
        query_session = get_data(i)

        if selected_folder_path and not query_session: 
            dir_name = st.text_input('Name of folder # ' + n,
                                label_visibility=st.session_state.visibility,
                                disabled=st.session_state.disabled)
            
            os.makedirs(os.path.join(selected_folder_path, dir_name), exist_ok=True)
            
            if dir_name:
                with st.sidebar:
                    bt = st.button(dir_name, on_click=save_image, args=([selected_folder_path, dir_name, img]), disabled=st.session_state.disabled)
                    add_data(i, dir_name)
    
    if bt and not st.session_state.disabled: 
        show_image(selected_folder_path, img)  
    
    # todo 
    # delete_button = st.button("Delete img", on_click=undo_image, args=([selected_folder_path, dir_name, img]))

    st.markdown("""---""")

    st.markdown("""
    <style>
    .big-font {
        font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Parameters for creating annotations and moving files</p>', unsafe_allow_html=True)

    if finish_button or st.session_state.counter == len_images - 1:
        st.session_state.disabled = True

        img_path = st.session_state.get("img_path", None)
        img_path_select_button = st.button("Path to original images") 
        if img_path_select_button:
            img_path = select_folder()
            st.session_state.img_path = img_path
        st.write('img_path: ', img_path)

        json_path = st.session_state.get("json_path", None)
        json_path_select_button = st.button("Path to json annotation file") 
        if json_path_select_button:
            json_path = select_folder()
            st.session_state.json_path = json_path
        st.write('json_path: ', json_path)

        good_folder = st.text_input('Name of good images folder')
        good_path = os.path.join(selected_folder_path, good_folder) 
        st.write('good_path: ', good_path)

        new_folder = st.text_input('Name of new images folder')
        new_path = os.path.join(selected_folder_path, new_folder)
        os.makedirs(new_path, exist_ok=True) 
        st.write('new_path: ', new_path)

        create_json = st.button("Create json file")
        move_images = st.button("Move files") 

        if create_json: 
            os.system('python3 YOLOX/tools/create_json.py ' + \
                    '--path ' + img_path + ' ' + \
                    '--save_result ' + \
                    '--exp_file YOLOX/exps/default/yolox_s_CPU ' + \
                    '--ckpt YOLOX/YOLOX_outputs/yolox_s_tl_2 ' + \
                    '--tsize 640 ' + \
                    '--save_file_name ' + good_path + ' ' + \
                    '--conf 0.3 ' + \
                    '--json_path ' + json_path)
            
            os.system('python3 YOLOX/tools/find_missing_img ' + \
                    '--json_file_path ' + json_path + ' ' + \
                    '--good_path ' + good_path)

        if move_images: 
            good_images = os.listdir(good_path)
            for i in good_images: 
                name = os.path.join(img_path, i) 
                os.system('cp ' + name + ' ' + new_path)
            len_good_images = len(good_images)
            st.write('Moved ', len_good_images, ' images to ', new_path) 
        


