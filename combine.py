import streamlit as st 
import os 

from upload_download import select_folder

st.markdown("<h1 style='text-align: center; \
            color: grey;'>Combine json files</h1>", unsafe_allow_html=True)

if "combine_json_paths" not in st.session_state: 
    st.session_state.combine_json_paths = set() 

if "combined" not in st.session_state: 
    st.session_state.combined = False


def get_data(key, name): 
    if len(st.session_state[name]) - 1 < key: 
        return False 
    
    return isinstance(list(st.session_state[name])[key], str) 

def add_data(value, name):
    st.session_state[name].add(value) 

def combine_json(num): 
    st.write('Enter the absolute path of the annotation files.')
    st.write('The helper combines \'train_filtered.json\'s and \'val_filtered.json\'s in all the paths')
    
    num = int(num)
    for i in range(num): 
        selected_json_path = get_data(i, 'combine_json_paths')

        if not selected_json_path:
            json_path = st.text_input("Absolute path of json_path #" + str(i + 1))
            
            if json_path:
                add_data(json_path, 'combine_json_paths')             
                st.write('json path: ', json_path)

        
    if len(st.session_state.combine_json_paths) == num: 
        train_paths = ','.join([os.path.join(directory, 'train_filtered.json') \
                        for directory in st.session_state.combine_json_paths if len(directory)!=0])
        val_paths = ','.join([os.path.join(directory, 'val_filtered.json') \
                        for directory in st.session_state.combine_json_paths if len(directory)!=0])
        
        target_json_path = st.button("Select the target folder") 
        if target_json_path:
            selected_json_path = select_folder()
            st.session_state.target_json_path = selected_json_path
            st.write('Selected json path: ', selected_json_path)

            os.system('python3 YOLOX/tools/combine_json.py ' + \
                    '--json ' + train_paths + ' ' + \
                    '--target_json_path ' + selected_json_path + ' ' + \
                    '--name ' + 'train.json')

            os.system('python3 YOLOX/tools/combine_json.py ' + \
                    '--json ' + val_paths + ' ' + \
                    '--target_json_path ' + selected_json_path + ' ' + \
                    '--name ' + 'val.json')
        
            st.session_state.combined = True 

            st.write("Done combining json files!")



st.write('Enter the absolute path of the annotation files.')
st.write('The helper combines \'train_filtered.json\'s and \'val_filtered.json\'s in all the paths')

num_json_paths = st.text_input('How many json files to combine?')
if len(num_json_paths) != 0:
    combine_json(num_json_paths) 


# todo
# clear_button = st.button('Clear all')
# if clear_button: 
#     st.session_state.combine_json_paths = set() 
