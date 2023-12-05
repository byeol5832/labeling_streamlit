Labelling Helper

- Uses YOLOX model to inference images 
- Manually separate the inference images to good & bad 
- Create json annotation files for good images 
- Move the good original images to a new folder 

User guide 

1. create a conda environment 
2. pip install -r requirements.txt 
3. pip install -r YOLOX/requirements.txt 
4. python3 -m streamlit run upload_download.py 
