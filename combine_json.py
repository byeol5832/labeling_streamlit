# 하나의 json 파일로 합치는 코드 

import json 
import argparse
import os

height = 1080 
width = 1920 

def make_parser():
    parser = argparse.ArgumentParser("combine json files!")
    parser.add_argument('--json', required=True)
    parser.add_argument('--target_json_path', default='')
    parser.add_argument('--name', default='new.json')

    return parser 

args = make_parser().parse_args()

json_files = args.json.split(',')
target = os.path.join(args.target_json_path, args.name)

data = {} 
data['images'] = [] 
data['categories'] = [] 
data['annotations'] = [] 
for i, jf in enumerate(json_files): 
    if i == 0 :
        with open(jf, 'r') as f: 
            t_data = json.load(f)  

            if len(t_data['images']) > 0: 
                data['images'] = t_data['images'] 
                data['categories'] = t_data['categories'] 
                data['annotations'] = t_data['annotations'] 

                last_id = int(data['images'][-1]['id'])
                last_anno_id = int(data['annotations'][-1]['id'])
    else: 
        with open(jf, 'r') as f: 
            v_data = json.load(f)

            if len(v_data['images']) > 0: 
                idx = 0 
                anno_idx = last_anno_id + 1
                for id in range(last_id+1, last_id+len(v_data['images'])+1): 
                    tmp = int(v_data['images'][idx]['id'])
                    data['images'].append({
                        'file_name': v_data['images'][idx]['file_name'], 
                        'height': height, 
                        'width': width, 
                        'id': id
                    })

                    idx += 1 

                    for anno_id in range(len(v_data['annotations'])):
                        if int(v_data['annotations'][anno_id]['image_id']) == tmp:    
                            data['annotations'].append({
                                    'id': anno_idx, 
                                'image_id': id,   
                                    'bbox': v_data['annotations'][anno_id]['bbox'], 
                                    'area': v_data['annotations'][anno_id]['area'], 
                                    'iscrowd': 0, 
                                    'category_id': v_data['annotations'][anno_id]['category_id']
                                })
                        
                            anno_idx += 1
                
                last_id = int(data['images'][-1]['id'])
                last_anno_id = int(data['annotations'][-1]['id'])
    

with open(target, 'w', encoding='utf-8') as outfile: 
    json.dump(data, outfile, indent='\t')  
