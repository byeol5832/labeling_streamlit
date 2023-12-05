import json
import os
import argparse

def make_parser():
    parser = argparse.ArgumentParser("find missing img!")
    parser.add_argument("--json_file_path", default="")
    parser.add_argument("--good_path", type=str, default="")

    return parser

# json_file_path = [
#     'tl_dataset/annotations/231030_E100#3/231030_E100#3_train_aug.json',
#     'tl_dataset/annotations/231030_E100#3/231030_E100#3_val_aug.json']
# new_names = ['231030_E100#3_train_aug_filter.json', 
#              '231030_E100#3_val_aug_filter.json']
# filter_path = 'results/231030_anyang_E100#3'
# new_anno_path = 'tl_dataset/annotations/231030_E100#3/'

args = make_parser().parse_args()

json_path = [os.path.join(args.json_file_path, 'train.json'), 
             os.path.join(args.json_file_path, 'val.json')]

new_names = [os.path.join(args.json_file_path, 'train_filtered.json'), 
             os.path.join(args.json_file_path, 'val_filtered.json')]

for json_file, new_name in zip(json_path, new_names):
    with open(json_file, 'r') as file:
        data = json.load(file)

    images_to_remove = []
    for image in data['images']:
        file_name = image['file_name'].split('/')[-1]
        file_path = os.path.join(args.good_path, file_name)
        if not os.path.exists(file_path):
            images_to_remove.append(image)

    for image in images_to_remove:
        image_id = image['id']
        data['images'].remove(image)
        annotations_to_remove = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]
        for annotation in annotations_to_remove:
            data['annotations'].remove(annotation)

    output_json_file = new_name 
    with open(output_json_file, 'w') as json_out:
        json.dump(data, json_out, indent=4)
