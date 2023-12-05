# draw inference results on input images and make json file based on the results

import argparse
import os
import time
import json 
from loguru import logger

import cv2

import torch
import numpy as np
import natsort
import random

import sys 
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('/home/armstrong/Desktop/hboh/streamlit/YOLOX/')

from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from classes import classes

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

height = 1080
width = 1920 

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
    ]
).astype(np.float32).reshape(-1, 3)

train_anno = {}
train_anno['images'] = []
train_anno['categories'] = []
train_anno['annotations'] = [] 

val_anno = {}
val_anno['images'] = []
val_anno['categories'] = []
val_anno['annotations'] = [] 

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("--path", default="")
    parser.add_argument("-n", "--name", type=str, default='YOLOX_S', help="model name")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--save_file_name", 
        default="", 
    )
    parser.add_argument(
        "--json_path", 
        default = ""
    )
    parser.add_argument(
        "--streamlit", 
        default=False, 
        action="store_true"
    )

    return parser

def categories(anno): 
    for i in range(len(classes)): 
        anno['categories'].append({
            "supercategory": "Defect", 
            "id": i,
            "name": classes[i]
        })

    return anno

def get_image_list(path):
    image_names = []
    all_ = os.listdir(path)

    for filename in natsort.natsorted(all_):
        apath = os.path.join(path, filename)
        ext = os.path.splitext(apath)[1]
        if ext in IMAGE_EXT:
            image_names.append(apath)
    return image_names

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=classes,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def create_json(anno, output, img_info, img_name, image_id, id_, conf=0.3):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    cnt = 0 

    if output is None:
        return img, cnt, image_id, id_
    
    output = output.cpu()
    bboxes = output[:, 0:4]
    bboxes /= ratio

    cls_ids = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    for i in range(len(bboxes)):
        box = bboxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]

        if score < conf:
            continue

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        box_width = x1 - x0
        box_height = y1 - y0
        bbox = [x0, y0, box_width, box_height]

        anno['annotations'].append({
                        'id': id_, 
                        'image_id': image_id, 
                        'bbox': bbox, 
                        'area': round(box_width * box_height), 
                        'iscrowd': 0, 
                        'category_id': cls_id
                    })

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(classes[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - int(2 * txt_size[1])),
            (x0 + txt_size[0] + 1, y0),
            txt_bk_color,
            -1
        )

        cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4, txt_color, thickness=1)
                
        id_ += 1
        cnt += 1

    if cnt > 0: 
        anno['images'].append({
            'file_name': img_name, 
            'height': height, 
            'width': width, 
            'id': image_id
        })
        
    image_id += 1

    return img, cnt, image_id, id_

def image_demo(predictor, path, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]

    # generate random numbers 
    ran = list(random.sample(range(0, len(files)), len(files)))

    # split train/val/test 
    ninety_five = int(len(ran)*0.95)
    five = int(len(ran)*0.05) 

    train = [files[i] for i in ran[0:ninety_five+1]]
    val = [files[i] for i in ran[ninety_five+1:ninety_five+five+1]]

    train_val_files = [train, val]
    train_val_anno = [train_anno, val_anno]

    image_id = 0 
    id_ = 0 

    for i, anno in enumerate(train_val_anno):
        categories(anno) 
        for image_name in train_val_files[i]:   
            if i > 0 and i % 100 == 0: 
                print('done testing', i, 'files')
            outputs, img_info = predictor.inference(image_name)
            result_image, cnt, image_id, id_ = create_json(anno, outputs[0], img_info, image_name, image_id, id_)

            if save_result and cnt > 0: 
               # save_folder = os.path.join(
                #     vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                # )
                name = ''.join(image_name.replace('txt', 'jpg'))
                name = name.split('/')[-1]
                save_folder = args.save_file_name
                os.makedirs(save_folder, exist_ok=True)
                # save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                # logger.info("Saving detection result in {}".format(save_folder))
                cv2.imwrite(save_folder + '/' + name, result_image)

def main(exp, args):
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name

    # file_name = os.path.join(exp.output_dir, args.experiment_name)
    # os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    # if args.save_result:
    #     vis_folder = os.path.join(file_name, "vis_res")
    #     os.system('mkdir -p ' + vis_folder) 
        # os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
         exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    # model = exp.get_model(args.model_yaml)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    trt_file = None
    decoder = None

    predictor = Predictor(
        model, exp, classes, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    image_demo(predictor, args.path, args.save_result)

    anno_path = args.json_path + '/train.json'
    with open(anno_path, 'w', encoding='utf-8') as outfile: 
        json.dump(train_anno, outfile, indent='\t') 
    
    anno_path = args.json_path + '/val.json'
    with open(anno_path, 'w', encoding='utf-8') as outfile: 
        json.dump(val_anno, outfile, indent='\t') 

    if args.streamlit: 
        os.system('rm -r ' + '/' + '/'.join(ckpt_file.split('/')[:-1]))

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    # exp = get_exp_edgeyolo(args.exp_file, args.name)

    main(exp, args)

    