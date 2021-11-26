import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import json
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

######################
# Data
######################


path = '/home/ubuntu/KJW/Data/DSLR/png_json/'
label_paths = []

for file_path in os.listdir(path):
    if file_path.split('.')[1] == 'json':
        label_paths.append(file_path)

train_files = []
for json_file in label_paths:
    with open(path+json_file, 'rt', encoding='utf-8') as fp:
        data = json.load(fp)
        train_files.append([path+data['imagePath'], path+json_file])
		
def create_category(label_paths):
    category_list = []
    
    for label_path in label_paths:
        with open(path+label_path, 'rt', encoding='utf-8') as fp:
            # load json
            data = json.load(fp)
            for label_names in data['shapes']:                
                if label_names['label'] not in category_list:
                    category_list.append(label_names['label'])
                    
    return category_list
category_list = create_category(label_paths)        

import numpy as np
from detectron2.structures import BoxMode



def get_box_dict(data_list):
    dataset_dicts = []
    for i, path in enumerate(data_list):
        img_path = path[0]
        label_path = path[1]
        with open(label_path, 'rt', encoding='utf-8') as fp:
            # load json
            data = json.load(fp)
            height, width = cv2.imread(img_path).shape[:2]
            record = {}
            record['file_name'] = img_path
            record['image_id'] = i
            record['height']= height
            record['width']= width
#             print(record['file_name'], record['image_id'], record['height'], record['width'])
            #for i in data_list[1] to get bbox and category
            objs = []
            for one_label in data['shapes']:
                one_box = one_label['points']
                xyxy_one_box = [one_box[0][0], one_box[0][1], one_box[1][0], one_box[1][1]]
                
                obj = {
                    "bbox": xyxy_one_box,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": category_list.index(one_label['label']),
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


######################
# Training
######################

from detectron2.data import DatasetCatalog, MetadataCatalog
for d,x in [("train",train_files)]:
    print("Caries_" + d)
    DatasetCatalog.register("Caries_" + d, lambda x=x: get_box_dict(x))
#     MetadataCatalog.get("Caries_" + d).set(thing_classes=['gold cr', 'resin', 'am',  'caries enamel',  'caries dentin',  'metal cr',  'irm',  'zirconia cr',  'gold inlay',  'pfm metal',  'pfm porcelain',  'fx',  'root rest',  'implant',  'tempit',  'blue resin'],
#                                         thing_colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)])
    MetadataCatalog.get("Caries_" + d).set(thing_classes=['gold cr', 'resin', 'am',  'caries enamel',  'caries dentin',  'metal cr',  'irm',  'zirconia cr',  'gold inlay',  'pfm metal',  'pfm porcelain',  'fx',  'root rest',  'implant',  'tempit',  'blue resin'])
# PCB_metadata = MetadataCatalog.get("Caries_train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("Caries_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 30000    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4096 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16 
cfg.OUTPUT_DIR = '/home/ubuntu/KJW/Project/Caries/DSLR/r50/'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


######################
# Inference
######################

def select_color(class_):
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (255, 255, 255),
                 (0, 255, 255), (0, 0, 0), (120, 0, 0), (120, 120, 0),  (120, 0, 120) , (0, 120, 120) ,
                  (120, 120, 120) , (0, 120, 0),  (0, 0, 120),  (120, 50, 120)] 
    return color_list[class_]

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

output_path = '/home/ubuntu/KJW/Project/Caries/DSLR/what/compare/'
testset_path = '/home/ubuntu/KJW/Data/DSLR/Test200/'
for label_path in label_paths:
    with open(label_path, 'r', encoding = 'utf-8') as json_file:
        json_data = json.load(json_file)
        compare_img_path = testset_path+json_data['imagePath']
        img = cv2.imread(compare_img_path)
        hei, wid = img.shape[:2]
        pred_img = img.copy()
        label_img = img.copy()
        
        for shapes in json_data['shapes']:
            label = shapes['label']
            point = shapes['points']
            
            color = select_color(category_list_.index(label))
            cv2.rectangle(label_img, (int(point[0][0]), int(point[0][1])), (int(point[1][0]), int(point[1][1])), color)
            cv2.putText(label_img, label, (int(point[0][0]), int(point[0][1])), 1, 1, color)            

        output = predictor(img)
        boxes = output['instances'].pred_boxes.tensor.to('cpu').numpy()
        classes = output['instances'].pred_classes.to('cpu').numpy()
        for pred_box, pred_class in zip(boxes, classes):
            color = select_color(pred_class)
            cv2.rectangle(pred_img, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), color)
            cv2.putText(pred_img, category_list[pred_class], (int(pred_box[0]), int(pred_box[1])), 1, 1, color)

        compared_img = np.hstack((label_img, pred_img))
        cv2.imwrite(output_path+json_data['imagePath'], compared_img)
