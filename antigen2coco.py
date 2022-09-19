import os
import json
from glob import glob
from tqdm import tqdm

antigen_image_dir = "../antigen/datagen/data/generated"
antigen_json_dir = "../antigen/datagen/data/json"
output_json = "data/annotations/train.json"

images = []
annotations = []
categories = [
    {'id': 0, 'name': '0', 'supercategory': 'angle'},
    {'id': 1, 'name': '90', 'supercategory': 'angle'},
    {'id': 2, 'name': '180', 'supercategory': 'angle'},
    {'id': 3, 'name': '270', 'supercategory': 'angle'},
]

cat_id = {}
for i in categories:
    cat_id[i['name']] = i['id']

image_id = 0

for f in tqdm(glob(antigen_image_dir+'/*.jpg')):
    fn = os.path.split(f)[-1] # 文件名
    fbn, _ = os.path.splitext(fn)

    # 读入 json
    json_file = os.path.join(antigen_json_dir, fbn+'.json')
    with open(json_file, 'r') as fp:
        j = json.load(fp)

    assert j['shapes'][0]['label']=='box'

    w, h = j['imageWidth'], j['imageHeight']
    p1 = j['shapes'][0]['points']

    y = [
        p1[0][0], # box
        p1[0][1],
        p1[1][0],
        p1[1][1],
    ]

    # 计算需选择角度
    rotate_angle = 0
    box1 = y

    if box1[0]<box1[2]: # 起点 在左
        if box1[1]<box1[3]: # 起点 在上
            rotate_angle = 0
            x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
        else:
            rotate_angle = 90
            x1, y1, x2, y2 = box1[0], box1[3], box1[2], box1[1]
    else: # 起点 在右
        if box1[1]<box1[3]: # 起点 在上
            rotate_angle = 270
            x1, y1, x2, y2 = box1[2], box1[1], box1[0], box1[3]
        else:
            rotate_angle = 180
            x1, y1, x2, y2 = box1[2], box1[3], box1[0], box1[1]

    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    # 计算面积
    bbw, bbh = abs(x1-x2), abs(y1-y2)
    area = bbw * bbh


    # 图片列表
    images.append({
        'file_name': fn, 
        'id': image_id, 
        'width': w, 
        'height': h,
    })

    # 标记列表
    annotations.append({
        'area': area, 
        'bbox': [x1, y1, bbw, bbh], 
        'category_id': cat_id[str(rotate_angle)], 
        'id': image_id, 
        'image_id': image_id, 
        'iscrowd': 0, 
        'segmentation': [[x1, y1, x2, y1, x2, y2, x2, y1]]
    })

    image_id += 1

labels = {
    'categories'  : categories, 
    'annotations' : annotations, 
    'images'      : images,
}

with open(output_json, 'w') as f:
    json.dump(labels, f, indent=4)
