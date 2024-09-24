import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
model_checkpoint_path = "output/checkpoint0011.pth" # change the path of the model checkpoint

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# load coco names
with open('output/name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}

from PIL import Image
import datasets.transforms as T
# 指定輸入圖像資料夾和输出 JSON 文件路徑
input_image_folder = "COCODIR/test"
output_json_file = "output/output.json"
# input_image_folder = "COCODIR/test"
# output_json_file = "output/outputtest.json"

# 讀取所有图像
image_files = [f for f in os.listdir(input_image_folder) if f.endswith((".jpg", ".jpeg", ".png"))]
#results = []
results = {}
for image_file in image_files:
    image_path = os.path.join(input_image_folder, image_file)
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(image, None)
    output = model.cuda()(image[None].cuda())
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]


    # 篩選結果
    # threshold = 0.3
    threshold = 0
    select_mask = output['scores'] > threshold
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])[select_mask]
    # 转换 coco 格式到新格式
    new_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    new_boxes[:, 0] = new_boxes[:, 0] * W
    new_boxes[:, 1] = new_boxes[:, 1] * H
    new_boxes[:, 2] = new_boxes[:, 2] * W
    new_boxes[:, 3] = new_boxes[:, 3] * H

    #boxes = box_ops.box_cxcywh_to_xyxy(output['boxes'])[select_mask]
    labels = output['labels'][select_mask]
    scores = output['scores'][select_mask]

    # 转换 scores Tensor 为 NumPy 数


    # 保存結果
    result = {
         #image_file:{                      #"image_file": image_file,
        "boxes":  new_boxes.tolist(),
        "labels": labels.tolist(),
        "scores": scores.tolist()
        #}
    }
    #results.append(result)
    results[image_file] = result

# 保存结果到 JSON 文件
with open(output_json_file, 'w') as json_file:
    json.dump(results, json_file)
