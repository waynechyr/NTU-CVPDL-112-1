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

image = Image.open("./COCODIR/test/IMG_8396_jpg.rf.106a6ced5c649ea81f0de8ecaa4ff3b8.jpgco").convert("RGB") # load image

# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)

# predict images
output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

# visualize outputs
thershold = 0.3 # set a thershold

vslzr = COCOVisualizer()

scores = output['scores']
labels = output['labels']
#boxes = box_ops.box_cxcywh_to_xyxy(output['boxes'])
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
#new_boxes = box_ops.box_cxcywh_to_xyxy(boxes)
select_mask = scores > thershold

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': torch.Tensor([image.shape[1], image.shape[2]]),
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir=None, dpi=100)