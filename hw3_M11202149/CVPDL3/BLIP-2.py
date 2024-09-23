import json
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b-coco", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  

with open('./COCODIR/annotations/instances_train2017.json', 'r') as f:
    coco = json.load(f)

categories = {cat['id']: cat['name'] for cat in coco['categories']}
image_dir = "/data/CVPDL3/COCODIR/train2017"

outputs = []
selected_per_category = {cat_id: 0 for cat_id in categories.keys()}

for image_info in coco['images']:
    if len(outputs) >= len(categories) * 20:
        break  # Stop if the desired total number of images is reached

    image_path = f"{image_dir}/{image_info['file_name']}"
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    boxes = []
    labels = []

    for ann in filter(lambda x: x['image_id'] == image_info['id'], coco['annotations']):
        bbox = ann['bbox']
        x_min, y_min, w, h = bbox
        x_max = x_min + w
        y_max = y_min + h
        bbox = [x_min / image_info['width'], y_min / image_info['height'], x_max / image_info['width'], y_max / image_info['height']]

        boxes.append(bbox)
        labels.append(categories[ann['category_id']])

    # Check conditions: no more than 6 bounding boxes and only one label category
    
    if (len(boxes) <= 6 or labels[0]=="jellyfish") and len(set(labels)) == 1:
        category_id = ann['category_id']
        if selected_per_category[category_id] < 20:
            # Check if the limit of 20 images per category is reached
            selected_per_category[category_id] += 1

            output = {
                "image": image_info['file_name'],
                "label": labels[0],  # Only keep one label
                "height": image_info['height'],
                "width": image_info['width'],
                "bboxes": boxes,
                "generated_text": generated_text,
                "generated_prompt": f" {generated_text}, height: {image_info['height']}, width: {image_info['width']}, single species, diversity, masterpieces, fantasy vivid colors, detailed"
            }

            outputs.append(output)

# Output to JSON file
with open('/data/CVPDL3/output6.7b-coco.json', 'w') as json_file:
    json.dump(outputs, json_file)
# with open('/data/CVPDL3/output6.7b-coco.json', 'w') as json_file:
#     json.dump(outputs, json_file)
# with open('/data/CVPDL3/output.json', 'w') as json_file:
#     json.dump(outputs, json_file)


# ./COCODIR/annotations/instances_train2017.json