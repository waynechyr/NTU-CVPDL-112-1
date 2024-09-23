import shutil
import os
import json

with open('/data/CVPDL3/output2.7b.json', 'r') as json_file:
    data = json.load(json_file)

# 這裡改成新的目錄路徑
path = "/data/CVPDL3/COCODIR/train2017"
new_path = "/data/CVPDL3/COCODIR/origintrain2017new"

for entry in data:
    image_path = os.path.join(path, entry["image"])
    
    # 檢查檔案是否存在
    if os.path.exists(image_path):
        new_filename = os.path.join(new_path, entry["image"])
        
        # 複製檔案
        shutil.copy(image_path, new_filename)
# 檢查新目錄中的影像數量
new_files = os.listdir(new_path)
image_files = [file for file in new_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
num_images = len(image_files)

print(f"新目錄中有 {num_images} 張影像。")
