from PIL import Image
import os

input_folder = "../COCODIR/origintrain2017new"
output_folder = "./resized_picture"


target_size = (512, 512)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    
    
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    
    img = Image.open(input_path)

    
    resized_img = img.resize(target_size)

    
    output_path = os.path.join(output_folder, filename)
    resized_img.save(output_path)

print("Resize complete!")
