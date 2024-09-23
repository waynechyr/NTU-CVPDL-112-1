import os
folder_path = "/data/CVPDL3/GLIGEN/generation_samples/text_prompt1"

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        new_filename = filename.replace(".jpg", "_generated.jpg")
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
