import os
import re
# Buat ngebantu datasets class
def class_to_image_path(class_image, data_dir):
    class_to_images = {cls: [] for cls in class_image}
    for cls in class_image:
        for image in os.listdir(data_dir):
            if (extract_class(image) == cls):
                image_path = os.path.join(data_dir, image)
                class_to_images[cls].append(image_path)
    return class_to_images

def extract_class(txt):
    raw_class = re.split(r"[-_.]", txt, 2)[0:2]    
    complete_class = raw_class[0] + "_" + raw_class[1]
    complete_class = re.sub(r'[\d\s]+$', '', complete_class)
    return complete_class