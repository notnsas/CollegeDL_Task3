from facenet_pytorch import MTCNN
import os
import imageio
from PIL import Image, ImageOps
import cv2 
import numpy as np
import torch

class DataPreprocessing():
    def __init__(self):
        self.detector = MTCNN(
        # image_size=160,      # final cropped face size
        # min_face_size=40,    # ignore tiny detections (<40px)
        # thresholds=[0.5, 0.6, 0.7],  # P-Net, R-Net, O-Net thresholds; lower => more sensitive
        # factor=0.7,          # scale factor for the image pyramid (default 0.709)
        # keep_all=False,      # set True if you want all faces in an image
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    def detect_faces(self, image):
        boxes, probs = self.detector.detect(image)
        if boxes is None:
            return None
    
        # Convert first box to int
        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    
        # Clip coordinates to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)  # width
        y2 = min(image.shape[0], y2)  # height
    
        return [x1, y1, x2, y2]

    def __save_image(self, image, image_name, output_dir):
        # Make the folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        full_path = os.path.join(output_dir, image_name)
        
        # Save image (img1 must be a numpy array)
        imageio.imwrite(full_path, image)

    def __transform(self, image):
        img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        img_y_cr_cb[:, :, 0] = cv2.equalizeHist(img_y_cr_cb[:, :, 0])
        image_eq = cv2.cvtColor(img_y_cr_cb, cv2.COLOR_YCrCb2RGB)
        return image_eq

    def transform_folder(self, data_dir, output_dir):
        p = 10
        image_names = os.listdir(data_dir)
        for image_name in image_names:
            try:
                image_name_path = os.path.join(data_dir, image_name)
                print(image_name_path)
                # Load image with PIL
                image = Image.open(image_name_path)

                # Apply EXIF auto-orientation
                image = ImageOps.exif_transpose(image)

                # If you need a NumPy array (for your pipeline)
                image = np.array(image)
            except: 
                print("File tidak bisa di load")
                continue
            image_eq = self.__transform(image)
                        # image_grey = imageio.imread(image_name_path, pilmode = "L")
            faces = self.detect_faces(image=image_eq)
            if faces is None:
                print(f"No faces detected in {image_name_path}")
                continue  # Skip this image
            x1, y1, x2, y2 = faces
            print(faces)
            # for (xf, yf, wf, hf) in faces:
            #     cropped_image = image[yf-p+1 : yf+hf+p, xf-p+1 : xf+wf+p]
            cropped_image = image[y1 : y2, x1 : x2, :]
            # imshow(cropped_image, cmap = 'Greys_r')
            self.__save_image(cropped_image, image_name, output_dir)
            print(f"{image_name} sudah di save")