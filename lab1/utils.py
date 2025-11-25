import numpy as np
import cv2


class Erosion:
    def __init__(self, kernel=None):
        if kernel is None:
            kernel = np.ones((3, 3), dtype=np.uint8)
        self.kernel = kernel

        self.kernel_h, self.kernel_w = self.kernel.shape
        self.pad_h = self.kernel_h // 2
        self.pad_w = self.kernel_w // 2
    
    def erode(self, image: np.ndarray):
        img_h, img_w = image.shape
        
        padded_img = np.pad(image, ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), mode='constant', constant_values=0)
        
        output = np.zeros_like(image)
        
        for i in range(img_h):
            for j in range(img_w):
                region = padded_img[i:i+self.kernel_h, j:j+self.kernel_w]
                
                match = np.where(self.kernel == 1, region, 1)
                output[i, j] = np.all(match == 1)
        
        return output


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    return binary_image


def image_to_bin_array(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized_img = img / 255
    binarized_img = np.where(binarized_img > 0.5, 1, 0)
    return binarized_img


def save_binarized_flowers_images():
    import os
    
    output_folder = 'flowers_binarized/'
    os.makedirs(output_folder, exist_ok=True)
    
    input_folder = 'flowers/'
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder {input_folder} does not exist")
    
    for dir in os.listdir(input_folder):
        for filename in os.listdir(os.path.join(input_folder, dir)):
            if filename.lower().endswith('.jpg'):
                img_path = os.path.join(input_folder, dir, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    binary_img = otsu_binarize(img)
                                        
                    output_path = os.path.join(output_folder, dir, filename)
                    output_dir = os.path.join(output_folder, dir)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(output_path, binary_img)

