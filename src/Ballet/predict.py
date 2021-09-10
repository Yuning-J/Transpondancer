import io
import torch

import torchvision.transforms as transforms
from PIL import Image

def custom_transform(padding=(0,0)):
    """
    padding[0] is the height
    padding[1] is the width
    """
    custom = transforms.Compose([
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Pad(padding, fill=0),
                        transforms.Resize((90, 160)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5],
                                            [0.5])])
                        
    return custom

def collate_function(image):

    with open(image) as f:
        ratio = f.read()
    # images = []
    ratio = image.width/image.height
    # print(ratio)
    if 16/9 -0.03<= ratio <= 16/9 +0.03:
        transform = custom_transform()
        image = transform(image)
    elif ratio > 16/9:
            x = int((9/16*image.width - image.height)/2)
            transform = custom_transform((0,x))
            image = transform(image)
    elif ratio < 16/9:
            x = int((16/9*image.height-image.width)/2)
            transform = custom_transform((x,0))
            # print(transform)
            image = transform(image)

    # images.append(image)
    # print(images)
        
    return image


# with open("test_pic.jpg") as f:
#     # image_bytes = f.read()
#     tensor = collate_function(f)
#     print(tensor)

image = collate_function("test_pic.jpg")

print(image)