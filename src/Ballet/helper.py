from glob import glob
from PIL import Image
import numpy as np

paths = glob("Dataset/Train/Pirouette/*")

# print(paths)

all_ratios= []

# prints paths of all images 
for path in paths:
    image = Image.open(path)
    ratio = np.round(image.width/image.height, 2)
    print(image.width)
    print(image.height)
    all_ratios.append(ratio)


unique, counts = np.unique(all_ratios, return_counts=True)

# print(unique)
# print(unique, counts)

""" Arabesquae
    1.5 - 15
    1.0 - 11

    Battement
    1.78 - 11

    Grand_Plié
    1.0 - 8
    0.75 - 6

    Pirouette
    0.67 - 8
    1.0 - 8
""" 