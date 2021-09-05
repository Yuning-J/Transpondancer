import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


'''
If you need to load an image dataset, it's more convenient to use the ImageFolder class from the 
torchvision.datasets module.
To do so, you need to structure your data as follows:
root_dir
    |_train
        |_class_1
            |_xxx.png
        .....
        .....    
        |_class_n
            |_xxx.png
    |_validation
        |_class_1
            |_xxx.png
        .....
        .....
        |_class_n
            |_xxx.png
that means that each class has its own directory.
By giving this structure, the name of the class will be taken by the name of the folder!
    '''

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

def collate_function(batch):
    # print(batch)
    samples = [sample[0] for sample in batch]
    # print(samples)
    labels = [sample[1] for sample in batch]
    # print(labels)
    images = []
    for image in samples:
        print(type(image))
        # print(image.width)
        # print(image.height)
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
        # print(image.shape)
        # print(image.shape[2]/image.shape[1])
        # print(type(image))
        images.append(image)
        # print(images)
        
    return images, torch.tensor(labels)

# define a function which takes in path of root_directory, batchsize anad returns the dataloaders
# for both train and test.
def pre_processor(root_dir, batchsize):
    # apply the transformation to both train and test data
    train_data = datasets.ImageFolder(root_dir + '/Train')
    test_data = datasets.ImageFolder(root_dir + '/Validation')

    # create the dataloaders
    train_loader = DataLoader(train_data, batch_size=batchsize, collate_fn=collate_function,  shuffle=True)

    # train_loader = DataLoader(train_data, batch_size=batchsize,
    #                                         shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batchsize, collate_fn=collate_function,
                                            shuffle=False)

    return train_loader, test_loader