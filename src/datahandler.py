import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


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


# define a function which takes in path of root_directory, batchsize anad returns the dataloaders
# for both train and test.
def pre_processor(root_dir, batchsize):

    train_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        #convert to grayscale
                                        transforms.Resize(255),
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        # need to double check the transforms here
                                        transforms.Normalize([0.5, 0.5, 0.5], # this is for RGB images
                                                            [0.5, 0.5, 0.5])])


    test_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize(255),
                                        #transforms.CenterCrop(224), # do we do centercrop in test??
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])


    # apply the transformation to both train and test data
    train_data = datasets.ImageFolder(root_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(root_dir + '/validation', transform=test_transforms)

    # create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batchsize,
                                            shuffle=False)

    return train_loader, test_loader