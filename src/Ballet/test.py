'''
This file is only for testing things
    '''
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import datahandler as dh

#pre_preocessor function takes absolute path and batch_size as arguments
batch_size = 4
train_loader, test_loader = dh.pre_processor(
                                '../Dataset/', batchsize=batch_size)

# print(train_loader)
# print(test_loader)

classes = ('Arabesquae', 'Grand_Plié', 'Pirouette')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
# dataiter = iter(train_loader)
images, labels = next(iter(train_loader))
print(type(images))

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


"""
    If you run the script and it returns the following 3 outputs, then your dataloaders are working.

    1. displays the images
    2. Torch size such as ([2 ,1 ,x, x)] where 2 is the batch size of images that are being loaded,
        1 is the grayscale images and x*x is the image size.
    3. Classes of the images
    """