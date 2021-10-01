import torch
from torchvision import  transforms
from torchvision import datasets, transforms

root_dir = '..\..\Dataset\MVTEC_AD'
batchsize = 64

# apply the transformation to both train and test data
train_data = datasets.ImageFolder(root_dir + '/Train', transform=transforms.Compose([transforms.Resize((900,900)), #transforms.Resize() uses PIL.Image.BILINEAR interpolation by default.
                                                                            transforms.ToTensor()]))

test_data = datasets.ImageFolder(root_dir + '/Test', transform=transforms.ToTensor())

# create the dataloaders
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batchsize,
                                        shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=batchsize,
                                        shuffle=False)

def get_mean_std(loader):

    # VAR[X] = E[X**2] - E[X]**2
    # std = sqrt(VAR)
    # mean = E[X]
    # channels_squared_sum = E[X**2]

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:

        """ dim 0 = batches
            dim 1 = channels --> we don't do accross this dimension and thus works for greyscale and rgb
            dim 2 = Height
            dim 3 = Width 
        """
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std

mean, std = get_mean_std(train_loader)
print(mean)
print(std)

# Result:
# mean: tensor([0.4318, 0.4012, 0.3913])
# std:  tensor([0.2597, 0.2561, 0.2525])