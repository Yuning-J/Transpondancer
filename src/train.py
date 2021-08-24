from model import CNN
import datahandler as dh

import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary


# Hyperparameters. We can tune these as we validate the results
epochs = 5
batch_size = 32
learning_rate = 0.001
print_every = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: ", device)



train_loader, test_loader = dh.pre_processor(
                                'path_to_dataset',
                                batch_size= batch_size)

classes = ('arabesque', 'Pirouette', 'Battement', 'Gran_Pliet', 'Assemble')


model = CNN().to(device)

# (summary(model, (1, x, x))) # prints the summary of the model. x, x is image size


optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#scheduler = ReduceLROnPlateau(optimizer, 'max', factor = 0.5, verbose=True)
criterion = nn.CrossEntropyLoss()


#    ******************** Start the training loop here ********************     #