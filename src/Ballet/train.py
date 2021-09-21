from model import CNN
import datahandler as dh

import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
import matplotlib.pyplot as plt



# Hyperparameters that we can tune
epochs = 1000
batch_size = 32
learning_rate = 0.0005
print_every = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: ", device)



train_loader, test_loader = dh.pre_processor(
                                '../../../Dataset',
                                batchsize= batch_size)

classes = ('Arabesquae', 'Grand_Plié', 'Pirouette')


model = CNN().to(device)

# print(next(model.parameters()).is_cuda)

# (summary(model, (1, 90, 160))) # prints the summary of the model. 90,160 is image size


# define the optimizer, scheduler and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
criterion = nn.CrossEntropyLoss()


#    ******************** Start the training loop here ********************     #

train_losses = []
val_losses = []
total_acc = []
n_total_steps = len(train_loader)
best_val = 2

for e in range(epochs):

    running_loss = 0

    for i, (images, labels) in enumerate(iter(train_loader)):
        
        images = torch.stack(images).to(device)
        labels = labels.to(device)

        # reset the grdients
        optimizer.zero_grad()
        
        output = model(images)              # 1) Forward pass
        loss = criterion(output, labels)    # 2) Compute loss

        loss.backward()                     # 3) Backward pass
        optimizer.step()                    # 4) Update model
        
        running_loss += loss.item()
        
    if e%2 == 0:
        # print('Entering val loop')
        train_losses.append(running_loss/images.shape[0])
            
        correct = 0
        total = 0
        val_loss = 0

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():

        # torch.inference is torch.no_grad on steroids
        
        # with torch.inference_mode():
        # InferenceMode is a new context manager analogous to no_grad to be used when you are certain your 
        # operations will have no interactions with autograd (e.g., model training). Code run under this mode gets better 
        # performance by disabling view tracking and version counter bumps.

            for i, (images, labels) in enumerate(iter(test_loader)):
                # print(i)
                images = torch.stack(images).to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss_val = criterion(outputs, labels)
                val_loss += loss_val.item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss =  val_loss/images.shape[0]
        val_losses.append(val_loss)

        acc = 100 * correct / total
        total_acc.append(acc)

        print('Epoch : ',e, "\t Train loss: {:.2f}".format(running_loss/images.shape[0]),
            "\t Validation loss: {:.2f}".format(val_loss), "\t Accuracy: {:.2f} %" .format(acc))

        # Save the model only if the val_loss decreases w.r.t previous loss
        if val_loss < best_val:
            best_val = val_loss
            PATH = '../../models/ballet_1000epochs_cnn.pth'
            torch.save(model.state_dict(), PATH)
            print("MODEL HAS BEEN SAVED")

    scheduler.step()

# plot and save the losses
fig = plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label = "train")
plt.plot(val_losses, label = "val")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
fig.savefig('Losses_1kepochs.png')


# plot and save the accuracy
fig = plt.figure(figsize=(10,5))
plt.title("Accuracy")
plt.plot(total_acc, label = "acc")
plt.xlabel("iterations")
plt.ylabel("Acc.")
plt.legend()
fig.savefig('Accuracy_1kepochs.png')