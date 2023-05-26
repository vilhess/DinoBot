import pandas as pd
from dataset import DinoDataset, transformer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torchvision.models
import torch.optim as optim
from tqdm import tqdm
import gc
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from inceptionV3 import *

path = 'efficientnet_v2_s.pth'  # for inception v2 from torch

key_frame = pd.read_csv('labels_dino.csv')
train, test = train_test_split(key_frame, test_size=0.2)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

batch_size = 8

trainset = DinoDataset(root_dir="captures",
                       dataframe=train, transform=transformer)
trainloader = DataLoader(trainset, batch_size=batch_size)

testset = DinoDataset(root_dir="captures",
                      dataframe=test, transform=transformer)
testloader = DataLoader(testset, batch_size=batch_size)

device = 'mps'
model = torchvision.models.efficientnet_v2_s()
model.classifier = torch.nn.Linear(in_features=1280, out_features=2)

model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.009)


def train_model(model, criterion, optimizer, num_epochs=25):
    epochs = 15  # number of training passes over the mini batches
    loss_container = []  # container to store the loss values after each epoch
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for data in tqdm(trainloader, position=0, leave=True):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_container.append(running_loss)

        print(f'[{epoch + 1}] | loss: {running_loss / len(trainloader):.3f}')
        running_loss = 0.0

    print('Finished Training')

    torch.save(model.state_dict(), path)

    # plot the loss curve
    plt.plot(np.linspace(1, epochs, epochs).astype(int), loss_container)
    plt.show()

    # clean up the gpu memory
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train_model(model, criterion, optimizer)
