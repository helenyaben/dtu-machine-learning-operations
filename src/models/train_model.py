import os
import pathlib
from typing import Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.pyplot import show
from src.models.model import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.data.custom_Dataset import ImageFolderCustom 

import wandb

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):

    wandb.init(project="test_on_mnist", entity="dtu_mlops_2023", config={
    "learning_rate": lr,
    "epochs": 30,
    "batch_size": 64
    })

    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    
    trainset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=True)
    # Train DataLoader
    trainloader = DataLoader(dataset=trainset, # use custom created train Dataset
                                        batch_size=64, # how many samples per batch?
                                        shuffle=True) # shuffle the data?
    epochs = 30
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    # wandb_results_table = wandb.Table()
    
    for e in range(epochs):

        epoch_losses = 0

        for images, labels in trainloader:
            # Flatten images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            # Reset gradients
            optimizer.zero_grad()
            # Obtain log probabilities
            log_ps = model(images)
            # Calculate loss
            loss = criterion(log_ps, labels)
            # Apply backward
            loss.backward()
            # Move optimizer 
            optimizer.step()
            # Add batch loss to epoch losses list
            epoch_losses += loss.item()

            # if e == 1:
                
            #     wandb.log({"examples":[wandb.Image(im.squeeze(0).numpy()) for im in images[:15]]})
                # wandb_results_table.add_column("image", [wandb.Image(im.squeeze(0).numpy()) for im in images])
                # wandb_results_table.add_column("label", labels.numpy())
        
        train_losses.append(epoch_losses/len(trainloader))
        wandb.log({"loss": epoch_losses/len(trainloader), "epoch": e})
        print(f"Train loss in epoch {e}: {epoch_losses/len(trainloader)}")

    # wandb_results_table.add_column("prediction", )

    torch.save(model.state_dict(), os.path.join('models','my_trained_model.pt')) 

    print('Trained model saved in /models/my_trained_model.pt')
    
    # plt.plot(train_losses)
    # show()
    
if __name__ == "__main__":
    train()

    