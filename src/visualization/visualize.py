import importlib.util
import os
import sys

import click
import torch

# Import model class
spec = importlib.util.spec_from_file_location("mnist", os.path.join('src', 'models', 'model.py'))
foo = importlib.util.module_from_spec(spec)
sys.modules["mnist"] = foo
spec.loader.exec_module(foo) 

from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import show
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from src.data.custom_Dataset import ImageFolderCustom

# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

@click.command()
@click.option("--checkpoint", default=os.path.join('src', 'models', 'my_trained_model.pt'), help='Path to file with state dict of the model')
def visualize(checkpoint):

    print("Visualizing day and night")

    # TODO: Implement evaluating loop here
    model = foo.MyAwesomeModel()

    # Register hooks on each layer
    hookF = [Hook(layer[1]) for layer in list(model._modules.items())]    
    
    trainset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=False)
    # Test DataLoader
    trainloader = DataLoader(dataset=trainset, # use custom created train Dataset
                                        batch_size=64, # how many samples per batch?
                                        shuffle=True) # shuffle the data?
    
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    with torch.no_grad():
    # Set model to evaluation mode to turn off dropout
        model.eval()
        for images, labels in trainloader:
            # Caculate log probabilities
            log_ps = model(images)
            # T-SNE representations of layer outputs
            fig, axes = plt.subplots(2, 2, figsize=(10,10), )
            axes = axes.flatten()
            fig.suptitle('2D T-SNE representation of layer outputs for first batch')
            for ix, layer in enumerate(list(model._modules.items())[:-1]):
                tsne = TSNE(n_components=2, verbose=0, random_state=123)
                z= tsne.fit_transform(hookF[ix].output.numpy()) 
                df = pd.DataFrame()
                df["y"] = labels
                df["comp-1"] = z[:,0]
                df["comp-2"] = z[:,1]

                sns.scatterplot(ax=axes[ix], x="comp-1", y="comp-2", hue='y',
                                palette=sns.color_palette("hls", 10),
                                data=df, legend="full").set(title=f"MNIST data: T-SNE projection of {layer[0]} output") 
                
                
                axes[ix].get_legend().remove()
            
            lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
            lines, labels = zip(*lines_labels)
            fig.legend(lines[0], labels[0], loc='center left', bbox_to_anchor=(1, 0.5), )
            plt.tight_layout()
            plt.savefig(os.path.join('reports', 'figures', 'tsne_train.png'),  bbox_inches='tight')
            
            break
        
    
if __name__ == "__main__":
    visualize()
