import os

import click
import torch
from src.models.model import MyAwesomeModel
from torch.utils.data import DataLoader
from src.data.custom_Dataset import ImageFolderCustom 

@click.command()
@click.option("--checkpoint", default=os.path.join('models', 'my_trained_model.pt'), help='Path to file with state dict of the model')
def evaluate(checkpoint):

    print("Predicting day and night")

    # TODO: Implement evaluating loop here
    model = MyAwesomeModel()
    
    testset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=False)
    # Test DataLoader
    testloader = DataLoader(dataset=testset, # use custom created train Dataset
                                        batch_size=64, # how many samples per batch?
                                        shuffle=True) # shuffle the data?
    
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    test_accuracies = []

    with torch.no_grad():
         # Set model to evaluation mode to turn off dropout
        model.eval()
        for images, labels in testloader:
            # Flatten images into a 784 long vector
            # images = images.view(images.shape[0], -1)
            # Caculate log probabilities
            log_ps = model(images)
            # Calculate probabilities
            ps = torch.exp(log_ps)
            # Obtain top probabilities and corresponding classes
            top_p, top_class = ps.topk(1, dim=1)
            # Compare with true labels
            equals = top_class == labels.view(*top_class.shape)   
            # Obtain accuracy
            batch_acccuracy = torch.mean(equals.type(torch.FloatTensor))       
            test_accuracies.append(batch_acccuracy) 

    accuracy = sum(test_accuracies)/len(testloader)  

    print(f'Accuracy on test: {accuracy.item()*100}%')

    
if __name__ == "__main__":
    evaluate()
