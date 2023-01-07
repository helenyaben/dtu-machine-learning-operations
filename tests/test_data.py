from tests import _PATH_DATA
from src.data.custom_Dataset import ImageFolderCustom 
import os
import torch

def test_dataset():

    N_train = 25000
    N_test = 5000

    trainset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=True)
    testset = ImageFolderCustom(targ_dir=os.path.join(os.getcwd(), 'data', 'processed'), train=False)

    data_classes = set(range(0, 10))

    # Assert that the number of train and test loaded elements is as expected
    assert len(trainset) == N_train,  f"Train set did not have the correct number of samples (Expected: {N_train}, Current: {len(trainset)})"
    assert len(testset) == N_test,  f"Test set did not have the correct number of samples (Expected: {N_test}, Current:{len(testset)}"

    # Assert that the shape of the loaded images is as expected and that label corresponds to set of possible classes
    for dataset in [trainset, testset]:
        labels = []
        for ix in range(0, len(dataset)):
            img, label = dataset.__getitem__(ix)
            labels.append(label)
            assert img.shape == torch.Size([1, 28, 28]), f"Image did not have the correct shape (Expected: {torch.Size([1, 28, 28])}, Current:{img.shape})"
        # Assert that all classes are represented
        assert set(labels) == data_classes, f"Set of labels is not as expected (Expected: {data_classes}, Current:{set(labels)})"


