from src.models.model import MyAwesomeModel
import torch
from torch import nn
import numpy as np

def test_output():

    model = MyAwesomeModel()

    input = torch.rand((64, 1, 28, 28))

    output = model(input)

    # Assert output of model has expected shape
    assert output.shape == torch.Size([64, 10]), f"Model output shape not as expected (Expected: {torch.Size([64, 10])}, Current:{output.shape})"