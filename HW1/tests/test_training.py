import torch
from models.cnn import ColorConstancyCNN
from models.training import angular_error_loss

def test_angular_error_loss():
    pred = torch.tensor([[1.0,0,0],[0,1,0]])
    true = torch.tensor([[1.0,0,0],[1,0,0]])
    err = angular_error_loss(pred, true)
    assert err.shape == ()
    assert err > 0
