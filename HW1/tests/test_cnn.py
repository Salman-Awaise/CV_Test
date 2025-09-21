import torch
from models.cnn import ColorConstancyCNN

def test_cnn_forward_shape():
    model = ColorConstancyCNN()
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 3)
    assert torch.allclose(torch.norm(out, dim=1), torch.ones(2), atol=1e-4)
