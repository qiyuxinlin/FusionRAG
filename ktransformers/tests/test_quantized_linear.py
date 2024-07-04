import torch

from model import quant
from model.quant import relative_l2_error

@torch.inference_mode()
def test(in_features:int = 512, out_features:int = 1024, qtype = 'quint8'):
    print(f'Testing {qdtype}, in_features={in_features}, out_features={out_features}')
    linear = quant.QuantizedLinear.new(in_features, out_features, bias=False, qdtype=qtype)
    for param in linear.parameters():
        if param.dtype.is_floating_point:
            param.random_(-100, 100)
        else:
            param.random_(0, 128)
    x = torch.randn(in_features, dtype=torch.bfloat16)
    y1 = linear.forward_vanilla(x)
    y2 = linear.forward(x, allow_fallback=False)
    error = relative_l2_error(y1, y2)
    print('Relative error:', error)
    assert error < 0.1

for qdtype in ['bf16', 'quint8']:
    test(512, 1024, qdtype)
    test(5120, 1536, qdtype)
    test(1536, 5120, qdtype)
    