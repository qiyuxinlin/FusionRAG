import torch
from model.quant import relative_l2_error
from model.modeling_deepseek import DeepseekV2MLP

@torch.inference_mode()
def test(quant: str, hidden_size=5120, intermediate_size=1536):
    class Config:
        def __init__(self):
            self.hidden_act = 'silu'
    mlp = DeepseekV2MLP(config=Config(), hidden_size=hidden_size, intermediate_size=intermediate_size, quant=quant)
    for param in mlp.parameters():
        if param.dtype.is_floating_point:
            param.random_(-100, 100)
        else:
            param.random_(0, 128)
    x = torch.randn(hidden_size, dtype=torch.bfloat16)
    y1 = mlp.forward(x)
    y2 = mlp.forward_cpu(x)
    error = relative_l2_error(y1, y2)
    print('Relative error:', relative_l2_error(y1, y2))
    assert error < 0.2

for quant in ['bf16', 'quint8']:
    print(f'Testing {quant} ...');
    test(quant)
