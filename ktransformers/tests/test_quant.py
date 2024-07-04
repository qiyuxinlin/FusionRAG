import itertools
import torch

from model import quant

@torch.inference_mode()
def test(in_features:int = 512, out_features:int = 1024, dtype = torch.bfloat16, qtype = 'quint8'):
    print(f'Testing {dtype} -> {qdtype}, in_features={in_features}, out_features={out_features}')
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=dtype)
    qlinear = quant.QuantizedLinear.quantize(linear, qtype)
    x = torch.randn(in_features, dtype=dtype)
    werror = quant.relative_l2_error(linear.weight, qlinear.dequantize_weight().to(dtype=linear.weight.dtype))
    print('Weight error:', werror)
    assert werror < 0.05
    px = linear(x)
    pqx = qlinear.forward_vanilla(x)
    perror = quant.relative_l2_error(px, pqx)
    print('Projection error:', perror)
    assert perror < 0.05
    quantized_pqx = qlinear.forward(x, allow_fallback=False)
    qperror = quant.relative_l2_error(px, quantized_pqx)
    print('Quantized projection error:', perror)
    assert qperror < 0.05

for dtype, qdtype in itertools.product([torch.bfloat16, torch.float32], ['quint8', 'bf16']):
    test(512, 1024, dtype, qdtype)
    test(5120, 1536, dtype, qdtype)
    test(1536, 5120, dtype, qdtype)
    