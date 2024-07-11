import fire
import torch
import itertools
import time
from model.modeling_deepseek import DeepseekV2MLP

@torch.inference_mode()
def stress(quant: str, hidden_size=5120, intermediate_size=1536, num_mlps=100, warmup_cycle=10000, test_cycle=10000):
    class Config:
        def __init__(self):
            self.hidden_act = 'silu'
    def create_mlp():
        mlp = DeepseekV2MLP(config=Config(), hidden_size=hidden_size, intermediate_size=intermediate_size, quant=quant)
        for param in mlp.parameters():
            if param.dtype.is_floating_point:
                param.random_(-100, 100)
            else:
                param.random_(0, 128)
        return mlp
    mlps = [create_mlp() for _ in range(num_mlps)]
    x = torch.randn(hidden_size, dtype=torch.bfloat16)
    def cycle(count):
        for mlp in itertools.islice(itertools.cycle(mlps), count):
            _ = mlp.forward_cpu(x)

    # warmup
    cycle(warmup_cycle)
    # test
    start = time.perf_counter()
    cycle(test_cycle)
    end = time.perf_counter()
    elements = hidden_size * intermediate_size * 3 * test_cycle
    print(f"throughput: {elements/(end-start):.4E} elts/s")

def main(quant: str = 'bf16', hidden_size: int = 5120, intermediate_size: int = 1536, num_mlps: int = 100,
         warmup_cycle: int = 10000, test_cycle: int = 10000):
    stress(quant, hidden_size, intermediate_size, num_mlps, warmup_cycle, test_cycle)

fire.Fire(main)
