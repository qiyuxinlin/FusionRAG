echo "bench_linear"
numactl -N 1 python bench_linear.py
echo "bench_mlp"
numactl -N 1 python bench_mlp.py
echo "bench_moe"
numactl -N 1 python bench_moe.py

echo "bench_linear_torch"
numactl -N 1 python bench_linear_torch.py
echo "bench_mlp_torch"
numactl -N 1 python bench_mlp_torch.py
echo "bench_moe_torch"
numactl -N 1 python bench_moe_torch.py