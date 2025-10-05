import torch
import torch.nn.functional as F
import time
import pandas as pd
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt

# Import your implementations
try:
    from minimal_flash_attention import MinimalFlashAttention
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠ Triton not available. Install with 'pip install triton'")

try:
    import flash_attention_cuda
    from test_cuda_flash_attention import CUDAFlashAttention
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠ CUDA extension not available. Run 'python setup.py install' to build it.")

@dataclass
class Config:
    batch_size: int = 1
    num_heads: int = 32
    seq_len: int = 2048
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    causal: bool = True
    device: str = "cuda"

class ComprehensiveBenchmark:
    def __init__(self):
        self.results = []

    def profile_implementation(self, impl_name: str, forward_fn, config: Config,
                             num_warmup: int = 10, num_iter: int = 100):
        """Profile a specific implementation"""

        # Create inputs
        q = torch.randn(config.batch_size, config.num_heads, config.seq_len,
                       config.head_dim, dtype=config.dtype, device=config.device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(num_warmup):
            try:
                _ = forward_fn(q, k, v)
            except Exception as e:
                print(f"Error during warmup for {impl_name}: {e}")
                return None

        torch.cuda.synchronize()

        # Measure
        torch.cuda.reset_peak_memory_stats()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iter)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iter)]

        for i in range(num_iter):
            start_events[i].record()
            out = forward_fn(q, k, v)
            end_events[i].record()

        torch.cuda.synchronize()

        # Collect times
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        # Memory stats
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        # Compute FLOPs
        flops = self._compute_attention_flops(config)
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        tflops = (flops / avg_time) / 1e9  # TFLOPS

        result = {
            'implementation': impl_name,
            'seq_len': config.seq_len,
            'batch_size': config.batch_size,
            'num_heads': config.num_heads,
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'peak_memory_gb': peak_memory,
            'tflops': tflops
        }

        self.results.append(result)
        return result

    def _compute_attention_flops(self, config: Config) -> float:
        """Calculate theoretical FLOPs for attention"""
        b, h, s, d = config.batch_size, config.num_heads, config.seq_len, config.head_dim

        qk_flops = 2 * b * h * s * s * d
        softmax_flops = 3 * b * h * s * s
        sv_flops = 2 * b * h * s * s * d

        return qk_flops + softmax_flops + sv_flops

    def benchmark_all(self, seq_lengths: List[int] = None):
        """Benchmark all implementations across sequence lengths"""

        if seq_lengths is None:
            seq_lengths = [512, 1024, 2048, 4096]

        for seq_len in seq_lengths:
            print(f"\n{'='*60}")
            print(f"Benchmarking sequence length: {seq_len}")
            print(f"{'='*60}")

            config = Config(seq_len=seq_len)

            # 1. PyTorch Baseline (Flash Attention)
            print(f"\n1. PyTorch Flash Attention...")
            def torch_flash(q, k, v):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=False
                ):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=config.causal)

            try:
                result = self.profile_implementation("pytorch_flash", torch_flash, config)
                if result:
                    print(f"   ✓ Time: {result['avg_time_ms']:.3f} ms, TFLOPS: {result['tflops']:.2f}")
            except torch.cuda.OutOfMemoryError:
                print(f"   ✗ OOM at seq_len={seq_len}")
                continue

            # 2. Triton Implementation
            if TRITON_AVAILABLE:
                print(f"\n2. Triton Flash Attention...")
                triton_attn = MinimalFlashAttention()
                try:
                    result = self.profile_implementation("triton_flash", triton_attn.forward, config)
                    if result:
                        print(f"   ✓ Time: {result['avg_time_ms']:.3f} ms, TFLOPS: {result['tflops']:.2f}")
                except Exception as e:
                    print(f"   ✗ Error: {e}")

            # 3. CUDA Implementation (if available)
            if CUDA_AVAILABLE:
                print(f"\n3. CUDA Flash Attention...")
                cuda_attn = CUDAFlashAttention()
                try:
                    result = self.profile_implementation("cuda_flash", cuda_attn.forward, config)
                    if result:
                        print(f"   ✓ Time: {result['avg_time_ms']:.3f} ms, TFLOPS: {result['tflops']:.2f}")
                except Exception as e:
                    print(f"   ✗ Error: {e}")

        return pd.DataFrame(self.results)

    def print_summary(self):
        """Print comparison summary"""
        df = pd.DataFrame(self.results)

        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}\n")

        # Group by sequence length
        for seq_len in df['seq_len'].unique():
            seq_df = df[df['seq_len'] == seq_len].copy()
            print(f"\nSequence Length: {seq_len}")
            print("-" * 80)

            # Find baseline (PyTorch)
            baseline_time = seq_df[seq_df['implementation'] == 'pytorch_flash']['avg_time_ms'].values

            if len(baseline_time) > 0:
                baseline_time = baseline_time[0]
                seq_df['speedup'] = baseline_time / seq_df['avg_time_ms']
            else:
                seq_df['speedup'] = 1.0

            # Print comparison
            print(f"{'Implementation':<20} {'Time (ms)':<12} {'Speedup':<10} {'TFLOPS':<10} {'Memory (GB)':<12}")
            print("-" * 80)

            for _, row in seq_df.iterrows():
                speedup_str = f"{row['speedup']:.2f}x" if row['speedup'] != 1.0 else "baseline"
                print(f"{row['implementation']:<20} {row['avg_time_ms']:>10.3f}  "
                      f"{speedup_str:>10}  {row['tflops']:>8.2f}  {row['peak_memory_gb']:>10.3f}")

        return df

    def plot_comparison(self):
        """Generate comparison plots"""
        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        implementations = df['implementation'].unique()
        colors = ['blue', 'red', 'green', 'orange']

        # Plot 1: Latency
        for impl, color in zip(implementations, colors):
            impl_df = df[df['implementation'] == impl]
            axes[0].plot(impl_df['seq_len'], impl_df['avg_time_ms'],
                        marker='o', label=impl, color=color)

        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Latency Comparison')
        axes[0].set_xscale('log', base=2)
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Speedup vs PyTorch baseline
        baseline = df[df['implementation'] == 'pytorch_flash']
        for impl, color in zip(implementations, colors):
            if impl == 'pytorch_flash':
                continue
            impl_df = df[df['implementation'] == impl]
            speedups = []
            seq_lens = []
            for seq_len in impl_df['seq_len'].values:
                baseline_time = baseline[baseline['seq_len'] == seq_len]['avg_time_ms'].values
                impl_time = impl_df[impl_df['seq_len'] == seq_len]['avg_time_ms'].values
                if len(baseline_time) > 0 and len(impl_time) > 0:
                    speedups.append(baseline_time[0] / impl_time[0])
                    seq_lens.append(seq_len)

            if speedups:
                axes[1].plot(seq_lens, speedups, marker='s', label=impl, color=color)

        axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Speedup vs PyTorch')
        axes[1].set_title('Relative Performance')
        axes[1].set_xscale('log', base=2)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: TFLOPS
        for impl, color in zip(implementations, colors):
            impl_df = df[df['implementation'] == impl]
            axes[2].plot(impl_df['seq_len'], impl_df['tflops'],
                        marker='^', label=impl, color=color)

        axes[2].set_xlabel('Sequence Length')
        axes[2].set_ylabel('TFLOPS')
        axes[2].set_title('Compute Efficiency')
        axes[2].set_xscale('log', base=2)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('flash_attention_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: flash_attention_comparison.png")

        return fig

if __name__ == "__main__":
    print("🚀 Flash Attention Implementation Comparison\n")

    benchmark = ComprehensiveBenchmark()

    # Run benchmarks
    df = benchmark.benchmark_all()

    # Print summary
    df = benchmark.print_summary()

    # Generate plots
    benchmark.plot_comparison()

    # Save results
    df.to_csv('comparison_results.csv', index=False)
    print(f"\n💾 Results saved to: comparison_results.csv")
