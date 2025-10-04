import torch
import torch.nn.functional as F
import time
import pandas as pd
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt

@dataclass
class Config:
    batch_size: int = 1
    num_heads: int = 32
    seq_len: int = 2048
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    causal: bool = True
    device: str = "cuda"

class AttentionProfiler:
    def __init__(self):
        self.results = []

    def profile_torch_sdpa(self, config: Config, num_warmup: int = 10, num_iter: int = 100):
        """Profile PyTorch's scaled_dot_product_attention"""

        # Create inputs
        q = torch.randn(config.batch_size, config.num_heads, config.seq_len,
                       config.head_dim, dtype=config.dtype, device=config.device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(num_warmup):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=config.causal)

        torch.cuda.synchronize()

        # Measure
        torch.cuda.reset_peak_memory_stats()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iter)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iter)]

        for i in range(num_iter):
            start_events[i].record()
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=config.causal)
            end_events[i].record()

        torch.cuda.synchronize()

        # Collect times
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        # Memory stats
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB

        # Compute FLOPs
        flops = self._compute_attention_flops(config)
        avg_time = sum(times) / len(times)
        tflops = (flops / avg_time) / 1e9  # TFLOPS

        result = {
            'implementation': 'torch_flash',
            'seq_len': config.seq_len,
            'batch_size': config.batch_size,
            'num_heads': config.num_heads,
            'avg_time_ms': avg_time,
            'peak_memory_gb': peak_memory,
            'tflops': tflops
        }

        self.results.append(result)
        return result

    def _compute_attention_flops(self, config: Config) -> float:
        """Calculate theoretical FLOPs for attention"""
        # QK^T: 2 * batch * heads * seq * seq * dim
        # Softmax: ~3 * batch * heads * seq * seq (exp, sum, div)
        # Score * V: 2 * batch * heads * seq * seq * dim

        b, h, s, d = config.batch_size, config.num_heads, config.seq_len, config.head_dim

        qk_flops = 2 * b * h * s * s * d
        softmax_flops = 3 * b * h * s * s
        sv_flops = 2 * b * h * s * s * d

        return qk_flops + softmax_flops + sv_flops

    def sweep_sequence_lengths(self, seq_lengths: List[int] = None):
        """Benchmark across different sequence lengths"""

        if seq_lengths is None:
            seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

        for seq_len in seq_lengths:
            print(f"Profiling seq_len={seq_len}")
            config = Config(seq_len=seq_len)
            try:
                self.profile_torch_sdpa(config)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at seq_len={seq_len}")
                break

        return pd.DataFrame(self.results)

    def plot_results(self):
        """Generate performance plots"""

        df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Latency vs sequence length
        axes[0].plot(df['seq_len'], df['avg_time_ms'], marker='o')
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Attention Latency Scaling')
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True, alpha=0.3)

        # Memory vs sequence length
        axes[1].plot(df['seq_len'], df['peak_memory_gb'], marker='s', color='red')
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Peak Memory (GB)')
        axes[1].set_title('Memory Usage Scaling')
        axes[1].set_xscale('log', base=2)
        axes[1].grid(True, alpha=0.3)

        # TFLOPS vs sequence length
        axes[2].plot(df['seq_len'], df['tflops'], marker='^', color='green')
        axes[2].set_xlabel('Sequence Length')
        axes[2].set_ylabel('TFLOPS')
        axes[2].set_title('Compute Efficiency')
        axes[2].set_xscale('log', base=2)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('baseline_performance.png', dpi=150)
        plt.show()

        return fig

if __name__ == "__main__":
    profiler = AttentionProfiler()

    # Run baseline measurements
    df = profiler.sweep_sequence_lengths()
    print("\nBaseline Results:")
    print(df.to_string())

    # Generate plots
    profiler.plot_results()

    # Save results
    df.to_csv('baseline_results.csv', index=False)
