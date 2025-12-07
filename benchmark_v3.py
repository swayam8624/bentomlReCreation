import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import asyncio
import time
import json
import statistics
import random
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    import aiohttp
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetName,
        nvmlDeviceGetClockInfo, nvmlDeviceGetTemperature,
        NVML_CLOCK_GRAPHICS, NVML_TEMPERATURE_GPU
    )
except ImportError as e:
    print(f"FATAL: Missing dependency: {e}")
    print("Run: pip install aiohttp numpy nvidia-ml-py matplotlib")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GPU_ID = 0
POLLING_INTERVAL_SEC = 0.1  # 10Hz for cleaner data
REQUEST_TIMEOUT = 60
MODEL_PARAMS = 8e9  # Llama-8B
BACKEND_URL = "http://127.0.0.1:8888"

# Test Configurations
CONCURRENT_USERS = [10, 50, 100]  # Compare across different loads
REQUESTS_PER_USER = 5
WARMUP_REQUESTS = 20

# ═══════════════════════════════════════════════════════════════════════════
# GPU DATABASE
# ═══════════════════════════════════════════════════════════════════════════
GPU_SPECS = {
    "NVIDIA A100-SXM4-80GB": 312e12,
    "NVIDIA A100-PCIE-80GB": 312e12,
    "NVIDIA A100-SXM4-40GB": 312e12,
    "NVIDIA A40": 149.7e12,
    "NVIDIA A40-PCIE-48GB": 149.7e12,
    "NVIDIA H100-SXM5-80GB": 989e12,
    "NVIDIA H100-PCIE-80GB": 756e12,
    "Tesla V100-SXM2-32GB": 125e12,
    "Tesla V100-PCIE-32GB": 112e12,
    "NVIDIA GeForce RTX 4090": 165e12,
    "NVIDIA GeForce RTX 4080": 96e12,
    "NVIDIA GeForce RTX 3090": 71e12,
}

@dataclass
class GPUState:
    timestamp: float
    utilization_kernel: float  # NVML busy %
    memory_used_mb: float
    power_draw_w: float
    clock_sm_mhz: float
    temperature_c: float

@dataclass
class RequestMetric:
    request_id: str
    concurrent_level: int
    start_time: float
    end_time: float
    ttft: float  # Time to first token
    total_latency_ms: float
    output_tokens: int
    tokens_per_sec: float  # Individual request throughput
    success: bool
    error: str = ""

@dataclass
class LoadTestResult:
    concurrent_users: int
    total_requests: int
    successful_requests: int
    duration: float
    
    # Latency metrics
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput metrics
    aggregate_tokens_per_sec: float
    avg_request_tokens_per_sec: float
    p50_request_tokens_per_sec: float
    
    # Efficiency metrics
    real_mfu: float  # Calculated from actual throughput
    avg_power_w: float
    tokens_per_watt: float

# ═══════════════════════════════════════════════════════════════════════════
# GPU MONITOR
# ═══════════════════════════════════════════════════════════════════════════
class GPUMonitor:
    def __init__(self, device_index: int = 0):
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(device_index)
            self.gpu_name = nvmlDeviceGetName(self.handle)
            self.peak_flops = GPU_SPECS.get(self.gpu_name, None)
            self.running = False
            self.history: List[GPUState] = []
            print(f"✓ GPU Detected: {self.gpu_name}")
            if self.peak_flops:
                print(f"✓ Peak FP16 Tensor FLOPS: {self.peak_flops/1e12:.1f} TFLOPS")
            else:
                print(f"⚠ GPU not in database - add to GPU_SPECS dict")
        except Exception as e:
            print(f"✗ Failed to initialize NVML: {e}")
            self.handle = None
            self.gpu_name = "Unknown"
            self.peak_flops = None

    async def start_monitoring(self):
        if not self.handle:
            return
        self.running = True
        self.history = []
        
        while self.running:
            try:
                util = nvmlDeviceGetUtilizationRates(self.handle)
                mem = nvmlDeviceGetMemoryInfo(self.handle)
                power = nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_GRAPHICS)
                temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
                
                self.history.append(GPUState(
                    timestamp=time.time(),
                    utilization_kernel=util.gpu,  # Renamed to clarify
                    memory_used_mb=mem.used / 1024**2,
                    power_draw_w=power,
                    clock_sm_mhz=clock,
                    temperature_c=temp
                ))
            except Exception:
                pass
            await asyncio.sleep(POLLING_INTERVAL_SEC)

    def stop(self):
        self.running = False

    def get_avg_power(self) -> float:
        if not self.history:
            return 0
        return statistics.mean([s.power_draw_w for s in self.history])

# ═══════════════════════════════════════════════════════════════════════════
# LOAD GENERATOR
# ═══════════════════════════════════════════════════════════════════════════
class LoadGenerator:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url.rstrip("/")
        self.prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to reverse a string.",
            "What are the benefits of exercise?",
            "Describe the water cycle.",
            "How does photosynthesis work?",
            "Explain machine learning to a beginner.",
            "What causes seasons on Earth?",
            "Write a haiku about technology.",
        ] * 10

    async def _send_request(self, session: aiohttp.ClientSession, req_id: str, 
                           concurrent_level: int) -> RequestMetric:
        prompt = random.choice(self.prompts)
        start = time.time()
        ttft = 0
        
        payload = {
            "model": "/workspace/models/llama-8b-hf",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.7,
            "stream": False
        }
        url = f"{self.backend_url}/v1/chat/completions"
        
        try:
            async with session.post(url, json=payload, timeout=REQUEST_TIMEOUT) as resp:
                # For non-streaming, TTFT ≈ total time (we get response all at once)
                ttft = time.time() - start
                
                text = await resp.text()
                end = time.time()
                
                if resp.status != 200:
                    return RequestMetric(
                        str(req_id), concurrent_level, start, end, 
                        ttft * 1000, (end-start)*1000, 0, 0, False,
                        error=f"HTTP {resp.status}"
                    )
                
                data = json.loads(text)
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                
                if not tokens:
                    try:
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            tokens = max(1, len(content.split()))
                    except:
                        tokens = 1
                
                total_latency = (end - start) * 1000  # ms
                tokens_per_sec = tokens / (end - start) if (end - start) > 0 else 0
                
                return RequestMetric(
                    request_id=str(req_id),
                    concurrent_level=concurrent_level,
                    start_time=start,
                    end_time=end,
                    ttft=ttft * 1000,  # ms
                    total_latency_ms=total_latency,
                    output_tokens=tokens,
                    tokens_per_sec=tokens_per_sec,
                    success=True
                )
        except Exception as e:
            return RequestMetric(
                str(req_id), concurrent_level, start, time.time(),
                0, (time.time()-start)*1000, 0, 0, False, error=str(e)
            )

    async def run_concurrent_test(self, num_users: int, 
                                  requests_per_user: int) -> List[RequestMetric]:
        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║  LOAD TEST: {num_users} Concurrent Users                      ")
        print(f"╚══════════════════════════════════════════════════════════╝")
        
        metrics = []
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        semaphore = asyncio.Semaphore(num_users)
        
        async def user_session(user_id):
            async with semaphore:
                user_metrics = []
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    for i in range(requests_per_user):
                        m = await self._send_request(
                            session, f"u{user_id}-r{i}", num_users
                        )
                        user_metrics.append(m)
                        await asyncio.sleep(random.uniform(0.5, 2))  # Think time
                return user_metrics
        
        tasks = [user_session(u) for u in range(num_users)]
        results = await asyncio.gather(*tasks)
        
        for r in results:
            metrics.extend(r)
        
        success_count = sum(1 for m in metrics if m.success)
        print(f"   ✓ Complete: {success_count}/{len(metrics)} successful")
        return metrics

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def analyze_load_test(metrics: List[RequestMetric], avg_power: float, 
                     peak_flops: Optional[float]) -> Optional[LoadTestResult]:
    successful = [m for m in metrics if m.success]
    if not successful:
        print("   ⚠ No successful requests")
        return None
    
    concurrent_users = successful[0].concurrent_level
    duration = max(m.end_time for m in successful) - min(m.start_time for m in successful)
    
    # Latency metrics
    ttfts = [m.ttft for m in successful]
    latencies = [m.total_latency_ms for m in successful]
    
    # Throughput metrics
    total_tokens = sum(m.output_tokens for m in successful)
    aggregate_tps = total_tokens / duration if duration > 0 else 0
    request_tps = [m.tokens_per_sec for m in successful]
    
    # MFU (the REAL GPU utilization)
    real_mfu = 0
    if peak_flops:
        achieved_flops = aggregate_tps * MODEL_PARAMS * 2
        real_mfu = (achieved_flops / peak_flops) * 100
    
    tokens_per_watt = aggregate_tps / avg_power if avg_power > 0 else 0
    
    return LoadTestResult(
        concurrent_users=concurrent_users,
        total_requests=len(metrics),
        successful_requests=len(successful),
        duration=duration,
        
        avg_ttft_ms=statistics.mean(ttfts),
        p50_ttft_ms=np.percentile(ttfts, 50),
        p95_ttft_ms=np.percentile(ttfts, 95),
        p99_ttft_ms=np.percentile(ttfts, 99),
        
        avg_latency_ms=statistics.mean(latencies),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        
        aggregate_tokens_per_sec=aggregate_tps,
        avg_request_tokens_per_sec=statistics.mean(request_tps),
        p50_request_tokens_per_sec=np.percentile(request_tps, 50),
        
        real_mfu=real_mfu,
        avg_power_w=avg_power,
        tokens_per_watt=tokens_per_watt
    )

def print_result(result: LoadTestResult):
    print(f"\n╔═══════════════════════════════════════════════════════════════╗")
    print(f"║  RESULTS: {result.concurrent_users} Concurrent Users                              ║")
    print(f"╠═══════════════════════════════════════════════════════════════╣")
    print(f"║ Requests: {result.successful_requests}/{result.total_requests} successful                                  ║")
    print(f"║ Duration: {result.duration:.1f}s                                            ║")
    print(f"╟───────────────────────────────────────────────────────────────╢")
    print(f"║ TIME TO FIRST TOKEN (TTFT)                                    ║")
    print(f"║   Avg:  {result.avg_ttft_ms:8.1f} ms  | P50: {result.p50_ttft_ms:8.1f} ms             ║")
    print(f"║   P95:  {result.p95_ttft_ms:8.1f} ms  | P99: {result.p99_ttft_ms:8.1f} ms             ║")
    print(f"╟───────────────────────────────────────────────────────────────╢")
    print(f"║ TOTAL LATENCY                                                 ║")
    print(f"║   Avg:  {result.avg_latency_ms:8.1f} ms  | P95: {result.p95_latency_ms:8.1f} ms             ║")
    print(f"║   P99:  {result.p99_latency_ms:8.1f} ms                                     ║")
    print(f"╟───────────────────────────────────────────────────────────────╢")
    print(f"║ THROUGHPUT                                                    ║")
    print(f"║   Aggregate:     {result.aggregate_tokens_per_sec:8.1f} tok/s                      ║")
    print(f"║   Per Request:   {result.avg_request_tokens_per_sec:8.1f} tok/s (avg)                  ║")
    print(f"║   Per Request:   {result.p50_request_tokens_per_sec:8.1f} tok/s (p50)                  ║")
    print(f"╟───────────────────────────────────────────────────────────────╢")
    print(f"║ EFFICIENCY (Real GPU Utilization)                             ║")
    print(f"║   MFU:           {result.real_mfu:6.2f}%                                    ║")
    print(f"║   Power:         {result.avg_power_w:6.1f}W                                    ║")
    print(f"║   Efficiency:    {result.tokens_per_watt:6.2f} tok/W                            ║")
    print(f"╚═══════════════════════════════════════════════════════════════╝\n")

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def plot_comprehensive_analysis(results: List[LoadTestResult], 
                                all_metrics: Dict[int, List[RequestMetric]],
                                gpu_history: List[GPUState],
                                peak_flops: Optional[float]):
    """Generate multi-panel analysis dashboard"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    concurrent_levels = [r.concurrent_users for r in results]
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 1: TTFT Comparison (Top Left)
    # ═══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ttft_avg = [r.avg_ttft_ms for r in results]
    ttft_p95 = [r.p95_ttft_ms for r in results]
    ttft_p99 = [r.p99_ttft_ms for r in results]
    
    x = np.arange(len(concurrent_levels))
    width = 0.25
    
    ax1.bar(x - width, ttft_avg, width, label='Avg TTFT', color='#00FF41', alpha=0.8)
    ax1.bar(x, ttft_p95, width, label='P95 TTFT', color='#FFA500', alpha=0.8)
    ax1.bar(x + width, ttft_p99, width, label='P99 TTFT', color='#FF6B35', alpha=0.8)
    
    ax1.set_xlabel('Concurrent Users', fontweight='bold')
    ax1.set_ylabel('Time to First Token (ms)', fontweight='bold')
    ax1.set_title('TTFT Latency by Concurrency', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(concurrent_levels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 2: Throughput Scaling (Top Center)
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    aggregate_tps = [r.aggregate_tokens_per_sec for r in results]
    per_request_tps = [r.avg_request_tokens_per_sec for r in results]
    
    ax2.plot(concurrent_levels, aggregate_tps, 'o-', linewidth=3, 
             markersize=10, label='Aggregate tok/s', color='#00FF41')
    ax2.plot(concurrent_levels, per_request_tps, 's--', linewidth=2, 
             markersize=8, label='Per-Request tok/s', color='#4A90E2')
    
    ax2.set_xlabel('Concurrent Users', fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax2.set_title('Throughput Scaling', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 3: MFU Comparison (Top Right) - THE KEY METRIC
    # ═══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[0, 2])
    mfu_values = [r.real_mfu for r in results]
    colors = ['#00FF41' if m > 15 else '#FFA500' if m > 10 else '#FF6B35' 
              for m in mfu_values]
    
    bars = ax3.bar(concurrent_levels, mfu_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    for bar, mfu in zip(bars, mfu_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mfu:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Concurrent Users', fontweight='bold')
    ax3.set_ylabel('Real GPU Utilization (MFU %)', fontweight='bold')
    ax3.set_title('Model FLOPS Utilization (Real GPU Usage)', 
                  fontweight='bold', fontsize=12)
    ax3.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Target: 20%')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 4: Latency Distribution (Middle Left)
    # ═══════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 0])
    
    for level, metrics in all_metrics.items():
        successful = [m for m in metrics if m.success]
        latencies = [m.total_latency_ms for m in successful]
        ax4.hist(latencies, bins=30, alpha=0.5, label=f'{level} users', 
                edgecolor='black')
    
    ax4.set_xlabel('Total Latency (ms)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Latency Distribution', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 5: Tokens/Watt Efficiency (Middle Center)
    # ═══════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[1, 1])
    tok_per_watt = [r.tokens_per_watt for r in results]
    
    ax5.plot(concurrent_levels, tok_per_watt, 'o-', linewidth=3, 
             markersize=10, color='#00FF41')
    ax5.fill_between(concurrent_levels, tok_per_watt, alpha=0.3, color='#00FF41')
    
    ax5.set_xlabel('Concurrent Users', fontweight='bold')
    ax5.set_ylabel('Tokens per Watt', fontweight='bold')
    ax5.set_title('Energy Efficiency', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 6: Power Draw Timeline (Middle Right)
    # ═══════════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[1, 2])
    if gpu_history:
        timestamps = [s.timestamp - gpu_history[0].timestamp for s in gpu_history]
        power = [s.power_draw_w for s in gpu_history]
        
        ax6.plot(timestamps, power, linewidth=2, color='#FF6B35')
        ax6.fill_between(timestamps, power, alpha=0.3, color='#FF6B35')
        
        ax6.set_xlabel('Time (seconds)', fontweight='bold')
        ax6.set_ylabel('Power Draw (W)', fontweight='bold')
        ax6.set_title('Power Consumption Over Time', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 7: Request Timeline (Bottom Left)
    # ═══════════════════════════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[2, 0])
    
    for level, metrics in all_metrics.items():
        successful = [m for m in metrics if m.success]
        if not successful:
            continue
        base_time = min(m.start_time for m in successful)
        start_times = [(m.start_time - base_time) for m in successful]
        tps_values = [m.tokens_per_sec for m in successful]
        
        ax7.scatter(start_times, tps_values, alpha=0.6, label=f'{level} users', s=50)
    
    ax7.set_xlabel('Request Start Time (s)', fontweight='bold')
    ax7.set_ylabel('Tokens/sec (per request)', fontweight='bold')
    ax7.set_title('Request Throughput Timeline', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 8: Success Rate (Bottom Center)
    # ═══════════════════════════════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[2, 1])
    success_rates = [(r.successful_requests / r.total_requests * 100) for r in results]
    
    bars = ax8.bar(concurrent_levels, success_rates, color='#00FF41', 
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax8.set_xlabel('Concurrent Users', fontweight='bold')
    ax8.set_ylabel('Success Rate (%)', fontweight='bold')
    ax8.set_title('Request Success Rate', fontweight='bold', fontsize=12)
    ax8.set_ylim(0, 105)
    ax8.axhline(y=99, color='r', linestyle='--', alpha=0.5, label='Target: 99%')
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 9: Summary Table (Bottom Right)
    # ═══════════════════════════════════════════════════════════════════════
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    table_data = [['Metric', '10 Users', '50 Users', '100 Users']]
    
    metrics_to_show = [
        ('TTFT P95 (ms)', [f"{r.p95_ttft_ms:.0f}" for r in results]),
        ('Latency P99 (ms)', [f"{r.p99_latency_ms:.0f}" for r in results]),
        ('Throughput (tok/s)', [f"{r.aggregate_tokens_per_sec:.0f}" for r in results]),
        ('MFU (%)', [f"{r.real_mfu:.1f}" for r in results]),
        ('tok/W', [f"{r.tokens_per_watt:.2f}" for r in results]),
    ]
    
    for metric_name, values in metrics_to_show:
        row = [metric_name] + values
        table_data.append(row)
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax9.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)
    
    # ═══════════════════════════════════════════════════════════════════════
    # OVERALL TITLE
    # ═══════════════════════════════════════════════════════════════════════
    gpu_name = "GPU"
    if gpu_history and hasattr(gpu_history[0], 'gpu_name'):
        gpu_name = gpu_history[0].gpu_name
    
    fig.suptitle(f'LLM Inference Benchmark: Multi-Load Analysis\n'
                f'Model: Llama-8B | Hardware: {gpu_name}',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive analysis saved: comprehensive_benchmark.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   ENHANCED GPU BENCHMARK v3.0: Multi-Load Analysis           ║")
    print("║   Comparing 10, 50, 100 Concurrent Users                     ║")
    print("║   Metrics: TTFT, tok/s, MFU (Real GPU Utilization)          ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    monitor = GPUMonitor(GPU_ID)
    if not monitor.handle:
        print("✗ Cannot proceed without GPU access")
        return
    
    load_gen = LoadGenerator(BACKEND_URL)
    results = []
    all_metrics = {}
    
    # Start monitoring
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    await asyncio.sleep(1)
    
    # Warmup
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  WARMUP PHASE                                            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    await load_gen.run_concurrent_test(5, 4)
    await asyncio.sleep(3)
    
    # Run tests for each concurrency level
    for num_users in CONCURRENT_USERS:
        metrics = await load_gen.run_concurrent_test(num_users, REQUESTS_PER_USER)
        all_metrics[num_users] = metrics
        
        avg_power = monitor.get_avg_power()
        result = analyze_load_test(metrics, avg_power, monitor.peak_flops)
        
        if result:
            results.append(result)
            print_result(result)
        
        await asyncio.sleep(5)  # Cooldown between tests
    
    # Stop monitoring
    monitor.stop()
    await monitor_task
    
    # Generate comprehensive visualization
    if results:
        plot_comprehensive_analysis(results, all_metrics, monitor.history, 
                                   monitor.peak_flops)
        
        # Print comparison summary
        print("\n╔═══════════════════════════════════════════════════════════════╗")
        print("║                  COMPARATIVE ANALYSIS                         ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        
        for i, r in enumerate(results):
            if i == 0:
                print(f"║ BASELINE: {r.concurrent_users} users                                          ║")
                print(f"║   Throughput: {r.aggregate_tokens_per_sec:.1f} tok/s                                  ║")
                print(f"║   MFU: {r.real_mfu:.2f}%                                                ║")
                baseline_tps = r.aggregate_tokens_per_sec
                baseline_mfu = r.real_mfu
            else:
                tps_change = ((r.aggregate_tokens_per_sec - baseline_tps) / baseline_tps) * 100
                mfu_change = ((r.real_mfu - baseline_mfu) / baseline_mfu) * 100
                print(f"╟───────────────────────────────────────────────────────────────╢")
                print(f"║ {r.concurrent_users} users vs {results[0].concurrent_users} users                                       ║")
                print(f"║   Throughput: {r.aggregate_tokens_per_sec:.1f} tok/s ({tps_change:+.1f}%)                       ║")
                print(f"║   MFU: {r.real_mfu:.2f}% ({mfu_change:+.1f}%)                                  ║")
                print(f"║   TTFT P95: {r.p95_ttft_ms:.0f} ms                                        ║")
        
        print("╚═══════════════════════════════════════════════════════════════╝\n")
        
        # Key insights
        best_mfu = max(results, key=lambda x: x.real_mfu)
        best_tps = max(results, key=lambda x: x.aggregate_tokens_per_sec)
        
        print("KEY INSIGHTS:")
        print(f"  • Best MFU: {best_mfu.real_mfu:.2f}% at {best_mfu.concurrent_users} users")
        print(f"  • Best Throughput: {best_tps.aggregate_tokens_per_sec:.1f} tok/s at {best_tps.concurrent_users} users")
        print(f"  • MFU explains TRUE GPU utilization (not kernel occupancy)")
        print(f"  • Higher concurrency → Higher throughput but diminishing MFU gains")
        
        if monitor.peak_flops:
            print(f"\n  ⚠ Remember: {monitor.peak_flops/1e12:.1f} TFLOPS is theoretical peak.")
            print(f"    Inference is memory-bound. 15-25% MFU is excellent.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✗ Benchmark aborted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()