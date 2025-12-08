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
    from scipy.optimize import curve_fit
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetName,
        nvmlDeviceGetClockInfo, nvmlDeviceGetTemperature,
        NVML_CLOCK_GRAPHICS, NVML_TEMPERATURE_GPU
    )
except ImportError as e:
    print(f"FATAL: Missing dependency: {e}")
    print("Run: pip install aiohttp numpy nvidia-ml-py matplotlib scipy")
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
# ═══════════════════════════════════════════════════════════════════════════
# SATURATION ANALYSIS WITH CURVE FITTING
# ═══════════════════════════════════════════════════════════════════════════
def calculate_saturation_point(results: List[LoadTestResult]) -> Dict:
    """
    Calculate throughput saturation point using:
    1. Marginal gains analysis (empirical)
    2. Curve fitting to predict asymptotic limit (mathematical)
    """
    if len(results) < 2:
        return {"saturated": False, "saturation_point": None}
    
    throughputs = [(r.concurrent_users, r.aggregate_tokens_per_sec) for r in results]
    throughputs.sort(key=lambda x: x[0])
    
    users = np.array([t[0] for t in throughputs])
    tps = np.array([t[1] for t in throughputs])
    
    # ═══════════════════════════════════════════════════════════════════════
    # METHOD 1: Marginal Gains (Empirical)
    # ═══════════════════════════════════════════════════════════════════════
    marginal_gains = []
    for i in range(1, len(throughputs)):
        prev_users, prev_tps = throughputs[i-1]
        curr_users, curr_tps = throughputs[i]
        
        user_increase = curr_users - prev_users
        tps_increase = curr_tps - prev_tps
        marginal_gain_pct = (tps_increase / prev_tps) * 100 if prev_tps > 0 else 0
        marginal_gain_per_user = tps_increase / user_increase if user_increase > 0 else 0
        
        marginal_gains.append({
            "from_users": prev_users,
            "to_users": curr_users,
            "tps_gain": tps_increase,
            "gain_pct": marginal_gain_pct,
            "gain_per_user": marginal_gain_per_user
        })
    
    # Find empirical saturation (< 10% gain)
    saturation_threshold = 10.0
    empirical_saturation = None
    
    for i, gain in enumerate(marginal_gains):
        if gain["gain_pct"] < saturation_threshold:
            empirical_saturation = {
                "users": gain["to_users"],
                "throughput": throughputs[i+1][1],
                "marginal_gain_pct": gain["gain_pct"],
                "message": f"Empirical saturation at {gain['to_users']} users"
            }
            break
    
    # ═══════════════════════════════════════════════════════════════════════
    # METHOD 2: Curve Fitting (Mathematical Prediction)
    # ═══════════════════════════════════════════════════════════════════════
    # Fit Michaelis-Menten / Saturation curve: T(u) = T_max * u / (K + u)
    # Where: T_max = asymptotic throughput, K = half-saturation constant
    
    def saturation_curve(u, T_max, K):
        """Michaelis-Menten saturation model"""
        return T_max * u / (K + u)
    
    try:
        from scipy.optimize import curve_fit
        
        # Initial guess: T_max = 2x current max, K = median users
        initial_guess = [tps[-1] * 2, np.median(users)]
        
        # Fit curve
        params, covariance = curve_fit(saturation_curve, users, tps, 
                                      p0=initial_guess, maxfev=10000)
        T_max_predicted, K = params
        
        # Calculate confidence
        residuals = tps - saturation_curve(users, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tps - np.mean(tps))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Predict saturation point: where we reach 95% of T_max
        # T(u) = 0.95 * T_max → u = 19 * K (from algebra)
        predicted_saturation_users = int(19 * K)
        predicted_saturation_tps = saturation_curve(predicted_saturation_users, *params)
        
        # Calculate current % of theoretical max
        current_max_tps = tps[-1]
        current_pct_of_max = (current_max_tps / T_max_predicted) * 100
        
        curve_fit_result = {
            "T_max": T_max_predicted,
            "K": K,
            "r_squared": r_squared,
            "predicted_saturation_users": predicted_saturation_users,
            "predicted_saturation_tps": predicted_saturation_tps,
            "current_pct_of_max": current_pct_of_max,
            "fit_quality": "Excellent" if r_squared > 0.95 else "Good" if r_squared > 0.85 else "Fair"
        }
        
        # Generate prediction curve for plotting
        u_pred = np.linspace(users[0], predicted_saturation_users * 1.5, 100)
        tps_pred = saturation_curve(u_pred, *params)
        curve_fit_result["prediction_curve"] = (u_pred, tps_pred)
        
    except Exception as e:
        print(f"  ⚠ Curve fitting failed: {e}")
        curve_fit_result = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # METHOD 3: Diminishing Returns Analysis
    # ═══════════════════════════════════════════════════════════════════════
    # Calculate marginal throughput per additional user
    marginal_tps_per_user = np.diff(tps) / np.diff(users)
    
    # Find where marginal gain drops below threshold (e.g., 10 tok/s per user)
    marginal_threshold = 10.0  # tok/s per additional user
    diminishing_returns_point = None
    
    for i, marginal in enumerate(marginal_tps_per_user):
        if marginal < marginal_threshold:
            diminishing_returns_point = {
                "users": int(users[i+1]),
                "marginal_tps_per_user": marginal,
                "message": f"Diminishing returns at {int(users[i+1])} users"
            }
            break
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════════════
    efficiency_scores = [(r.concurrent_users, r.aggregate_tokens_per_sec / r.concurrent_users) 
                         for r in results]
    best_efficiency = max(efficiency_scores, key=lambda x: x[1])
    
    # Determine if saturated
    saturated = empirical_saturation is not None
    
    # If curve fit succeeded and predicts saturation beyond our test range
    if curve_fit_result and curve_fit_result['current_pct_of_max'] < 90:
        recommendation = f"Test up to {curve_fit_result['predicted_saturation_users']} users to reach theoretical max"
    elif curve_fit_result and curve_fit_result['current_pct_of_max'] >= 90:
        recommendation = f"Near saturation ({curve_fit_result['current_pct_of_max']:.1f}% of predicted max)"
        saturated = True
    else:
        recommendation = "Increase test range to find saturation point"
    
    return {
        "saturated": saturated,
        "saturation_point": empirical_saturation,
        "marginal_gains": marginal_gains,
        "best_efficiency_users": best_efficiency[0],
        "best_efficiency_tps_per_user": best_efficiency[1],
        "throughput_curve": throughputs,
        "curve_fit": curve_fit_result,
        "diminishing_returns": diminishing_returns_point,
        "recommendation": recommendation
    }

def plot_gpu_telemetry_timeline(gpu_history: List[GPUState], 
                                all_metrics: Dict[int, List[RequestMetric]],
                                gpu_name: str):
    """Generate detailed GPU telemetry timeline (separate plot)"""
    
    if not gpu_history:
        print("⚠ No GPU telemetry data to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    base_time = gpu_history[0].timestamp
    timestamps = [(s.timestamp - base_time) for s in gpu_history]
    
    # Extract telemetry data
    kernel_util = [s.utilization_kernel for s in gpu_history]
    power_draw = [s.power_draw_w for s in gpu_history]
    temperature = [s.temperature_c for s in gpu_history]
    
    # Find load transition points from request metrics
    load_transitions = []
    for level in sorted(all_metrics.keys()):
        metrics = all_metrics[level]
        if metrics:
            start = min(m.start_time for m in metrics) - base_time
            end = max(m.end_time for m in metrics) - base_time
            load_transitions.append((start, end, level))
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 1: Kernel Occupancy (NVML GPU Utilization %)
    # ═══════════════════════════════════════════════════════════════════════
    axes[0].plot(timestamps, kernel_util, linewidth=2, color='#00FF41', label='Kernel Occupancy')
    axes[0].fill_between(timestamps, kernel_util, alpha=0.3, color='#00FF41')
    
    # Shade load regions
    for start, end, level in load_transitions:
        axes[0].axvspan(start, end, alpha=0.15, color='#FFA500')
        mid = (start + end) / 2
        axes[0].text(mid, 95, f'{level} users', ha='center', fontsize=10, 
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[0].set_ylabel('Kernel Occupancy (%)', fontweight='bold', fontsize=11)
    axes[0].set_title('GPU Kernel Utilization (NVML) - NOT Real Compute Utilization', 
                     fontweight='bold', fontsize=12)
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    
    # Add warning annotation
    axes[0].text(0.98, 0.5, 
                '⚠ This shows "GPU busy"\nNOT "compute utilization"\nSee MFU for real usage',
                transform=axes[0].transAxes, fontsize=9, 
                verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 2: Power Draw & Temperature (Dual Y-axis)
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = axes[1]
    ax2_temp = ax2.twinx()
    
    # Power on left axis
    line1 = ax2.plot(timestamps, power_draw, linewidth=2, color='#FF6B35', label='Power Draw')
    ax2.fill_between(timestamps, power_draw, alpha=0.3, color='#FF6B35')
    
    for start, end, level in load_transitions:
        ax2.axvspan(start, end, alpha=0.15, color='#FFA500')
    
    avg_power = statistics.mean(power_draw)
    line2 = ax2.axhline(y=avg_power, color='red', linestyle='--', linewidth=1.5, 
                       label=f'Avg Power: {avg_power:.1f}W')
    
    ax2.set_xlabel('Time (seconds)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Power Draw (W)', fontweight='bold', fontsize=11, color='#FF6B35')
    ax2.tick_params(axis='y', labelcolor='#FF6B35')
    ax2.grid(True, alpha=0.3)
    
    # Temperature on right axis
    line3 = ax2_temp.plot(timestamps, temperature, linewidth=2, color='#FF1493', 
                          label='GPU Temp', alpha=0.7)
    
    max_temp = max(temperature)
    line4 = ax2_temp.axhline(y=83, color='orange', linestyle=':', linewidth=2, 
                            label='Throttle: 83°C', alpha=0.7)
    
    ax2_temp.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=11, color='#FF1493')
    ax2_temp.tick_params(axis='y', labelcolor='#FF1493')
    
    # Combined legend
    lines = line1 + [line2] + line3 + [line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    ax2.set_title('Power Consumption & Thermal State', fontweight='bold', fontsize=12)
    
    # Overall title
    fig.suptitle(f'GPU Telemetry Timeline: {gpu_name}\n'
                f'Kernel Occupancy vs. Power vs. Temperature',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('gpu_telemetry_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ GPU telemetry timeline saved: gpu_telemetry_timeline.png")
    plt.close()

def plot_comprehensive_analysis(results: List['LoadTestResult'], 
                                all_metrics: Dict[int, List['RequestMetric']],
                                gpu_history: List['GPUState'],
                                peak_flops: Optional[float],
                                saturation_analysis: Dict):
    """Generate multi-panel analysis dashboard (Fixed Version)"""
    
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
    
    ax1.plot(concurrent_levels, ttft_avg, 'o-', linewidth=3, markersize=8, 
             label='Avg TTFT', color='#00FF41')
    ax1.plot(concurrent_levels, ttft_p95, 's-', linewidth=2, markersize=7,
             label='P95 TTFT', color='#FFA500')
    ax1.plot(concurrent_levels, ttft_p99, '^-', linewidth=2, markersize=7,
             label='P99 TTFT', color='#FF6B35')
    
    ax1.set_xlabel('Concurrent Users', fontweight='bold')
    ax1.set_ylabel('Time to First Token (ms)', fontweight='bold')
    ax1.set_title('TTFT Latency by Concurrency', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 2: Throughput Scaling with Saturation Point (Top Center)
    # ═══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    aggregate_tps = [r.aggregate_tokens_per_sec for r in results]
    per_request_tps = [r.avg_request_tokens_per_sec for r in results]
    
    ax2.plot(concurrent_levels, aggregate_tps, 'o-', linewidth=3, 
             markersize=10, label='Aggregate tok/s', color='#00FF41')
    ax2.plot(concurrent_levels, per_request_tps, 's-', linewidth=2, 
             markersize=8, label='Per-Request tok/s', color='#4A90E2')
    
    # Plot prediction curve if available
    if saturation_analysis.get('curve_fit'):
        cf = saturation_analysis['curve_fit']
        u_pred, tps_pred = cf['prediction_curve']
        ax2.plot(u_pred, tps_pred, '--', linewidth=2, color='red', 
                alpha=0.7, label=f"Predicted curve (R²={cf['r_squared']:.3f})")
        
        # Mark theoretical max
        ax2.axhline(y=cf['T_max'], color='orange', linestyle=':', linewidth=2,
                   label=f"Theoretical max: {cf['T_max']:.0f} tok/s", alpha=0.7)
        
        # Mark predicted saturation
        ax2.axvline(x=cf['predicted_saturation_users'], color='red', linestyle='--', 
                   linewidth=2, label=f"95% max at ~{cf['predicted_saturation_users']} users", 
                   alpha=0.7)
    
    # Mark empirical saturation if found
    if saturation_analysis.get('saturated') and saturation_analysis.get('saturation_point'):
        sat_point = saturation_analysis['saturation_point']
        # Only plot if the saturation point exists in our current x-axis
        if sat_point['users'] in concurrent_levels:
            sat_idx = concurrent_levels.index(sat_point['users'])
            ax2.plot(sat_point['users'], aggregate_tps[sat_idx], 'r*', 
                    markersize=20, label=f"Empirical sat: {sat_point['users']} users")
    
    ax2.set_xlabel('Concurrent Users', fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax2.set_title('Throughput Scaling', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 3: MFU Comparison (Top Right)
    # ═══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[0, 2])
    mfu_values = [r.real_mfu for r in results]
    colors = ['#00FF41' if m > 15 else '#FFA500' if m > 10 else '#FF6B35' 
              for m in mfu_values]
    
    bars = ax3.bar(concurrent_levels, mfu_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2, width=max(concurrent_levels)*0.1) # Added width control
    
    # Add labels on bars
    for bar, mfu in zip(bars, mfu_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mfu:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Concurrent Users', fontweight='bold')
    ax3.set_ylabel('Real GPU Utilization (MFU %)', fontweight='bold')
    ax3.set_title('Model FLOPS Utilization', fontweight='bold', fontsize=12)
    ax3.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Target: 20%')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 4: Marginal Gains Analysis (Middle Left)
    # ═══════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[1, 0])
    
    if saturation_analysis.get('marginal_gains'):
        gains_x = [g['to_users'] for g in saturation_analysis['marginal_gains']]
        gains_y = [g['gain_pct'] for g in saturation_analysis['marginal_gains']]
        
        ax4.plot(gains_x, gains_y, 'o-', linewidth=3, markersize=10, color='#00FF41')
        ax4.fill_between(gains_x, gains_y, alpha=0.3, color='#00FF41')
        ax4.axhline(y=10, color='red', linestyle='--', linewidth=2, 
                   label='Threshold (10%)', alpha=0.7)
        
        for x, y in zip(gains_x, gains_y):
            ax4.text(x, y + 2, f'{y:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Concurrent Users', fontweight='bold')
    ax4.set_ylabel('Marginal Throughput Gain (%)', fontweight='bold')
    ax4.set_title('Marginal Gains Analysis', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
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
    # PANEL 6: Efficiency Score (Throughput per User) (Middle Right)
    # ═══════════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[1, 2])
    efficiency = [r.aggregate_tokens_per_sec / r.concurrent_users for r in results]
    
    ax6.plot(concurrent_levels, efficiency, 'o-', linewidth=3, 
             markersize=10, color='#4A90E2')
    ax6.fill_between(concurrent_levels, efficiency, alpha=0.3, color='#4A90E2')
    
    best_eff_idx = efficiency.index(max(efficiency))
    ax6.axvline(x=concurrent_levels[best_eff_idx], color='green', 
               linestyle='--', linewidth=2, 
               label=f'Optimal: {concurrent_levels[best_eff_idx]} users', alpha=0.7)
    
    ax6.set_xlabel('Concurrent Users', fontweight='bold')
    ax6.set_ylabel('Throughput per User (tok/s)', fontweight='bold')
    ax6.set_title('Efficiency Score', fontweight='bold', fontsize=12)
    ax6.legend()
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
        
        ax7.plot(start_times, tps_values, '-', alpha=0.7, linewidth=2, label=f'{level} users')
    
    ax7.set_xlabel('Test Duration (s)', fontweight='bold')
    ax7.set_ylabel('Tokens/sec (per request)', fontweight='bold')
    ax7.set_title('Throughput Stability', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PANEL 8: Success Rate (Bottom Center)
    # ═══════════════════════════════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[2, 1])
    success_rates = [(r.successful_requests / r.total_requests * 100) for r in results]
    
    bars = ax8.bar(concurrent_levels, success_rates, color='#00FF41', 
                   alpha=0.8, edgecolor='black', linewidth=2, width=max(concurrent_levels)*0.1)
    
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
    
    # Dynamically create header based on actual users run
    header_row = ['Metric'] + [f'{u} Users' for u in concurrent_levels]
    table_data = [header_row]
    
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
    
    # Dynamically calculate column widths
    col_widths = [0.3] + [0.7 / len(concurrent_levels)] * len(concurrent_levels)
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(header_row)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(header_row)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax9.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)
    
    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SAVE
    # ═══════════════════════════════════════════════════════════════════════
    gpu_name = "GPU"
    if gpu_history and hasattr(gpu_history[0], 'gpu_name'):
        gpu_name = gpu_history[0].gpu_name
    elif gpu_history:
         gpu_name = "NVIDIA A40" # Fallback if attribute missing
    
    fig.suptitle(f'LLM Inference Benchmark: Multi-Load Analysis\n'
                f'Model: Llama-8B | Hardware: {gpu_name}',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive analysis saved: comprehensive_benchmark.png")
    plt.show() # Safe to call show after save
    plt.close()

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
        # Calculate saturation analysis
        saturation_analysis = calculate_saturation_point(results)
        
        # First: Generate GPU telemetry timeline
        plot_gpu_telemetry_timeline(monitor.history, all_metrics, monitor.gpu_name)
        
        # Second: Generate multi-panel analysis dashboard
        plot_comprehensive_analysis(results, all_metrics, monitor.history, 
                                   monitor.peak_flops, saturation_analysis)
        
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
        
        # Saturation analysis summary
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║              SATURATION ANALYSIS                              ║")
        print("╠═══════════════════════════════════════════════════════════════╣")
        
        # Show curve fit prediction first
        if saturation_analysis.get('curve_fit'):
            cf = saturation_analysis['curve_fit']
            print(f"║ 📊 MATHEMATICAL PREDICTION (Michaelis-Menten Curve Fit)      ║")
            print(f"║   Theoretical Max: {cf['T_max']:.1f} tok/s                               ║")
            print(f"║   Half-Saturation (K): {cf['K']:.1f} users                              ║")
            print(f"║   Fit Quality: {cf['fit_quality']} (R² = {cf['r_squared']:.3f})                         ║")
            print(f"║   Current Status: {cf['current_pct_of_max']:.1f}% of theoretical max                ║")
            print(f"║                                                               ║")
            print(f"║   Predicted 95% Saturation: {cf['predicted_saturation_users']} users                     ║")
            print(f"║   Expected Throughput: {cf['predicted_saturation_tps']:.1f} tok/s                       ║")
            print(f"║                                                               ║")
            
            if cf['current_pct_of_max'] < 80:
                print(f"║ 💡 RECOMMENDATION: System has significant headroom.           ║")
                print(f"║   Test up to {cf['predicted_saturation_users']} users to approach theoretical limit. ║")
            elif cf['current_pct_of_max'] < 95:
                print(f"║ 💡 RECOMMENDATION: Approaching saturation.                    ║")
                print(f"║   Test {cf['predicted_saturation_users']} users to confirm model prediction.         ║")
            else:
                print(f"║ ✓ NEAR THEORETICAL MAX: {cf['current_pct_of_max']:.1f}% of predicted capacity.      ║")
                print(f"║   Further scaling will yield minimal gains.                  ║")
        
        print(f"╟───────────────────────────────────────────────────────────────╢")
        
        if saturation_analysis['saturated'] and saturation_analysis['saturation_point']:
            sat = saturation_analysis['saturation_point']
            print(f"║ ⚠ EMPIRICAL SATURATION DETECTED                               ║")
            print(f"║   Point: {sat['users']} concurrent users                                  ║")
            print(f"║   Throughput: {sat['throughput']:.1f} tok/s                              ║")
            print(f"║   Marginal Gain: {sat['marginal_gain_pct']:.1f}% (< 10% threshold)                 ║")
            print(f"║                                                               ║")
            print(f"║ 💡 OPERATIONAL RECOMMENDATION:                                ║")
            print(f"║   Optimal concurrency: {saturation_analysis['best_efficiency_users']} users                              ║")
            print(f"║   (Best throughput per user: {saturation_analysis['best_efficiency_tps_per_user']:.1f} tok/s/user)        ║")
        else:
            print(f"║ ✓ NO EMPIRICAL SATURATION (< 10% gain threshold)             ║")
            print(f"║   All test points show healthy scaling.                      ║")
            print(f"║                                                               ║")
            best_eff = saturation_analysis['best_efficiency_users']
            print(f"║ 💡 CURRENT BEST EFFICIENCY: {best_eff} users                             ║")
            print(f"║   (Highest throughput per user: {saturation_analysis['best_efficiency_tps_per_user']:.1f} tok/s/user)     ║")
        
        print("╟───────────────────────────────────────────────────────────────╢")
        print("║ MARGINAL GAINS BREAKDOWN:                                     ║")
        for gain in saturation_analysis['marginal_gains']:
            status = " ⚠" if gain['gain_pct'] < 10 else ""
            print(f"║   {gain['from_users']:3d} → {gain['to_users']:3d} users: +{gain['gain_pct']:5.1f}% (+{gain['tps_gain']:5.0f} tok/s){status}         ║")
        
        if saturation_analysis.get('diminishing_returns'):
            dr = saturation_analysis['diminishing_returns']
            print(f"╟───────────────────────────────────────────────────────────────╢")
            print(f"║ ⚠ DIMINISHING RETURNS DETECTED:                               ║")
            print(f"║   At {dr['users']} users: {dr['marginal_tps_per_user']:.1f} tok/s per additional user          ║")
            print(f"║   (Below {10.0} tok/s threshold)                                ║")
        
        print("╚═══════════════════════════════════════════════════════════════╝\n")
        
        # Key insights
        best_mfu = max(results, key=lambda x: x.real_mfu)
        best_tps = max(results, key=lambda x: x.aggregate_tokens_per_sec)
        
        print("KEY INSIGHTS:")
        print(f"  • Best MFU: {best_mfu.real_mfu:.2f}% at {best_mfu.concurrent_users} users")
        print(f"  • Best Throughput: {best_tps.aggregate_tokens_per_sec:.1f} tok/s at {best_tps.concurrent_users} users")
        print(f"  • MFU explains TRUE GPU utilization (not kernel occupancy)")
        
        if saturation_analysis['saturated']:
            sat = saturation_analysis['saturation_point']
            print(f"  • ⚠ Throughput saturates at {sat['users']} users - batching/memory-bound")
        else:
            print(f"  • System can likely handle more load - consider testing 150+ users")
        
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