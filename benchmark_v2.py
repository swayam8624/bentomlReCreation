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
    from matplotlib.patches import Rectangle
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetName,
        nvmlDeviceGetClockInfo, nvmlDeviceGetTemperature,
        NVML_CLOCK_GRAPHICS, NVML_TEMPERATURE_GPU
    )
except ImportError as e:
    print(f"FATAL: Missing dependency: {e}")
    print("Run: pip install aiohttp numpy pynvml matplotlib")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
GPU_ID = 0
POLLING_INTERVAL_SEC = 0.05  # 20Hz sampling for smooth graphs
REQUEST_TIMEOUT = 60
MODEL_PARAMS = 8e9  # Llama-8B parameter count
BACKEND_URL = "http://127.0.0.1:8888"

# Phase Durations
WARMUP_DURATION = 20      # Poisson warmup
COOLDOWN_DURATION = 10    # Idle period
STRESS_DURATION = 30      # Max parallel flood

# Load Parameters
WARMUP_RPS = 8
STRESS_MAX_PARALLEL = 50  # Max concurrent requests

# ═══════════════════════════════════════════════════════════════════════════
# GPU DATABASE: Peak FLOPS for Common RunPod/Cloud GPUs
# ═══════════════════════════════════════════════════════════════════════════
GPU_SPECS = {
    # NVIDIA Data Center
    "NVIDIA A100-SXM4-80GB": 312e12,      # 312 TFLOPS FP16 Tensor
    "NVIDIA A100-PCIE-80GB": 312e12,
    "NVIDIA A100-SXM4-40GB": 312e12,
    "NVIDIA H100-SXM5-80GB": 989e12,      # 989 TFLOPS FP16 Tensor
    "NVIDIA H100-PCIE-80GB": 756e12,
    "Tesla V100-SXM2-32GB": 125e12,
    "Tesla V100-PCIE-32GB": 112e12,
    
    # Consumer/Prosumer
    "NVIDIA GeForce RTX 4090": 165e12,    # 165 TFLOPS FP16
    "NVIDIA GeForce RTX 4080": 96e12,
    "NVIDIA GeForce RTX 3090": 71e12,
    "NVIDIA RTX A6000": 77e12,
    "NVIDIA RTX A5000": 54e12,
    
    # AMD (Approximate FP16)
    "AMD Instinct MI250": 362e12,
    "AMD Instinct MI210": 181e12,
}

@dataclass
class GPUState:
    timestamp: float
    phase: str
    utilization_gpu: float
    memory_used_mb: float
    power_draw_w: float
    clock_sm_mhz: float
    temperature_c: float

@dataclass
class RequestMetric:
    request_id: str
    phase: str
    start_time: float
    end_time: float
    latency_ms: float
    output_tokens: int
    success: bool
    error: str = ""

@dataclass
class PhaseResult:
    phase_name: str
    duration: float
    total_requests: int
    successful_requests: int
    avg_tokens_per_sec: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_power_w: float
    avg_utilization: float
    peak_memory_mb: float

@dataclass
class BenchmarkReport:
    gpu_name: str
    peak_flops: float
    model_params: float
    phases: List[PhaseResult]
    overall_mfu: float
    stress_mfu: float
    tokens_per_watt: float

# ═══════════════════════════════════════════════════════════════════════════
# GPU MONITOR: Daemon Thread for Continuous Telemetry
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
            self.current_phase = "idle"
            print(f"✓ GPU Detected: {self.gpu_name}")
            if self.peak_flops:
                print(f"✓ Peak FP16 Tensor FLOPS: {self.peak_flops/1e12:.1f} TFLOPS")
            else:
                print(f"⚠ GPU not in database - MFU will be raw FLOPS only")
        except Exception as e:
            print(f"✗ Failed to initialize NVML: {e}")
            self.handle = None
            self.gpu_name = "Unknown"
            self.peak_flops = None

    def set_phase(self, phase: str):
        self.current_phase = phase

    async def start_monitoring(self):
        if not self.handle:
            return
        self.running = True
        self.history = []
        print("   >>> GPU Telemetry Daemon Started (20Hz Sampling)")
        
        while self.running:
            try:
                util = nvmlDeviceGetUtilizationRates(self.handle)
                mem = nvmlDeviceGetMemoryInfo(self.handle)
                power = nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_GRAPHICS)
                temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
                
                self.history.append(GPUState(
                    timestamp=time.time(),
                    phase=self.current_phase,
                    utilization_gpu=util.gpu,
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

    def get_phase_stats(self, phase: str) -> Dict:
        phase_data = [x for x in self.history if x.phase == phase]
        if not phase_data:
            return {}
        return {
            "avg_power_w": statistics.mean([x.power_draw_w for x in phase_data]),
            "avg_utilization": statistics.mean([x.utilization_gpu for x in phase_data]),
            "peak_memory_mb": max([x.memory_used_mb for x in phase_data]),
            "avg_clock_mhz": statistics.mean([x.clock_sm_mhz for x in phase_data]),
            "max_temp_c": max([x.temperature_c for x in phase_data]),
        }

# ═══════════════════════════════════════════════════════════════════════════
# LOAD GENERATORS
# ═══════════════════════════════════════════════════════════════════════════
class LoadGenerator:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url.rstrip("/")
        self.prompts = [
            "Explain quantum entanglement in simple terms.",
            "Write a Python function to calculate Fibonacci numbers.",
            "What are the three laws of thermodynamics?",
            "Summarize the Roman Empire's rise and fall in 50 words.",
            "Debug this code: for i in range(10) print(i)",
            "Describe how neural networks learn from data.",
            "What is the difference between supervised and unsupervised learning?",
            "Explain the concept of recursion with an example.",
        ] * 10

    async def _send_request(self, session: aiohttp.ClientSession, req_id: str, phase: str) -> RequestMetric:
        prompt = random.choice(self.prompts)
        start = time.time()
        payload = {
            "model": "/workspace/models/llama-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.7
        }
        url = f"{self.backend_url}/v1/chat/completions"
        
        try:
            async with session.post(url, json=payload, timeout=REQUEST_TIMEOUT) as resp:
                text = await resp.text()
                end = time.time()
                
                if resp.status != 200:
                    return RequestMetric(str(req_id), phase, start, end, (end-start)*1000, 0, False, 
                                       error=f"HTTP {resp.status}")
                
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
                
                return RequestMetric(
                    request_id=str(req_id),
                    phase=phase,
                    start_time=start,
                    end_time=end,
                    latency_ms=(end-start)*1000,
                    output_tokens=tokens,
                    success=True
                )
        except Exception as e:
            return RequestMetric(str(req_id), phase, start, time.time(), 
                               (time.time()-start)*1000, 0, False, error=str(e))

    async def run_poisson_warmup(self, duration_sec: int, target_rps: float) -> List[RequestMetric]:
        """Phase 1: Poisson arrival for gradual warmup"""
        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║  PHASE 1: WARMUP (Poisson Arrival @ {target_rps} RPS)       ║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        
        metrics = []
        start_time = time.time()
        req_count = 0
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            while time.time() - start_time < duration_sec:
                tasks.append(asyncio.create_task(
                    self._send_request(session, f"warmup-{req_count}", "warmup")
                ))
                req_count += 1
                sleep_time = random.expovariate(target_rps)
                await asyncio.sleep(sleep_time)
            
            print(f"   >>> Draining {len(tasks)} warmup requests...")
            results = await asyncio.gather(*tasks)
            metrics = list(results)
        
        success_count = sum(1 for m in metrics if m.success)
        print(f"   ✓ Warmup Complete: {success_count}/{len(metrics)} successful")
        return metrics

    async def run_stress_flood(self, duration_sec: int, max_parallel: int) -> List[RequestMetric]:
        """Phase 3: Max parallel stress test with semaphore"""
        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║  PHASE 3: STRESS TEST (Max {max_parallel} Parallel)         ║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        
        metrics = []
        start_time = time.time()
        req_count = 0
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def limited_request(session, req_id):
            async with semaphore:
                return await self._send_request(session, req_id, "stress")
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            # Flood the queue as fast as possible
            while time.time() - start_time < duration_sec:
                tasks.append(asyncio.create_task(
                    limited_request(session, f"stress-{req_count}")
                ))
                req_count += 1
                await asyncio.sleep(0.01)  # Minimal delay to prevent blocking
            
            print(f"   >>> Processing {len(tasks)} stress requests...")
            results = await asyncio.gather(*tasks)
            metrics = list(results)
        
        success_count = sum(1 for m in metrics if m.success)
        print(f"   ✓ Stress Test Complete: {success_count}/{len(metrics)} successful")
        return metrics

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def analyze_phase(phase_name: str, metrics: List[RequestMetric], 
                  gpu_stats: Dict, duration: float) -> Optional[PhaseResult]:
    successful = [m for m in metrics if m.success]
    if not successful:
        print(f"   ⚠ No successful requests in {phase_name}")
        return None
    
    latencies = [m.latency_ms for m in successful]
    total_tokens = sum([m.output_tokens for m in successful])
    
    return PhaseResult(
        phase_name=phase_name,
        duration=duration,
        total_requests=len(metrics),
        successful_requests=len(successful),
        avg_tokens_per_sec=total_tokens / duration,
        avg_latency_ms=statistics.mean(latencies),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        avg_power_w=gpu_stats.get('avg_power_w', 0),
        avg_utilization=gpu_stats.get('avg_utilization', 0),
        peak_memory_mb=gpu_stats.get('peak_memory_mb', 0)
    )

def calculate_mfu(tokens_per_sec: float, model_params: float, peak_flops: Optional[float]) -> float:
    """Calculate Model FLOPS Utilization"""
    if not peak_flops:
        return 0.0
    # FLOPs per token ≈ 2 * params (forward pass)
    achieved_flops = tokens_per_sec * model_params * 2
    mfu = (achieved_flops / peak_flops) * 100
    return mfu

def generate_report(monitor: GPUMonitor, phase_results: List[PhaseResult]) -> BenchmarkReport:
    """Compile comprehensive benchmark report"""
    stress_phase = next((p for p in phase_results if p.phase_name == "stress"), None)
    
    total_tokens = sum([p.avg_tokens_per_sec * p.duration for p in phase_results])
    total_duration = sum([p.duration for p in phase_results])
    overall_throughput = total_tokens / total_duration
    
    overall_mfu = calculate_mfu(overall_throughput, MODEL_PARAMS, monitor.peak_flops)
    stress_mfu = calculate_mfu(stress_phase.avg_tokens_per_sec, MODEL_PARAMS, 
                               monitor.peak_flops) if stress_phase else 0
    
    avg_power = statistics.mean([p.avg_power_w for p in phase_results if p.avg_power_w > 0])
    tokens_per_watt = overall_throughput / avg_power if avg_power > 0 else 0
    
    return BenchmarkReport(
        gpu_name=monitor.gpu_name,
        peak_flops=monitor.peak_flops or 0,
        model_params=MODEL_PARAMS,
        phases=phase_results,
        overall_mfu=overall_mfu,
        stress_mfu=stress_mfu,
        tokens_per_watt=tokens_per_watt
    )

def print_report(report: BenchmarkReport):
    """Print comprehensive benchmark report"""
    print(f"\n╔═══════════════════════════════════════════════════════════════╗")
    print(f"║           PRODUCTION BENCHMARK REPORT                        ║")
    print(f"╠═══════════════════════════════════════════════════════════════╣")
    print(f"║ GPU: {report.gpu_name:<53} ║")
    print(f"║ Model: Llama-8B ({report.model_params/1e9:.1f}B params)                              ║")
    if report.peak_flops:
        print(f"║ Theoretical Peak: {report.peak_flops/1e12:.1f} TFLOPS (FP16 Tensor)             ║")
    print(f"╠═══════════════════════════════════════════════════════════════╣")
    
    for phase in report.phases:
        print(f"║ [{phase.phase_name.upper():^10}] Duration: {phase.duration:.1f}s                           ║")
        print(f"║   Throughput:    {phase.avg_tokens_per_sec:8.1f} tok/s                           ║")
        print(f"║   Latency (Avg): {phase.avg_latency_ms:8.1f} ms  | P99: {phase.p99_latency_ms:8.1f} ms          ║")
        print(f"║   GPU Util:      {phase.avg_utilization:6.1f}%     | Power: {phase.avg_power_w:6.1f}W          ║")
        print(f"║   Requests:      {phase.successful_requests}/{phase.total_requests} successful                              ║")
        print(f"╟───────────────────────────────────────────────────────────────╢")
    
    print(f"║ EFFICIENCY METRICS                                            ║")
    print(f"╠═══════════════════════════════════════════════════════════════╣")
    if report.peak_flops:
        print(f"║ Overall MFU:     {report.overall_mfu:6.2f}%                                   ║")
        print(f"║ Stress MFU:      {report.stress_mfu:6.2f}%  (Peak Load)                     ║")
    print(f"║ Efficiency:      {report.tokens_per_watt:8.2f} tok/W                           ║")
    print(f"╚═══════════════════════════════════════════════════════════════╝\n")

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def plot_telemetry(monitor: GPUMonitor, phase_results: List[PhaseResult]):
    """Generate annotated GPU telemetry graph"""
    if not monitor.history:
        print("⚠ No telemetry data to plot")
        return
    
    timestamps = [s.timestamp - monitor.history[0].timestamp for s in monitor.history]
    utilizations = [s.utilization_gpu for s in monitor.history]
    powers = [s.power_draw_w for s in monitor.history]
    phases = [s.phase for s in monitor.history]
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # GPU Utilization
    ax1.plot(timestamps, utilizations, color='#00FF41', linewidth=2, label='GPU Utilization')
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GPU Utilization (%)', color='#00FF41', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#00FF41')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Power Draw (secondary axis)
    ax2 = ax1.twinx()
    ax2.plot(timestamps, powers, color='#FF6B35', linewidth=1.5, alpha=0.7, label='Power Draw')
    ax2.set_ylabel('Power Draw (W)', color='#FF6B35', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#FF6B35')
    
    # Phase annotations
    phase_colors = {
        'idle': '#555555',
        'warmup': '#FFA500',
        'cooldown': '#4A90E2',
        'stress': '#FF0000'
    }
    
    current_phase = phases[0]
    phase_start = 0
    
    for i, phase in enumerate(phases):
        if phase != current_phase or i == len(phases) - 1:
            ax1.axvspan(phase_start, timestamps[i], alpha=0.15, 
                       color=phase_colors.get(current_phase, '#CCCCCC'),
                       label=f'{current_phase.capitalize()} Phase')
            
            # Add phase label
            mid_point = (phase_start + timestamps[i]) / 2
            ax1.text(mid_point, 98, current_phase.upper(), 
                    ha='center', va='top', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=phase_colors.get(current_phase, '#CCCCCC'), 
                            alpha=0.7))
            
            current_phase = phase
            phase_start = timestamps[i]
    
    # Title and legend
    plt.title(f'GPU Telemetry: {monitor.gpu_name}\nPhase-Based Load Testing with MFU Analysis', 
              fontsize=14, fontweight='bold', pad=20)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('gpu_stress_telemetry.png', dpi=300, bbox_inches='tight')
    print("✓ Telemetry graph saved: gpu_stress_telemetry.png")
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║      GPU STRESS TEST & MFU BENCHMARK SUITE v2.0              ║")
    print("║      Phase-Based Load Testing with Real-Time Telemetry       ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    monitor = GPUMonitor(GPU_ID)
    if not monitor.handle:
        print("✗ Cannot proceed without GPU access")
        return
    
    load_gen = LoadGenerator(BACKEND_URL)
    phase_results = []
    
    # Start GPU monitoring daemon
    monitor.set_phase("idle")
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    await asyncio.sleep(2)  # Brief idle baseline
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: WARMUP
    # ═══════════════════════════════════════════════════════════════════════
    monitor.set_phase("warmup")
    metrics_warmup = await load_gen.run_poisson_warmup(WARMUP_DURATION, WARMUP_RPS)
    warmup_stats = monitor.get_phase_stats("warmup")
    warmup_result = analyze_phase("warmup", metrics_warmup, warmup_stats, WARMUP_DURATION)
    if warmup_result:
        phase_results.append(warmup_result)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: COOLDOWN
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  PHASE 2: COOLDOWN (Idle State Verification)            ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    monitor.set_phase("cooldown")
    await asyncio.sleep(COOLDOWN_DURATION)
    print(f"   ✓ Cooldown Complete: {COOLDOWN_DURATION}s idle period")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: STRESS TEST
    # ═══════════════════════════════════════════════════════════════════════
    monitor.set_phase("stress")
    metrics_stress = await load_gen.run_stress_flood(STRESS_DURATION, STRESS_MAX_PARALLEL)
    stress_stats = monitor.get_phase_stats("stress")
    
    # Calculate actual stress duration from metrics
    if metrics_stress:
        stress_actual_duration = max(m.end_time for m in metrics_stress) - min(m.start_time for m in metrics_stress)
    else:
        stress_actual_duration = STRESS_DURATION
    
    stress_result = analyze_phase("stress", metrics_stress, stress_stats, stress_actual_duration)
    if stress_result:
        phase_results.append(stress_result)
    
    # Stop monitoring
    monitor.stop()
    await monitor_task
    
    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS & REPORTING
    # ═══════════════════════════════════════════════════════════════════════
    if phase_results:
        report = generate_report(monitor, phase_results)
        print_report(report)
        plot_telemetry(monitor, phase_results)
    else:
        print("✗ Insufficient data for analysis")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✗ Benchmark aborted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()