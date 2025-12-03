import asyncio
import time
import json
import statistics
import random
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict

# Dependency Check
try:
    import aiohttp
    import numpy as np
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage,
        nvmlDeviceGetClockInfo, nvmlDeviceGetTemperature,
        NVML_CLOCK_GRAPHICS, NVML_TEMPERATURE_GPU
    )
except ImportError:
    print("❌ MISSING DEPENDENCIES: pip install aiohttp numpy pynvml")
    sys.exit(1)

# --- Configuration ---
GPU_ID = 0
POLL_INTERVAL = 0.1

@dataclass
class Telemetry:
    timestamp: float
    power_w: float
    clock_mhz: int
    temp_c: int
    mem_used_mb: float
    gpu_util: int

@dataclass
class RequestMetric:
    req_id: str
    latency: float
    tokens: int
    ttft: float  # Time to first token (approximated for non-stream)
    success: bool

class HardwareMonitor:
    """Async Daemon for GPU Telemetry"""
    def __init__(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(GPU_ID)
        self.running = False
        self.data: List[Telemetry] = []

    async def start(self):
        self.running = True
        self.data = []
        while self.running:
            try:
                self.data.append(Telemetry(
                    timestamp=time.time(),
                    power_w=nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
                    clock_mhz=nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_GRAPHICS),
                    temp_c=nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU),
                    mem_used_mb=nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2,
                    gpu_util=nvmlDeviceGetUtilizationRates(self.handle).gpu
                ))
            except:
                pass
            await asyncio.sleep(POLL_INTERVAL)

    def stop(self):
        self.running = False

    def get_efficiency_score(self, total_tokens, duration):
        if not self.data: return {}
        avg_power = statistics.mean([d.power_w for d in self.data])
        energy_joules = avg_power * duration
        return {
            "avg_power_w": avg_power,
            "peak_temp_c": max([d.temp_c for d in self.data]),
            "avg_clock_mhz": statistics.mean([d.clock_mhz for d in self.data]),
            "tokens_per_watt": (total_tokens / duration) / avg_power if avg_power > 0 else 0,
            "tokens_per_joule": total_tokens / energy_joules if energy_joules > 0 else 0
        }

class TrafficGenerator:
    def __init__(self, url):
        self.url = url
        self.prompts = [
            "Explain the theory of relativity in one sentence.",
            "Write a Python function to reverse a string.",
            "What is the capital of France?",
            "List 3 benefits of GPU quantization.",
            "Debug: print('Hello World"
        ]

    async def send_request(self, session, req_id):
        start = time.time()
        payload = {
            "model": "llama-8b", # Model name matching your TRT engine
            "messages": [{"role": "user", "content": random.choice(self.prompts)}],
            "max_tokens": 128,
            "temperature": 0.7
        }
        try:
            async with session.post(f"{self.url}/v1/chat/completions", json=payload) as resp:
                if resp.status != 200:
                    return RequestMetric(req_id, 0, 0, 0, False)
                data = await resp.json()
                latency = time.time() - start
                tokens = data.get('usage', {}).get('completion_tokens', 0)
                return RequestMetric(req_id, latency, tokens, latency, True)
        except Exception:
            return RequestMetric(req_id, 0, 0, 0, False)

    async def run_poisson(self, duration_sec, target_rps):
        """Simulate real-world traffic using Poisson arrival times"""
        print(f"🌊 Starting Poisson Load: Target {target_rps} RPS for {duration_sec}s")
        start_time = time.time()
        tasks = []
        metrics = []
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < duration_sec:
                tasks.append(asyncio.create_task(self.send_request(session, f"req-{len(tasks)}")))
                # Poisson arrival: sleep time is exponentially distributed
                await asyncio.sleep(random.expovariate(target_rps))
            
            results = await asyncio.gather(*tasks)
            metrics = [r for r in results if r.success]
            
        return metrics

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8001")
    parser.add_argument("--rps", type=float, default=5.0)
    args = parser.parse_args()

    monitor = HardwareMonitor()
    gen = TrafficGenerator(args.url)

    # Start Monitoring
    monitor_task = asyncio.create_task(monitor.start())
    
    # Run Load
    metrics = await gen.run_poisson(duration_sec=20, target_rps=args.rps)
    
    # Stop Monitoring
    monitor.stop()
    await monitor_task
    
    # Analysis
    if not metrics:
        print("❌ No successful requests.")
        return

    duration = 20.0
    total_tokens = sum(m.tokens for m in metrics)
    hw_stats = monitor.get_efficiency_score(total_tokens, duration)
    latencies = [m.latency * 1000 for m in metrics]

    print(f"\n╔════════════ BENCHMARK REPORT ════════════╗")
    print(f"║ Requests:      {len(metrics)} (Realized RPS: {len(metrics)/duration:.1f})")
    print(f"║ Throughput:    {total_tokens/duration:.1f} tokens/sec")
    print(f"║ Latency (P95): {np.percentile(latencies, 95):.1f} ms")
    print(f"╠════════════ GPU EFFICIENCY ══════════════╣")
    print(f"║ Avg Power:     {hw_stats['avg_power_w']:.1f} W")
    print(f"║ Efficiency:    {hw_stats['tokens_per_watt']:.2f} Tok/Watt (The Golden Ratio)")
    print(f"║ Peak Temp:     {hw_stats['peak_temp_c']} C")
    print(f"╚══════════════════════════════════════════╝")

if __name__ == "__main__":
    asyncio.run(main())