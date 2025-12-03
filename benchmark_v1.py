# 1) Overwrite the script with the fixed version
cat > /workspace/bentomlReCreation/benchmark_v1.py << 'PY'
import asyncio
import time
import json
import statistics
import random
import sys
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

try:
    import aiohttp
    import numpy as np
    from pynvml import (
        nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage,
        nvmlDeviceGetClockInfo, nvmlDeviceGetTemperature,
        NVML_CLOCK_GRAPHICS, NVML_TEMPERATURE_GPU
    )
except ImportError as e:
    print(f"FATAL: Missing dependency: {e}")
    print("Run: pip install aiohttp numpy pynvml")
    sys.exit(1)

# Configuration
GPU_ID = 0
POLLING_INTERVAL_SEC = 0.1  # High resolution monitoring
REQUEST_TIMEOUT = 60  # seconds per request

@dataclass
class GPUState:
    timestamp: float
    utilization_gpu: float
    memory_used_mb: float
    power_draw_w: float
    clock_sm_mhz: float
    temperature_c: float

@dataclass
class RequestMetric:
    request_id: str
    start_time: float
    end_time: float
    latency_ms: float
    output_tokens: int
    success: bool
    ttft_ms: float = 0.0
    error: str = ""

@dataclass
class BenchmarkResult:
    backend: str
    mode: str
    total_requests: int
    duration: float
    rps: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_tokens_per_sec: float
    avg_power_w: float
    tokens_per_watt: float
    peak_memory_mb: float
    avg_sm_clock_mhz: float
    max_temp_c: float

class GPUMonitor:
    def __init__(self, device_index: int = 0):
        try:
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(device_index)
            self.running = False
            self.history: List[GPUState] = []
        except Exception as e:
            print(f"Failed to init NVML: {e}")
            self.handle = None

    async def start_monitoring(self):
        if not self.handle:
            return
        self.running = True
        self.history = []
        print("   >>> GPU Telemetry Active (Power, Clocks, Thermal)")
        while self.running:
            try:
                util = nvmlDeviceGetUtilizationRates(self.handle)
                mem = nvmlDeviceGetMemoryInfo(self.handle)
                power = nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                clock = nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_GRAPHICS)
                temp = nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU)
                self.history.append(GPUState(
                    timestamp=time.time(),
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

    def get_stats(self) -> Dict:
        if not self.history:
            return {}
        powers = [x.power_draw_w for x in self.history]
        clocks = [x.clock_sm_mhz for x in self.history]
        temps = [x.temperature_c for x in self.history]
        mems = [x.memory_used_mb for x in self.history]
        return {
            "avg_power_w": statistics.mean(powers),
            "max_power_w": max(powers),
            "avg_clock_mhz": statistics.mean(clocks),
            "max_temp_c": max(temps),
            "peak_memory_mb": max(mems),
            "avg_utilization": statistics.mean([x.utilization_gpu for x in self.history])
        }

class LoadGenerator:
    def __init__(self, backend_url: str):
        # backend_url should be like "http://127.0.0.1:8888"
        self.backend_url = backend_url.rstrip("/")
        self.prompts = [
            "Explain quantum entanglement simply.",
            "Write a python script to sort a list.",
            "What are the 3 laws of thermodynamics?",
            "Summarize the history of Rome in 50 words.",
            "Debug this code: print('hello world')"
        ] * 10

    async def _send_request(self, session: aiohttp.ClientSession, req_id: str) -> RequestMetric:
        prompt = random.choice(self.prompts)
        start = time.time()
        payload = {
            "model": "/workspace/bentomlReCreation/models/llama-8b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.7
        }
        url = f"{self.backend_url}/v1/chat/completions"
        try:
            async with session.post(url, json=payload, timeout=REQUEST_TIMEOUT) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return RequestMetric(str(req_id), start, time.time(), (time.time()-start)*1000, 0, False, error=f"HTTP {resp.status} - {text[:400]}")
                data = json.loads(text)
                end = time.time()
                # Prefer official usage if present
                tokens = data.get("usage", {}).get("completion_tokens", 0) if isinstance(data, dict) else 0
                # Fallback estimate: count tokens by splitting returned text if choices present
                if not tokens:
                    try:
                        choices = data.get("choices", [])
                        if choices and isinstance(choices, list):
                            content = ""
                            c = choices[0]
                            # try a few schema variants
                            if "message" in c and isinstance(c["message"], dict):
                                content = c["message"].get("content", "")
                            elif "text" in c:
                                content = c.get("text", "")
                            tokens = max(1, len(content.split()))
                    except Exception:
                        tokens = 0
                return RequestMetric(
                    request_id=str(req_id),
                    start_time=start,
                    end_time=end,
                    latency_ms=(end-start)*1000,
                    output_tokens=tokens,
                    success=True
                )
        except Exception as e:
            return RequestMetric(str(req_id), start, time.time(), (time.time()-start)*1000, 0, False, error=str(e))

    async def run_poisson_arrival(self, duration_sec: int, target_rps: float) -> List[RequestMetric]:
        print(f"   >>> Starting Poisson Load: Target {target_rps} RPS for {duration_sec}s")
        metrics = []
        start_time = time.time()
        req_count = 0
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            while time.time() - start_time < duration_sec:
                tasks.append(asyncio.create_task(self._send_request(session, req_count)))
                req_count += 1
                sleep_time = random.expovariate(target_rps)
                await asyncio.sleep(sleep_time)
            print(f"   >>> Draining {len(tasks)} active requests...")
            results = await asyncio.gather(*tasks)
            metrics = list(results)
        return metrics

    async def run_concurrent_users(self, num_users: int, requests_per_user: int) -> List[RequestMetric]:
        print(f"   >>> Starting User Simulation: {num_users} users, {requests_per_user} reqs each")
        metrics = []
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def user_behavior(user_id):
                user_metrics = []
                for i in range(requests_per_user):
                    await asyncio.sleep(random.uniform(1, 3))
                    m = await self._send_request(session, f"{user_id}-{i}")
                    user_metrics.append(m)
                return user_metrics
            tasks = [user_behavior(u) for u in range(num_users)]
            results = await asyncio.gather(*tasks)
            for r in results:
                metrics.extend(r)
        return metrics

async def analyze_results(backend_name: str, mode: str, metrics: List[RequestMetric], gpu_stats: Dict, duration: float) -> BenchmarkResult:
    successful = [m for m in metrics if m.success]
    if not successful:
        print("❌ CRITICAL: No successful requests.")
        # print some failures for debugging
        for m in metrics[:10]:
            print("  FAIL:", m.request_id, m.error)
        return None
    latencies = [m.latency_ms for m in successful]
    total_tokens = sum([m.output_tokens for m in successful])
    avg_power = gpu_stats.get('avg_power_w', 1)
    tok_per_watt = (total_tokens / duration) / avg_power if avg_power > 0 else 0
    return BenchmarkResult(
        backend=backend_name,
        mode=mode,
        total_requests=len(metrics),
        duration=duration,
        rps=len(successful) / duration,
        avg_latency_ms=statistics.mean(latencies),
        p95_latency_ms=np.percentile(latencies, 95),
        p99_latency_ms=np.percentile(latencies, 99),
        avg_tokens_per_sec=total_tokens / duration,
        avg_power_w=avg_power,
        tokens_per_watt=tok_per_watt,
        peak_memory_mb=gpu_stats.get('peak_memory_mb', 0),
        avg_sm_clock_mhz=gpu_stats.get('avg_clock_mhz', 0),
        max_temp_c=gpu_stats.get('max_temp_c', 0)
    )

def print_report(res: BenchmarkResult):
    if not res:
        return
    print(f"\n╔════════════ BENCHMARK REPORT: {res.backend.upper()} ════════════╗")
    print(f"║ Mode: {res.mode:<49} ║")
    print(f"╠══════════════════════ PERFORMANCE ═════════════════════════╣")
    print(f"║ Throughput:      {res.avg_tokens_per_sec:8.1f} tok/s  | RPS: {res.rps:8.1f}       ║")
    print(f"║ Latency (Avg):   {res.avg_latency_ms:8.1f} ms     | P99: {res.p99_latency_ms:8.1f} ms   ║")
    print(f"╠══════════════════════ GPU EFFICIENCY ══════════════════════╣")
    print(f"║ Power Draw:      {res.avg_power_w:8.1f} W      | Peak Temp: {res.max_temp_c:4.1f}°C   ║")
    print(f"║ Efficiency:      {res.tokens_per_watt:8.2f} tok/W  | SM Clock:  {res.avg_sm_clock_mhz:4.0f}MHz  ║")
    print(f"║ Memory Peak:     {res.peak_memory_mb:8.0f} MB     |                        ║")
    print(f"╚════════════════════════════════════════════════════════════╝\n")

async def main():
    # point at the running lmdeploy instance on 8888
    backends = [
        ("http://127.0.0.1:8888", "LMDeploy"),
    ]
    print("Initializing Benchmark Suite...")
    monitor = GPUMonitor(GPU_ID)
    for url, name in backends:
        load_gen = LoadGenerator(url)
        # Poisson test
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        metrics_poisson = await load_gen.run_poisson_arrival(duration_sec=15, target_rps=10)
        monitor.stop()
        await monitor_task
        res_poisson = await analyze_results(name, "Poisson (Random Arrival)", metrics_poisson, monitor.get_stats(), 15.0)
        print_report(res_poisson)
        await asyncio.sleep(2)
        # Concurrent users
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        metrics_users = await load_gen.run_concurrent_users(num_users=20, requests_per_user=4)
        monitor.stop()
        await monitor_task
        if metrics_users:
            dur = (max(m.end_time for m in metrics_users) - min(m.start_time for m in metrics_users))
            res_users = await analyze_results(name, "Concurrent Users (Sessions)", metrics_users, monitor.get_stats(), dur)
            print_report(res_users)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAborted.")
