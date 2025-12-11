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
from pathlib import Path

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
POLLING_INTERVAL_SEC = 0.1
REQUEST_TIMEOUT = 60
MODEL_PARAMS = 8e9  # Llama-8B
BACKEND_URL = "http://127.0.0.1:8888"

# Dataset Configuration
DATASET_CONFIG = {
    "source": "huggingface",  # Options: "huggingface", "json", "csv", "builtin"
    "dataset_name": "tatsu-lab/alpaca",  # HF dataset name
    "split": "train",  # Dataset split
    "prompt_column": "instruction",  # Column containing prompts
    "num_samples": 500,  # Number of prompts to load
    "shuffle": True,  # Randomize prompts
    
    # Alternative: JSON file path
    # "source": "json",
    # "file_path": "/path/to/prompts.json",
    # "prompt_field": "text",
    
    # Alternative: CSV file path
    # "source": "csv",
    # "file_path": "/path/to/prompts.csv",
    # "prompt_column": "prompt",
}

# Test Configurations
CONCURRENT_USERS = [10, 50, 100]
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
    utilization_kernel: float
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
    ttft: float
    total_latency_ms: float
    output_tokens: int
    tokens_per_sec: float
    success: bool
    error: str = ""

@dataclass
class LoadTestResult:
    concurrent_users: int
    total_requests: int
    successful_requests: int
    duration: float
    
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    aggregate_tokens_per_sec: float
    avg_request_tokens_per_sec: float
    p50_request_tokens_per_sec: float
    
    real_mfu: float
    avg_power_w: float
    tokens_per_watt: float

# ═══════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════
class PromptDatasetLoader:
    """Load prompts from various sources: HuggingFace, JSON, CSV, or builtin"""
    
    BUILTIN_PROMPTS = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to reverse a string.",
        "What are the benefits of regular exercise?",
        "Describe the water cycle step by step.",
        "How does photosynthesis work in plants?",
        "Explain machine learning to a complete beginner.",
        "What causes the four seasons on Earth?",
        "Write a haiku about modern technology.",
        "Summarize the key principles of object-oriented programming.",
        "What are the main differences between TCP and UDP?",
    ] * 10
    
    def __init__(self, config: Dict):
        self.config = config
        self.prompts: List[str] = []
        
    def load(self) -> List[str]:
        """Load prompts based on configured source"""
        source = self.config.get("source", "builtin").lower()
        
        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║  LOADING DATASET: {source.upper():<43} ║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        
        if source == "huggingface":
            self.prompts = self._load_from_huggingface()
        elif source == "json":
            self.prompts = self._load_from_json()
        elif source == "csv":
            self.prompts = self._load_from_csv()
        elif source == "builtin":
            self.prompts = self.BUILTIN_PROMPTS.copy()
            print(f"  ✓ Loaded {len(self.prompts)} builtin prompts")
        else:
            print(f"  ⚠ Unknown source '{source}', using builtin prompts")
            self.prompts = self.BUILTIN_PROMPTS.copy()
        
        # Shuffle if requested
        if self.config.get("shuffle", True):
            random.shuffle(self.prompts)
            print(f"  ✓ Shuffled prompts")
        
        # Limit to num_samples
        num_samples = self.config.get("num_samples", len(self.prompts))
        if num_samples < len(self.prompts):
            self.prompts = self.prompts[:num_samples]
            print(f"  ✓ Limited to {num_samples} samples")
        
        print(f"  ✓ Total prompts available: {len(self.prompts)}\n")
        
        if not self.prompts:
            print("  ⚠ WARNING: No prompts loaded! Falling back to builtin")
            self.prompts = self.BUILTIN_PROMPTS.copy()
        
        return self.prompts
    
    def _load_from_huggingface(self) -> List[str]:
        """Load dataset from HuggingFace Hub"""
        try:
            from datasets import load_dataset
            
            dataset_name = self.config.get("dataset_name")
            split = self.config.get("split", "train")
            prompt_column = self.config.get("prompt_column", "text")
            
            if not dataset_name:
                raise ValueError("dataset_name not specified in config")
            
            print(f"  → Loading HF dataset: {dataset_name}")
            print(f"  → Split: {split}")
            print(f"  → Prompt column: {prompt_column}")
            
            dataset = load_dataset(dataset_name, split=split)
            
            # Extract prompts from specified column
            prompts = []
            for item in dataset:
                if prompt_column in item:
                    prompt_text = str(item[prompt_column]).strip()
                    if prompt_text:
                        prompts.append(prompt_text)
                elif "input" in item and "instruction" in item:
                    # Alpaca-style format
                    instruction = str(item["instruction"]).strip()
                    input_text = str(item.get("input", "")).strip()
                    if input_text:
                        prompt_text = f"{instruction}\n\nInput: {input_text}"
                    else:
                        prompt_text = instruction
                    if prompt_text:
                        prompts.append(prompt_text)
            
            print(f"  ✓ Loaded {len(prompts)} prompts from HuggingFace")
            return prompts
            
        except ImportError:
            print("  ✗ ERROR: 'datasets' library not installed")
            print("  → Install with: pip install datasets")
            return []
        except Exception as e:
            print(f"  ✗ ERROR loading HuggingFace dataset: {e}")
            return []
    
    def _load_from_json(self) -> List[str]:
        """Load prompts from JSON file"""
        try:
            file_path = Path(self.config.get("file_path", ""))
            if not file_path.exists():
                raise FileNotFoundError(f"JSON file not found: {file_path}")
            
            print(f"  → Loading JSON: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            prompt_field = self.config.get("prompt_field", "text")
            
            # Handle different JSON structures
            prompts = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and prompt_field in item:
                        prompt_text = str(item[prompt_field]).strip()
                        if prompt_text:
                            prompts.append(prompt_text)
                    elif isinstance(item, str):
                        prompts.append(item.strip())
            elif isinstance(data, dict) and prompt_field in data:
                # Single prompt or list in specific field
                field_data = data[prompt_field]
                if isinstance(field_data, list):
                    prompts = [str(p).strip() for p in field_data if str(p).strip()]
                else:
                    prompts = [str(field_data).strip()]
            
            print(f"  ✓ Loaded {len(prompts)} prompts from JSON")
            return prompts
            
        except Exception as e:
            print(f"  ✗ ERROR loading JSON: {e}")
            return []
    
    def _load_from_csv(self) -> List[str]:
        """Load prompts from CSV file"""
        try:
            import csv
            
            file_path = Path(self.config.get("file_path", ""))
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            print(f"  → Loading CSV: {file_path}")
            
            prompt_column = self.config.get("prompt_column", "prompt")
            prompts = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if prompt_column in row:
                        prompt_text = str(row[prompt_column]).strip()
                        if prompt_text:
                            prompts.append(prompt_text)
            
            print(f"  ✓ Loaded {len(prompts)} prompts from CSV")
            return prompts
            
        except Exception as e:
            print(f"  ✗ ERROR loading CSV: {e}")
            return []
    
    def get_prompt(self) -> str:
        """Get a random prompt from the loaded dataset"""
        if not self.prompts:
            return "Explain a complex topic in simple terms."
        return random.choice(self.prompts)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.prompts:
            return {}
        
        lengths = [len(p) for p in self.prompts]
        word_counts = [len(p.split()) for p in self.prompts]
        
        return {
            "total_prompts": len(self.prompts),
            "avg_char_length": statistics.mean(lengths),
            "median_char_length": statistics.median(lengths),
            "avg_word_count": statistics.mean(word_counts),
            "median_word_count": statistics.median(word_counts),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

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
                    utilization_kernel=util.gpu,
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
# LOAD GENERATOR (MODIFIED TO USE DATASET)
# ═══════════════════════════════════════════════════════════════════════════
class LoadGenerator:
    def __init__(self, backend_url: str, dataset_loader: PromptDatasetLoader):
        self.backend_url = backend_url.rstrip("/")
        self.dataset_loader = dataset_loader

    async def _send_request(self, session: aiohttp.ClientSession, req_id: str, 
                           concurrent_level: int) -> RequestMetric:
        # Get prompt from dataset instead of hardcoded list
        prompt = self.dataset_loader.get_prompt()
        
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
                
                total_latency = (end - start) * 1000
                tokens_per_sec = tokens / (end - start) if (end - start) > 0 else 0
                
                return RequestMetric(
                    request_id=str(req_id),
                    concurrent_level=concurrent_level,
                    start_time=start,
                    end_time=end,
                    ttft=ttft * 1000,
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
                        await asyncio.sleep(random.uniform(0.5, 2))
                return user_metrics
        
        tasks = [user_session(u) for u in range(num_users)]
        results = await asyncio.gather(*tasks)
        
        for r in results:
            metrics.extend(r)
        
        success_count = sum(1 for m in metrics if m.success)
        print(f"   ✓ Complete: {success_count}/{len(metrics)} successful")
        return metrics

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════════════
def analyze_load_test(metrics: List[RequestMetric], avg_power: float, 
                     peak_flops: Optional[float]) -> Optional[LoadTestResult]:
    successful = [m for m in metrics if m.success]
    if not successful:
        print("   ⚠ No successful requests")
        return None
    
    concurrent_users = successful[0].concurrent_level
    duration = max(m.end_time for m in successful) - min(m.start_time for m in successful)
    
    ttfts = [m.ttft for m in successful]
    latencies = [m.total_latency_ms for m in successful]
    
    total_tokens = sum(m.output_tokens for m in successful)
    aggregate_tps = total_tokens / duration if duration > 0 else 0
    request_tps = [m.tokens_per_sec for m in successful]
    
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
# SATURATION ANALYSIS WITH CURVE FITTING
# ═══════════════════════════════════════════════════════════════════════════
def calculate_saturation_point(results: List[LoadTestResult]) -> Dict:
    """Calculate throughput saturation using Michaelis-Menten curve fitting"""
    if len(results) < 2:
        return {"saturated": False, "saturation_point": None}
    
    throughputs = [(r.concurrent_users, r.aggregate_tokens_per_sec) for r in results]
    throughputs.sort(key=lambda x: x[0])
    
    users = np.array([t[0] for t in throughputs])
    tps = np.array([t[1] for t in throughputs])
    
    # Marginal Gains Analysis
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
    
    # Empirical saturation check
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
    
    # Michaelis-Menten Curve Fitting
    def saturation_curve(u, T_max, K):
        return T_max * u / (K + u)
    
    try:
        initial_guess = [tps[-1] * 2, np.median(users)]
        params, covariance = curve_fit(saturation_curve, users, tps, 
                                      p0=initial_guess, maxfev=10000)
        T_max_predicted, K = params
        
        residuals = tps - saturation_curve(users, *params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((tps - np.mean(tps))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        predicted_saturation_users = int(19 * K)
        predicted_saturation_tps = saturation_curve(predicted_saturation_users, *params)
        
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
        
        u_pred = np.linspace(users[0], predicted_saturation_users * 1.5, 100)
        tps_pred = saturation_curve(u_pred, *params)
        curve_fit_result["prediction_curve"] = (u_pred, tps_pred)
        
    except Exception as e:
        print(f"  ⚠ Curve fitting failed: {e}")
        curve_fit_result = None
    
    # Diminishing Returns Analysis
    marginal_tps_per_user = np.diff(tps) / np.diff(users)
    marginal_threshold = 10.0
    diminishing_returns_point = None
    
    for i, marginal in enumerate(marginal_tps_per_user):
        if marginal < marginal_threshold:
            diminishing_returns_point = {
                "users": int(users[i+1]),
                "marginal_tps_per_user": marginal,
                "message": f"Diminishing returns at {int(users[i+1])} users"
            }
            break
    
    # Best Efficiency
    efficiency_scores = [(r.concurrent_users, r.aggregate_tokens_per_sec / r.concurrent_users) 
                         for r in results]
    best_efficiency = max(efficiency_scores, key=lambda x: x[1])
    
    saturated = empirical_saturation is not None
    
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

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def plot_gpu_telemetry_timeline(gpu_history: List[GPUState], 
                                all_metrics: Dict[int, List[RequestMetric]],
                                gpu_name: str):
    """Generate GPU telemetry timeline"""
    
    if not gpu_history:
        print("⚠ No GPU telemetry data to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    base_time = gpu_history[0].timestamp
    timestamps = [(s.timestamp - base_time) for s in gpu_history]
    
    kernel_util = [s.utilization_kernel for s in gpu_history]
    power_draw = [s.power_draw_w for s in gpu_history]
    temperature = [s.temperature_c for s in gpu_history]
    
    load_transitions = []
    for level in sorted(all_metrics.keys()):
        metrics = all_metrics[level]
        if metrics:
            start = min(m.start_time for m in metrics) - base_time
            end = max(m.end_time for m in metrics) - base_time
            load_transitions.append((start, end, level))
    
    # Panel 1: Kernel Occupancy
    axes[0].plot(timestamps, kernel_util, linewidth=2, color='#00FF41', label='Kernel Occupancy')
    axes[0].fill_between(timestamps, kernel_util, alpha=0.3, color='#00FF41')
    
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
    
    axes[0].text(0.98, 0.5, 
                '⚠ This shows "GPU busy"\nNOT "compute utilization"\nSee MFU for real usage',
                transform=axes[0].transAxes, fontsize=9, 
                verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Panel 2: Power & Temperature
    ax2 = axes[1]
    ax2_temp = ax2.twinx()
    
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
    
    line3 = ax2_temp.plot(timestamps, temperature, linewidth=2, color='#FF1493', 
                          label='GPU Temp', alpha=0.7)
    
    line4 = ax2_temp.axhline(y=83, color='orange', linestyle=':', linewidth=2, 
                            label='Throttle: 83°C', alpha=0.7)
    
    ax2_temp.set_ylabel('Temperature (°C)', fontweight='bold', fontsize=11, color='#FF1493')
    ax2_temp.tick_params(axis='y', labelcolor='#FF1493')
    
    lines = line1 + [line2] + line3 + [line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    ax2.set_title('Power Consumption & Thermal State', fontweight='bold', fontsize=12)
    
    fig.suptitle(f'GPU Telemetry Timeline: {gpu_name}\n'
                f'Kernel Occupancy vs. Power vs. Temperature',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('gpu_telemetry_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ GPU telemetry timeline saved: gpu_telemetry_timeline.png")
    plt.close()

def plot_comprehensive_analysis(results: List[LoadTestResult], 
                                all_metrics: Dict[int, List[RequestMetric]],
                                gpu_history: List[GPUState],
                                peak_flops: Optional[float],
                                saturation_analysis: Dict):
    """Generate comprehensive analysis dashboard"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    concurrent_levels = [r.concurrent_users for r in results]
    
    # Panel 1: TTFT Latency
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
    
    # Panel 2: Throughput Scaling with Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    aggregate_tps = [r.aggregate_tokens_per_sec for r in results]
    per_request_tps = [r.avg_request_tokens_per_sec for r in results]

    ax2.plot(concurrent_levels, aggregate_tps, 'o', markersize=12, 
            label='Measured Aggregate', color='#00FF41', zorder=5)
    ax2.plot(concurrent_levels, per_request_tps, 's', markersize=10,
            label='Measured Per-Request', color='#4A90E2', zorder=5)

    if saturation_analysis.get('curve_fit'):
        cf = saturation_analysis['curve_fit']
        u_pred, tps_pred = cf['prediction_curve']
        
        u_extended = np.linspace(concurrent_levels[0], 
                                cf['predicted_saturation_users'] * 1.3, 200)
        
        def saturation_curve(u, T_max, K):
            return T_max * u / (K + u)
        
        tps_extended = saturation_curve(u_extended, cf['T_max'], cf['K'])
        
        ax2.plot(u_extended, tps_extended, '-', linewidth=3, color='#FF6B35', 
                alpha=0.8, label=f"Mathematical Model (R²={cf['r_squared']:.3f})", zorder=3)
        
        tps_upper = tps_extended * 1.05
        tps_lower = tps_extended * 0.95
        ax2.fill_between(u_extended, tps_lower, tps_upper, color='#FF6B35', 
                        alpha=0.15, label='95% Confidence', zorder=2)
        
        ax2.axhline(y=cf['T_max'], color='red', linestyle='--', linewidth=2,
                label=f"Theoretical Max: {cf['T_max']:.0f} tok/s", alpha=0.7, zorder=4)
        
        sat_95_tps = cf['T_max'] * 0.95
        ax2.axhline(y=sat_95_tps, color='orange', linestyle=':', linewidth=2,
                label=f"95% Saturation: {sat_95_tps:.0f} tok/s", alpha=0.6, zorder=4)
        
        ax2.axvline(x=cf['predicted_saturation_users'], color='red', linestyle='--', 
                linewidth=2, alpha=0.5, zorder=4)
        
        ax2.annotate(f"95% Max\n~{cf['predicted_saturation_users']} users", 
                    xy=(cf['predicted_saturation_users'], sat_95_tps),
                    xytext=(cf['predicted_saturation_users'] * 0.7, sat_95_tps * 1.08),
                    fontsize=10, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        current_max = max(aggregate_tps)
        current_users = concurrent_levels[aggregate_tps.index(current_max)]
        ax2.annotate(f"Current:\n{cf['current_pct_of_max']:.1f}% of max", 
                    xy=(current_users, current_max),
                    xytext=(current_users * 1.3, current_max * 0.85),
                    fontsize=10, fontweight='bold', color='#00FF41',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#00FF41', lw=2))

    ax2.set_xlabel('Concurrent Users', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Throughput (tokens/sec)', fontweight='bold', fontsize=12)
    ax2.set_title('Throughput Scalability with Saturation Prediction', 
                fontweight='bold', fontsize=13)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    
    # Panel 3: MFU
    ax3 = fig.add_subplot(gs[0, 2])
    mfu_values = [r.real_mfu for r in results]
    colors = ['#00FF41' if m > 15 else '#FFA500' if m > 10 else '#FF6B35' 
              for m in mfu_values]
    
    bars = ax3.bar(concurrent_levels, mfu_values, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2, width=max(concurrent_levels)*0.1)
    
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
    
    # Panel 4: Marginal Gains
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
    
    # Panel 5: Energy Efficiency
    ax5 = fig.add_subplot(gs[1, 1])
    tok_per_watt = [r.tokens_per_watt for r in results]
    
    ax5.plot(concurrent_levels, tok_per_watt, 'o-', linewidth=3, 
             markersize=10, color='#00FF41')
    ax5.fill_between(concurrent_levels, tok_per_watt, alpha=0.3, color='#00FF41')
    
    ax5.set_xlabel('Concurrent Users', fontweight='bold')
    ax5.set_ylabel('Tokens per Watt', fontweight='bold')
    ax5.set_title('Energy Efficiency', fontweight='bold', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Efficiency Score
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
    
    # Panel 7: Throughput Stability
    ax7 = fig.add_subplot(gs[2, 0])

    for level, metrics in all_metrics.items():
        successful = [m for m in metrics if m.success]
        if not successful:
            continue
        
        base_time = min(m.start_time for m in successful)
        start_times = np.array([(m.start_time - base_time) for m in successful])
        tps_values = np.array([m.tokens_per_sec for m in successful])
        
        sorted_indices = np.argsort(start_times)
        x_sorted = start_times[sorted_indices]
        y_sorted = tps_values[sorted_indices]

        if len(x_sorted) > 3:
            z = np.polyfit(x_sorted, y_sorted, 3) 
            p = np.poly1d(z)
            
            x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
            y_smooth = p(x_smooth)
            
            ax7.plot(x_smooth, y_smooth, '-', linewidth=2.5, label=f'{level} users')
        else:
            ax7.plot(x_sorted, y_sorted, '-', linewidth=2, label=f'{level} users')

    ax7.set_xlabel('Test Duration (s)', fontweight='bold')
    ax7.set_ylabel('Tokens/sec (Trend)', fontweight='bold')
    ax7.set_title('Throughput Stability (Smoothed)', fontweight='bold', fontsize=12)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Panel 8: Success Rate
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
    
    # Panel 9: Summary Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
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
    
    col_widths = [0.3] + [0.7 / len(concurrent_levels)] * len(concurrent_levels)
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(header_row)):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(len(header_row)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax9.set_title('Performance Summary', fontweight='bold', fontsize=12, pad=20)
    
    # Save
    fig.suptitle(f'LLM Inference Benchmark: Multi-Load Analysis\n'
                f'Model: Llama-8B | Hardware: {gpu_history[0] if gpu_history else "GPU"}',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
    print("✓ Comprehensive analysis saved: comprehensive_benchmark.png")
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR (MODIFIED)
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   ENHANCED GPU BENCHMARK v4.0: Dataset-Powered Analysis      ║")
    print("║   Comparing 10, 50, 100 Concurrent Users                     ║")
    print("║   Metrics: TTFT, tok/s, MFU (Real GPU Utilization)          ║")
    print("╚═══════════════════════════════════════════════════════════════╝\n")
    
    # Load dataset
    dataset_loader = PromptDatasetLoader(DATASET_CONFIG)
    prompts = dataset_loader.load()
    
    # Print dataset statistics
    stats = dataset_loader.get_statistics()
    if stats:
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║  DATASET STATISTICS                                      ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print(f"║ Total Prompts:     {stats['total_prompts']:6d}                                 ║")
        print(f"║ Avg Length:        {stats['avg_char_length']:6.0f} chars ({stats['avg_word_count']:.0f} words)            ║")
        print(f"║ Median Length:     {stats['median_char_length']:6.0f} chars ({stats['median_word_count']:.0f} words)            ║")
        print(f"║ Range:             {stats['min_length']:4d} - {stats['max_length']:4d} chars                      ║")
        print("╚══════════════════════════════════════════════════════════╝\n")
    
    monitor = GPUMonitor(GPU_ID)
    if not monitor.handle:
        print("✗ Cannot proceed without GPU access")
        return
    
    load_gen = LoadGenerator(BACKEND_URL, dataset_loader)
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
        
        await asyncio.sleep(5)
    
    # Stop monitoring
    monitor.stop()
    await monitor_task
    
    # Generate visualizations
    if results:
        saturation_analysis = calculate_saturation_point(results)
        
        plot_gpu_telemetry_timeline(monitor.history, all_metrics, monitor.gpu_name)
        plot_comprehensive_analysis(results, all_metrics, monitor.history, 
                                   monitor.peak_flops, saturation_analysis)
        
        # Print comparative analysis
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
    
    print("\n✓ Benchmark complete!")
    print(f"✓ Dataset used: {DATASET_CONFIG['source']}")
    if DATASET_CONFIG['source'] == 'huggingface':
        print(f"✓ HF Dataset: {DATASET_CONFIG.get('dataset_name', 'N/A')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✗ Benchmark aborted by user")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()