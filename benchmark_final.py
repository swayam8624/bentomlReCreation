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

# [REST OF THE CODE REMAINS THE SAME - saturation analysis and visualization functions]
# [Include all the plot functions and main orchestrator here]

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR (MODIFIED)
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║   ENHANCED GPU BENCHMARK : Dataset-Powered Analysis      ║")
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
    
    # Rest of main() remains the same - analysis and visualization
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