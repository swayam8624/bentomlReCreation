# Part 1: SGLang Experimentation Guide

### **The Iron Rule**

**SGLang requires Python 3.10.**
Do not use Python 3.11 or 3.12. The required wheels (kernels) do not exist for newer Python versions, leading to the dependency hell we previously navigated.

### **1. Environment Setup (Clean Slate)**

Run these commands strictly in order to create a fresh, stable environment on RunPod.

```bash
# 1. Kill any lingering processes on the target port (e.g., Jupyter)
apt-get update && apt-get install -y psmisc
fuser -k 8888/tcp

# 2. Install Python 3.10 tools and create the venv
apt-get install -y python3.10-venv python3.10-dev
python3.10 -m venv venv-sglang
source venv-sglang/bin/activate

# 3. Upgrade pip (mandatory for modern wheel support)
pip install --upgrade pip
```

### **2. Install Dependencies (The Holy Trinity)**

We pin versions to ensure compatibility with RunPod's standard CUDA 12.1 architecture (Ampere/Ada).

```bash
# 1. Install PyTorch 2.4 (The Stable Foundation)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Install FlashInfer (SGLang's Engine) - Must match Torch 2.4 exactly
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# 3. Install SGLang
pip install "sglang[all]"
```

### **3. The Launch Command**

Use this command to start the server.

- **Note:** We use `0.0.0.0` to expose the host and port `8888` (after killing Jupyter).

<!-- end list -->

```bash
# Optional: Disable DeepGemm if running on older GPUs or standard dense models (Llama)
export SGLANG_DISABLE_DEEPGEMM=1

python3 -m sglang.launch_server \
  --model-path /workspace/models/llama-8b-hf \
  --host 0.0.0.0 \
  --port 8888 \
  --mem-fraction-static 0.90 \
  --context-length 8192
```

```bash
python3 /workspace/BentomlReCreation/benchmark_v5.py
```

---

### **4. Potholes & Landmines (The "Kill List")**

- **The Version Trap:** If you use Python 3.12, `pip` pulls broken/incompatible wheels. **Stick to 3.10.**
- **The Port Conflict:** Port `8888` is owned by Jupyter by default. SGLang will crash with `Address already in use` unless you run `fuser -k 8888/tcp` first.
- **The Architecture Mismatch:** If you see `sm100` errors, you installed the Nvidia Blackwell kernel. You likely didn't use the specific index URLs provided above.
- **The "Weird Lines" in Logs:**
  - Lines like `#running-req: 11` or `#token: 921` are **telemetry**, not errors.
  - They indicate the server is healthy, actively batching tokens, and utilizing the RadixAttention cache.

---

# Part 2: Experiment Report (v5)

**Date:** 2025-12-10
**Hardware:** Single Nvidia A40 GPU
**Comparison:** SGLang (v0.4.9) vs. LMDeploy (v0.6.0)

## 1\. Executive Summary

**Winner: LMDeploy**
LMDeploy outperformed SGLang significantly across all metrics, delivering **24-37% higher throughput** and superior latency characteristics. SGLang's theoretical advantages (RadixAttention) did not translate to performance gains in this specific workload, likely due to low prefix overlap.

## 2\. Head-to-Head Performance Data

| Metric                | SGLang (Best)        | LMDeploy (Best)          | Delta                        |
| :-------------------- | :------------------- | :----------------------- | :--------------------------- |
| **Peak Throughput**   | 1637.6 tok/s (@100u) | **2023.3 tok/s** (@100u) | **LMDeploy +23.6%**          |
| **Per-Request T-put** | 21.6 tok/s           | **28.3 tok/s**           | **LMDeploy +31.0%**          |
| **MFU (Utilization)** | 17.50%               | **21.63%**               | **LMDeploy +23.6%**          |
| **TTFT (P95 @ 50u)**  | 5493 ms              | **4272 ms**              | **LMDeploy -22.2%** (Faster) |
| **Efficiency**        | 22.5 tok/s/user      | **34.5 tok/s/user**      | **LMDeploy +53.0%**          |

## 3\. Scaling & Efficiency Analysis

### A. Throughput Scaling

- **SGLang:** Scaled from 224 tok/s (10 users) to 1637 tok/s (100 users).
- **LMDeploy:** Scaled from 238 tok/s (10 users) to 2023 tok/s (100 users).
- **Insight:** LMDeploy scales more aggressively under load, suggesting superior kernel-level batching optimization for this specific architecture (A40).

### B. Hardware Utilization (MFU)

- **Observation:** Both frameworks are memory-bound (MFU \< 25%). The A40's compute potential (149 TFLOPS) is largely idle.
- **Efficiency:** LMDeploy extracts 23% more performance from the exact same VRAM bandwidth.

## 4\. Telemetry & Log Analysis

During the SGLang run, the logs displayed high activity in `Prefill batch` metrics:

- `#cached-token: 33` vs `#new-token: 2`
- **Interpretation:** SGLang was successfully hitting its Radix cache, but the overhead of managing this cache (RadixAttention) likely outweighed the benefits for this specific traffic pattern, contributing to the lower throughput compared to LMDeploy's simpler, raw-speed approach.

## 5\. Conclusion & Recommendations

1.  **Production Recommendation:** Deploy **LMDeploy**. It provides a 30%+ free performance boost and snappier user experience (22% lower latency).
2.  **SGLang Viability:** SGLang should only be reconsidered if the workload shifts to heavy **multi-turn conversations** or **constrained JSON generation**, where its specialized features (RadixAttention, compressed state) outshine raw throughput.
3.  **Next Steps (v6):**
    - Update SGLang configuration to `v6` settings.
    - Tune SGLang aggressively: Increase `mem-fraction-static` to 0.95 and test `schedule-policy lpm` (Longest Prefix Match) to try and close the gap.

**Status:** Experiment v5 Concluded. Configuration update scheduled for v6 run.
