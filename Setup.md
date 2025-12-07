# Setup & Benchmark Guide

**RunPod + LMDeploy + Llama 3 8B + MFU Stress Test**

This guide walks through **every step needed on a fresh RunPod instance** to:

- Set up the environment
- Download and prepare the model
- Start the LMDeploy server
- Run the GPU Stress Test & MFU Benchmark Suite
- Save and/or sync results

It includes **expected outputs** so you can confirm each step succeeded.

---

# 1. 🔧 Create Isolated Python Environment

RunPod images often contain conflicting system packages.
We isolate everything:

```bash
python3 -m venv /workspace/llm-env
source /workspace/llm-env/bin/activate
pip install --upgrade pip setuptools wheel
```

**Expected output:**

```
Successfully installed pip-x.x.x setuptools-x.x.x wheel-x.x.x
```

---

# 2. 📦 Install Required Packages

Your repository already includes `requirements.txt`.

```bash
pip install -r /workspace/bentomlReCreation/requirements.txt
```

Expected output (truncated):

```
Installing collected packages: pynvml, aiohttp, lmdeploy, ...
Successfully installed ...
```

If CUDA compatibility warnings appear, they are normal.

---

# 3. ⬇️ Download Llama 3 8B Model (HF)

### 3.1 Login to HF

```bash
huggingface-cli login
```

Paste token → Expected output:

```
Token is valid. Logged in as <username>
```

### 3.2 Download Only the Necessary Weights

We use HF’s optimized download pattern:

```bash
mkdir -p /workspace/models/llama-8b-hf

huggingface-cli download \
    meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir /workspace/models/llama-8b-hf \
    --include="*.json" \
    --include="*.safetensors"
```

**Expected output:**

```
Fetching 4 shards: model-00001-of-00004.safetensors ...
Completed ✓
```

**IMPORTANT NOTE FOR FORKERS:**
This model requires accepting a license on HuggingFace before downloading.

---

# 4. 🚀 Launch LMDeploy API Server

We use **TurboMind** backend for high throughput (faster than PyTorch backend).

```bash
nohup lmdeploy serve api_server \
    /workspace/models/llama-8b-hf \
    --server-port 8888 \
    --backend pytorch \
    --tp 1 \
    --cache-max-entry-count 0.9 \
    --max-batch-size 128 \
    --device cuda \
    --dtype float16 \
    > /workspace/server.log 2>&1 &
```

### Expected behavior:

Run `tail` to confirm load:

```bash
tail -n 50 /workspace/server.log
```

You should see:

```
[LlamaTritonModel] max_context_token_num = 8192
[pytorch] Model loaded successfully
Uvicorn running on http://0.0.0.0:8888
```

- using `pytorch` backend is more compatible but slower than `turbomind`. Using `turbomind` does not allow open_api calls in the benchmark script.

### Verify API Is Alive:

```bash
curl http://127.0.0.1:8888/v1/models
```

Expected JSON:

```json
{
  "object": "list",
  "data": [
    {
      "id": "/workspace/models/llama-8b-hf",
      "object": "model",
      "owned_by": "lmdeploy"
    }
  ]
}
```

---

# 5. 🔥 Run the GPU Stress Test / MFU Benchmark

From inside the environment:

```bash
source /workspace/llm-env/bin/activate
python -u /workspace/bentomlReCreation/benchmark_v2.py
```

### Expected output (first lines):

```
GPU STRESS TEST & MFU BENCHMARK SUITE v2.0
GPU Detected: NVIDIA A40
Phase 1: Warmup
...
```

At the end, you will see:

```
Throughput: XXX tok/s
MFU: YY%
Latency P50/P95/P99: ...
Saved telemetry graph → gpu_stress_telemetry.png
```

---

# 6. 📊 Retrieve Graphs From RunPod

If you ran:

```bash
python3 -m http.server 9000
```

And your graph exists at:

```
/workspace/bentomlReCreation/gpu_stress_telemetry.png
```

Then from **your laptop**:

```bash
curl -O http://<RUNPOD_IP>:9000/gpu_stress_telemetry.png
```

Or using SCP:

```bash
scp -P <RUNPOD_SSH_PORT> root@<RUNPOD_IP>:/workspace/bentomlReCreation/gpu_stress_telemetry.png .
```

Or use Git to commit results (if small enough).

---

# 7. 🛠 Troubleshooting

### Check what is running on port 8888:

```bash
lsof -i :8888
```

If something like Jupyter is using it:

```bash
pkill -f jupyter
```

### Stop a stuck LMDeploy server:

```bash
pkill -f lmdeploy
```

### Watch server logs:

```bash
tail -f /workspace/server.log
```

### GPU Monitor:

```bash
watch -n 1 nvidia-smi
```

---

# 8. 🧪 Notes for People Forking This Repo

- This repo assumes **RunPod**, **Ubuntu**, **NVIDIA GPU**, **CUDA 12**
- LMDeploy TurboMind backend requires **FP16-compatible GPUs**
- You **must** manually download the Llama model (cannot commit it)
- Never commit:

  - HF tokens
  - GitHub tokens
  - server.log
  - model weights

If you fork this repo without understanding LMDeploy, use:

```bash
bash setup.sh
```

(if present in repo)

---

# 9. ✔️ Optional: Auto-Start Server on Boot

Create:

`/workspace/start_lmdeploy.sh`

```bash
#!/bin/bash
source /workspace/llm-env/bin/activate
nohup lmdeploy serve api_server \
    /workspace/models/llama-8b-hf \
    --server-port 8888 \
    --backend pytorch \
    --dtype float16 \
    > /workspace/server.log 2>&1 &
```
