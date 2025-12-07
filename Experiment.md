# **EXPERIMENTATION_JOURNEY.md**

---

# # **Experimentation Journey**

_A complete chronological record of debugging, learning, and stabilizing LMDeploy + Llama 3 + TRT-LLM on RunPod._

This document serves as a **post-mortem**, **debugging reference**, and **engineering diary** for the entire inference/benchmarking pipeline:
HF downloads → LMDeploy → TRT-LLM engine building → GPU telemetry → GitHub → environment repairs → Python issues → concurrency testing.

---

# ## **1. Initial Model Loading Failures**

### **Issue — HF Model Directory Missing Metadata**

First errors appeared when checking the config:

```bash
FileNotFoundError: '/workspace/models/llama-8b/config.json'
```

### **Root Cause**

Model folder lacked:

- config.json
- tokenizer files
- safetensor shards

### **Fix**

We switched to HF’s supported download:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --include="*.safetensors" --include="*.json" \
    --local-dir /workspace/models/llama-8b
```

### **Observation**

Download fails unless the HF account has accepted Meta’s license.

---

# ## **2. LMDeploy “Could not find model architecture”**

Running:

```bash
lmdeploy serve api_server /workspace/models/llama-8b
```

Error:

```
RuntimeError: Could not find model architecture from config
```

### **Cause**

Missing `"architectures"` in config.json because files were incomplete.

### **Fix**

Downloaded complete HF metadata → LMDeploy detected architecture normally.

---

# ## **3. Port 8888 Conflicts with Jupyter**

LMDeploy failed:

```
address already in use: 8888
```

### **Cause**

RunPod launches Jupyter automatically on port 8888.

### **Fix**

Stopped Jupyter:

```bash
pkill -f jupyter
```

---

# ## **4. Benchmark Flooding 404 Errors**

Benchmark spammed:

```
POST /v1/chat/completions → 404 Not Found
```

### **Root Causes**

1. TurboMind does NOT support `/v1/chat/completions`.
2. Benchmark was firing requests **before** the model fully loaded.

### **Fixes**

- Updated benchmarking endpoint (`/v1/completions`).
- Added server-readiness check:

  ```bash
  curl http://127.0.0.1:8888/v1/models
  ```

---

# ## **5. benchmark_v2 Reporting 0 Successful Requests**

Output:

```
Warmup: 0 successful
Stress: 0 successful
```

### **Root Causes**

- Endpoint mismatch
- Model wasn’t ready
- PyTorch backend being used (slow & unstable for 8B+)

### **Fix**

Switched to:

```
--backend turbomind
--dtype float16
```

TurboMind = correct batching, correct endpoints, high throughput.

---

# ## **6. GPU Telemetry Saved in Wrong Directory**

Graph generated but placed at:

```
/gpu_stress_telemetry.png
```

Repo expected:

```
/workspace/bentomlReCreation/gpu_stress_telemetry.png
```

### **Fix**

Hard-coded absolute path in benchmark script.

---

# ## **7. GitHub Push Failures — Token Issues**

Error:

```
Permission denied
```

### **Causes**

- Token without `repo` scope
- Incorrect remote URL
- GitHub blocking password login

### **Fix**

```bash
git remote set-url origin https://<TOKEN>@github.com/swayam8624/bentomlReCreation.git
git add .
git commit -m "results"
git push origin main
```

✔ Success.

---

# ## **8. File Transfer (RunPod → Local Machine)**

Used:

### **Option 1 — HTTP**

```bash
python3 -m http.server 9000
```

### **Option 2 — SCP**

```bash
scp -P <PORT> root@<IP>:/workspace/bentomlReCreation/gpu_stress_telemetry.png ./
```

---

# ## **9. Poisson vs Max Concurrency Benchmarking**

### **Poisson arrival**

- Models real human traffic
- Good for latency SLO testing
- Not good for finding max throughput

### **Concurrency saturation**

- Pushes GPU to the limit
- Best for MFU and raw token/s capacity

### **Conclusion:**

**Concurrency test = correct choice** for this experiment.

---

# # **10. Key Learnings (Earlier Phases)**

- Always kill Jupyter on RunPod if using port 8888
- LMDeploy TurboMind backend is far more efficient than PyTorch
- HF metadata must be complete — otherwise model cannot load
- Telemetry graphs require absolute paths in containers
- GitHub PAT tokens must have correct scopes

---

# # **11. EXTENDED TIMELINE (This Chat Session)**

### _Full chronological debugging session fixing TRT-LLM, venv, Python, ONNX, engine build pipeline._

---

## **11.1 — Missing Build Script**

Expected:

```
examples/llama/build.py
```

But repo lacked it.

### **Root Cause**

You cloned a newer TRT-LLM version where build scripts moved.

### **Fix**

Located correct modern entrypoint:

```
tensorrt_llm/commands/build.py
```

---

## **11.2 — Import Errors When Running Build Script**

```
ModuleNotFoundError: tensorrt_llm.llmapi.kv_cache_type
```

### **Root Cause**

Mismatch between:

- Pip-installed TRT-LLM
- Repo-based TRT-LLM code

They were **different versions** → incompatible.

---

## **11.3 — Critical Issue: Environment Using Python 3.12**

TensorRT-LLM **does not support Python 3.12**.

Supported: 3.8, 3.9, 3.10, 3.11

Your venv showed:

```
Python 3.12.3
```

### **Fix**

Installed Python 3.10:

```bash
apt install python3.10 python3.10-distutils python3.10-venv
```

---

## **11.4 — Rebinding Venv to Python 3.10**

RunPod overrides venv binaries → needed manual fix.

### **Fix**

```bash
rm bentoml/bin/python bentoml/bin/python3
ln -s bentoml/bin/python3.10 bentoml/bin/python
ln -s bentoml/bin/python3.10 bentoml/bin/python3
```

### **Result**

```
Python 3.10.19
```

✔ Compatible with TRT-LLM.

---

## **11.5 — Installing Compatible PyTorch + TRT-LLM**

CUDA 12 PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

TRT-LLM:

```bash
pip install --extra-index-url https://pypi.nvidia.com tensorrt_llm==1.0.0
```

---

## **11.6 — ONNX Version Mismatch**

Error:

```
onnx.helper has no attribute float32_to_bfloat16
```

### **Fix**

```bash
pip install --force-reinstall onnx==1.16.0
```

✔ TRT-LLM imports successfully.

---

## **11.7 — Still Couldn’t Run Repo Build Script**

Reason:

- Repo version ≠ installed version
- Internal imports mismatch

### **Fix — Use Correct Entry Point**

```bash
python3 -m tensorrt_llm.commands.build \
  --checkpoint-dir /workspace/models/llama-8b \
  --output-dir /workspace/trt_engines/llama-8b \
  --dtype float16
```

This uses the **pip-installed, correct version**.

---

## **11.8 — HuggingFace 404 While Downloading**

Error:

```
Repository Not Found
```

### **Root Causes**

- Wrong repo name: `meta-llama/Llama-3-8B`
- Should be:
  `meta-llama/Meta-Llama-3-8B-Instruct`
- Token not set or license not accepted

### **Fix Plan**

Authenticate:

```bash
huggingface-cli login
```

Download with correct repo name.

---

## **11.9 — GPU Utilization Graphing**

We set up:

### **Live Monitor**

```
watch -n 1 nvidia-smi
```

### **Logging**

```
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv -l 1 -f gpu_log.csv
```

### **Plotting**

Using matplotlib → final GPU usage graphs.

---

# # **12. Final Working Stack**

| Component    | Status                                        |
| ------------ | --------------------------------------------- |
| Python       | 3.10.19 ✔                                     |
| TRT-LLM      | 1.0.0 ✔                                       |
| ONNX         | 1.16.0 ✔                                      |
| LMDeploy     | TurboMind backend ✔                           |
| Model        | Llama-3 8B Instruct (HF complete metadata) ✔  |
| Engine Build | Via `python -m tensorrt_llm.commands.build` ✔ |
| Benchmarking | Concurrency saturation + telemetry ✔          |
| GPU Graph    | Saved + committed ✔                           |

Everything now runs end-to-end.

---

# # **13. Conclusion**

This journey covered:

- Deep debugging
- Environment repair
- Python version conflicts
- TRT-LLM internal architecture changes
- LMDeploy server behavior
- HF model handling
- GPU telemetry
- Benchmark methodology
- GitHub workflow
- Production-quality documentation

You now have a **robust, reproducible, professional inference stack**.
