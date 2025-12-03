# ============================================================================
# 5. INSTALL TensorRT-LLM (RunPod Optimized)
# ============================================================================
echo ""
echo "🚀 Setting up TensorRT-LLM on RunPod..."

# [LOGIC] Check for System PyTorch first
echo "  🔍 Verifying System PyTorch..."
if python3 -c "import torch; print(f'  ✓ Found PyTorch {torch.__version__} with CUDA {torch.version.cuda}')" 2>/dev/null; then
    echo "  ✓ System environment confirmed."
else
    echo "  ❌ CRITICAL: PyTorch not found in system packages."
    echo "     RunPod images usually come with this pre-installed."
    echo "     Attempting to install, but this may break CUDA linking..."
    pip install torch --quiet
fi

echo "  Installing TensorRT-LLM $TENSORRT_LLM_VERSION..."
# [LOGIC] Install TRT-LLM using the extra index for pre-built wheels
pip install "tensorrt-llm==$TENSORRT_LLM_VERSION" --extra-index-url https://pypi.nvidia.com --quiet

echo "  Cloning TensorRT-LLM repository..."
if [ ! -d "TensorRT-LLM" ]; then
    git clone https://github.com/NVIDIA/TensorRT-LLM.git --depth=1 --quiet
fi

cd TensorRT-LLM

# [LOGIC] Strip 'torch' from requirements to prevent downgrading the System PyTorch
echo "  Sanitizing requirements.txt (Preserving System Torch)..."
grep -v "torch" requirements.txt > requirements_safe.txt
pip install -r requirements_safe.txt --quiet 2>/dev/null || true

cd ..

echo "  Converting checkpoint to TensorRT format..."
echo "  (This takes ~40 seconds...)"
python3 TensorRT-LLM/examples/llama/convert_checkpoint.py \
    --model_dir "$MODELS_DIR/llama-8b" \
    --output_dir "./trt_checkpoints/llama-8b" \
    --dtype bfloat16 \
    --tp 1

echo "  Building TensorRT engine..."
echo "  (This takes ~24 seconds and requires ~34GB RAM...)"
trtllm-build \
    --checkpoint_dir "./trt_checkpoints/llama-8b" \
    --output_dir "./trt_engines/llama-8b" \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --max_batch_size 256 \
    --max_input_len 1024 \
    --max_output_len 512 \
    --workers 1

echo "  ✓ TensorRT-LLM ready"

# ============================================================================
# 6. CREATE STARTUP SCRIPTS (RunPod Optimized)
# ============================================================================
echo ""
echo "📝 Creating startup scripts..."

# LMDeploy startup script
# [LOGIC] Removed 'source venv' to ensure we use Global/System Torch
cat > "start_lmdeploy.sh" << 'EOF'
#!/bin/bash
lmdeploy serve api_server \
    ./models/llama-8b-turbomind-fp16 \
    --server-name 0.0.0.0 \
    --server-port 8000 \
    --tp 1 \
    --log-level INFO
EOF
chmod +x start_lmdeploy.sh

# TensorRT-LLM startup script
# [LOGIC] Removed 'source venv' to ensure we use Global/System Torch
cat > "start_tensorrt.sh" << 'EOF'
#!/bin/bash
python3 ./TensorRT-LLM/examples/openai_api_server.py \
    --engine_dir ./trt_engines/llama-8b \
    --tokenizer_dir ./models/llama-8b \
    --host 0.0.0.0 \
    --port 8001
EOF
chmod +x start_tensorrt.sh

echo "  ✓ Startup scripts created"

# ============================================================================
# 7. COPY BENCHMARK SCRIPTS
# ============================================================================
echo ""
echo "📋 Preparing benchmark scripts..."

if [ -f "../quick_benchmark.py" ]; then
    cp ../quick_benchmark.py .
fi

# Create main benchmark runner
cat > "run_benchmark.sh" << 'EOF'
#!/bin/bash
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Starting Benchmark Suite                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Start servers
echo ""
echo "Starting backends..."
echo ""

# Terminal 1: LMDeploy
echo "Terminal 1 - LMDeploy:"
./start_lmdeploy.sh > lmdeploy.log 2>&1 &
LMDEPLOY_PID=$!
echo "  PID: $LMDEPLOY_PID"

# Wait for server to start (RunPod storage can be slow, giving 15s)
sleep 15

# Terminal 2: TensorRT-LLM
echo ""
echo "Terminal 2 - TensorRT-LLM:"
./start_tensorrt.sh > tensorrt.log 2>&1 &
TENSORRT_PID=$!
echo "  PID: $TENSORRT_PID"

# Wait for server to start
sleep 15

# Terminal 3: Run benchmark
echo ""
echo "Terminal 3 - Running benchmark..."
echo ""
# Ensure we use the python that has dependencies installed
python3 quick_benchmark.py

# Cleanup
echo ""
echo "Shutting down servers..."
kill $LMDEPLOY_PID $TENSORRT_PID 2>/dev/null || true
EOF
chmod +x run_benchmark.sh

echo "  ✓ Benchmark scripts prepared"

# ============================================================================
# 8. SUMMARY
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              ✓ Setup Complete (RunPod Edition)                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📍 Workspace Location: $(pwd)"
echo "⚠️  NOTE: RunPod utilizes global PyTorch. Virtual Envs were skipped."
echo "     Log files will be generated as lmdeploy.log and tensorrt.log"