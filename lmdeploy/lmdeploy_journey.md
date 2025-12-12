# First Attempt without Dataset

lmdeploy serve api_server \
 /workspace/models/llama-8b-hf \
 --server-port 8888 \
 --backend pytorch \
 --tp 1 \
 --dtype float16 \
 --device cuda \
 --max-batch-size 256 \
 --cache-max-entry-count 0.92

╔═══════════════════════════════════════════════════════════════╗
║ ENHANCED GPU BENCHMARK v3.0: Multi-Load Analysis ║
║ Comparing 10, 50, 100 Concurrent Users ║
║ Metrics: TTFT, tok/s, MFU (Real GPU Utilization) ║
╚═══════════════════════════════════════════════════════════════╝

✓ GPU Detected: NVIDIA A40
✓ Peak FP16 Tensor FLOPS: 149.7 TFLOPS

╔══════════════════════════════════════════════════════════╗
║ WARMUP PHASE ║
╚══════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 5 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 20/20 successful

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 10 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 50/50 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 10 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 50/50 successful ║
║ Duration: 31.9s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 4783.6 ms | P50: 5069.0 ms ║
║ P95: 5975.1 ms | P99: 5976.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 4783.7 ms | P95: 5975.1 ms ║
║ P99: 5976.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 184.1 tok/s ║
║ Per Request: 24.0 tok/s (avg) ║
║ Per Request: 25.1 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 1.97% ║
║ Power: 242.3W ║
║ Efficiency: 0.76 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 50 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 250/250 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 50 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 250/250 successful ║
║ Duration: 37.1s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 5922.9 ms | P50: 6231.2 ms ║
║ P95: 6835.6 ms | P99: 7856.4 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 5922.9 ms | P95: 6835.6 ms ║
║ P99: 7856.4 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 756.9 tok/s ║
║ Per Request: 18.5 tok/s (avg) ║
║ Per Request: 20.3 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 8.09% ║
║ Power: 250.4W ║
║ Efficiency: 3.02 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 100 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 500/500 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 100 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 500/500 successful ║
║ Duration: 40.1s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 6281.1 ms | P50: 6521.6 ms ║
║ P95: 7321.3 ms | P99: 7918.1 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 6281.2 ms | P95: 7321.4 ms ║
║ P99: 7918.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 1457.8 tok/s ║
║ Per Request: 18.1 tok/s (avg) ║
║ Per Request: 19.2 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 15.58% ║
║ Power: 257.0W ║
║ Efficiency: 5.67 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

✓ GPU telemetry timeline saved: gpu_telemetry_timeline.png
✓ Comprehensive analysis saved: comprehensive_benchmark.png

╔═══════════════════════════════════════════════════════════════╗
║ COMPARATIVE ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ BASELINE: 10 users ║
║ Throughput: 184.1 tok/s ║
║ MFU: 1.97% ║
╟───────────────────────────────────────────────────────────────╢
║ 50 users vs 10 users ║
║ Throughput: 756.9 tok/s (+311.1%) ║
║ MFU: 8.09% (+311.1%) ║
║ TTFT P95: 6836 ms ║
╟───────────────────────────────────────────────────────────────╢
║ 100 users vs 10 users ║
║ Throughput: 1457.8 tok/s (+691.7%) ║
║ MFU: 15.58% (+691.7%) ║
║ TTFT P95: 7321 ms ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║ SATURATION ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ 📊 MATHEMATICAL PREDICTION (Michaelis-Menten Curve Fit) ║
║ Theoretical Max: 14424.3 tok/s ║
║ Half-Saturation (K): 891.1 users ║
║ Fit Quality: Excellent (R² = 0.999) ║
║ Current Status: 10.1% of theoretical max ║
║ ║
║ Predicted 95% Saturation: 16930 users ║
║ Expected Throughput: 13703.1 tok/s ║
║ ║
║ 💡 RECOMMENDATION: System has significant headroom. ║
║ Test up to 16930 users to approach theoretical limit. ║
╟───────────────────────────────────────────────────────────────╢
║ ✓ NO EMPIRICAL SATURATION (< 10% gain threshold) ║
║ All test points show healthy scaling. ║
║ ║
║ 💡 CURRENT BEST EFFICIENCY: 10 users ║
║ (Highest throughput per user: 18.4 tok/s/user) ║
╟───────────────────────────────────────────────────────────────╢
║ MARGINAL GAINS BREAKDOWN: ║
║ 10 → 50 users: +311.1% (+ 573 tok/s) ║
║ 50 → 100 users: + 92.6% (+ 701 tok/s) ║
╚═══════════════════════════════════════════════════════════════╝

KEY INSIGHTS:
• Best MFU: 15.58% at 100 users
• Best Throughput: 1457.8 tok/s at 100 users
• MFU explains TRUE GPU utilization (not kernel occupancy)
• System can likely handle more load - consider testing 150+ users

⚠ Remember: 149.7 TFLOPS is theoretical peak.
Inference is memory-bound. 15-25% MFU is excellent.

# Second Attempt with Dataset -

lmdeploy serve api_server \
 /workspace/models/llama-8b-hf \
 --server-port 8888 \
 --backend pytorch \
 --tp 1 \
 --dtype float16 \
 --device cuda \
 --max-batch-size 256 \
 --cache-max-entry-count 0.92 \
 --session-len 8192

╔═══════════════════════════════════════════════════════════════╗
║ ENHANCED GPU BENCHMARK v4.0: Dataset-Powered Analysis ║
║ Comparing 10, 50, 100 Concurrent Users ║
║ Metrics: TTFT, tok/s, MFU (Real GPU Utilization) ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOADING DATASET: HUGGINGFACE ║
╚══════════════════════════════════════════════════════════╝
→ Loading HF dataset: tatsu-lab/alpaca
→ Split: train
→ Prompt column: instruction
✓ Loaded 52002 prompts from HuggingFace
✓ Shuffled prompts
✓ Limited to 500 samples
✓ Total prompts available: 500

╔══════════════════════════════════════════════════════════╗
║ DATASET STATISTICS ║
╠══════════════════════════════════════════════════════════╣
║ Total Prompts: 500 ║
║ Avg Length: 61 chars (10 words) ║
║ Median Length: 57 chars (9 words) ║
║ Range: 21 - 193 chars ║
╚══════════════════════════════════════════════════════════╝

✓ GPU Detected: NVIDIA A40
✓ Peak FP16 Tensor FLOPS: 149.7 TFLOPS

╔══════════════════════════════════════════════════════════╗
║ WARMUP PHASE ║
╚══════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 5 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 20/20 successful

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 10 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 50/50 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 10 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 50/50 successful ║
║ Duration: 27.7s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 3725.0 ms | P50: 4609.8 ms ║
║ P95: 5211.3 ms | P99: 5259.3 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 3725.0 ms | P95: 5211.3 ms ║
║ P99: 5259.3 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 159.5 tok/s ║
║ Per Request: 21.8 tok/s (avg) ║
║ Per Request: 24.6 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 1.71% ║
║ Power: 247.0W ║
║ Efficiency: 0.65 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 50 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 250/250 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 50 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 250/250 successful ║
║ Duration: 36.5s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 5412.1 ms | P50: 5759.1 ms ║
║ P95: 8021.5 ms | P99: 8304.4 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 5412.1 ms | P95: 8021.5 ms ║
║ P99: 8304.4 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 631.3 tok/s ║
║ Per Request: 16.1 tok/s (avg) ║
║ Per Request: 18.9 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 6.75% ║
║ Power: 251.6W ║
║ Efficiency: 2.51 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 100 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 500/500 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 100 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 500/500 successful ║
║ Duration: 39.5s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 5617.5 ms | P50: 6591.6 ms ║
║ P95: 7650.9 ms | P99: 7786.7 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 5617.5 ms | P95: 7650.9 ms ║
║ P99: 7786.7 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 1133.8 tok/s ║
║ Per Request: 14.5 tok/s (avg) ║
║ Per Request: 16.9 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 12.12% ║
║ Power: 256.8W ║
║ Efficiency: 4.41 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

✓ GPU telemetry timeline saved: gpu_telemetry_timeline.png
✓ Comprehensive analysis saved: comprehensive_benchmark.png

╔═══════════════════════════════════════════════════════════════╗
║ COMPARATIVE ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ BASELINE: 10 users ║
║ Throughput: 159.5 tok/s ║
║ MFU: 1.71% ║
╟───────────────────────────────────────────────────────────────╢
║ 50 users vs 10 users ║
║ Throughput: 631.3 tok/s (+295.7%) ║
║ MFU: 6.75% (+295.7%) ║
║ TTFT P95: 8021 ms ║
╟───────────────────────────────────────────────────────────────╢
║ 100 users vs 10 users ║
║ Throughput: 1133.8 tok/s (+610.7%) ║
║ MFU: 12.12% (+610.7%) ║
║ TTFT P95: 7651 ms ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║ SATURATION ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ 📊 MATHEMATICAL PREDICTION (Michaelis-Menten Curve Fit) ║
║ Theoretical Max: 4944.5 tok/s ║
║ Half-Saturation (K): 336.9 users ║
║ Fit Quality: Excellent (R² = 0.999) ║
║ Current Status: 22.9% of theoretical max ║
║ ║
║ Predicted 95% Saturation: 6401 users ║
║ Expected Throughput: 4697.2 tok/s ║
║ ║
║ 💡 RECOMMENDATION: System has significant headroom. ║
║ Test up to 6401 users to approach theoretical limit. ║
╟───────────────────────────────────────────────────────────────╢
║ ✓ NO EMPIRICAL SATURATION (< 10% gain threshold) ║
║ All test points show healthy scaling. ║
║ ║
║ 💡 CURRENT BEST EFFICIENCY: 10 users ║
║ (Highest throughput per user: 16.0 tok/s/user) ║
╟───────────────────────────────────────────────────────────────╢
║ MARGINAL GAINS BREAKDOWN: ║
║ 10 → 50 users: +295.7% (+ 472 tok/s) ║
║ 50 → 100 users: + 79.6% (+ 502 tok/s) ║
╚═══════════════════════════════════════════════════════════════╝

KEY INSIGHTS:
• Best MFU: 12.12% at 100 users
• Best Throughput: 1133.8 tok/s at 100 users
• MFU explains TRUE GPU utilization (not kernel occupancy)
• System can likely handle more load - consider testing 150+ users

⚠ Remember: 149.7 TFLOPS is theoretical peak.
Inference is memory-bound. 15-25% MFU is excellent.

✓ Benchmark complete!
✓ Dataset used: huggingface
✓ HF Dataset: tatsu-lab/alpaca

# Third Try with Dataset without external config

lmdeploy serve api_server \
 /workspace/models/llama-8b-hf \
 --server-port 8888 \
 --backend pytorch \
 --tp 1 \
 --dtype float16 \
 --device cuda \
 --max-batch-size 256

╔═══════════════════════════════════════════════════════════════╗
║ ENHANCED GPU BENCHMARK v4.0: Dataset-Powered Analysis ║
║ Comparing 10, 50, 100 Concurrent Users ║
║ Metrics: TTFT, tok/s, MFU (Real GPU Utilization) ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOADING DATASET: HUGGINGFACE ║
╚══════════════════════════════════════════════════════════╝
→ Loading HF dataset: tatsu-lab/alpaca
→ Split: train
→ Prompt column: instruction
✓ Loaded 52002 prompts from HuggingFace
✓ Shuffled prompts
✓ Limited to 500 samples
✓ Total prompts available: 500

╔══════════════════════════════════════════════════════════╗
║ DATASET STATISTICS ║
╠══════════════════════════════════════════════════════════╣
║ Total Prompts: 500 ║
║ Avg Length: 58 chars (10 words) ║
║ Median Length: 55 chars (9 words) ║
║ Range: 21 - 220 chars ║
╚══════════════════════════════════════════════════════════╝

✓ GPU Detected: NVIDIA A40
✓ Peak FP16 Tensor FLOPS: 149.7 TFLOPS

╔══════════════════════════════════════════════════════════╗
║ WARMUP PHASE ║
╚══════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 5 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 20/20 successful

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 10 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 50/50 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 10 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 50/50 successful ║
║ Duration: 29.0s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 3795.2 ms | P50: 4795.4 ms ║
║ P95: 5269.7 ms | P99: 5376.9 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 3795.2 ms | P95: 5269.7 ms ║
║ P99: 5377.0 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 154.9 tok/s ║
║ Per Request: 22.0 tok/s (avg) ║
║ Per Request: 24.4 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 1.66% ║
║ Power: 247.5W ║
║ Efficiency: 0.63 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 50 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 250/250 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 50 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 250/250 successful ║
║ Duration: 33.6s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 4642.9 ms | P50: 5281.0 ms ║
║ P95: 6035.7 ms | P99: 6119.5 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 4643.0 ms | P95: 6035.7 ms ║
║ P99: 6119.5 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 660.7 tok/s ║
║ Per Request: 17.8 tok/s (avg) ║
║ Per Request: 21.2 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 7.06% ║
║ Power: 254.9W ║
║ Efficiency: 2.59 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 100 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 500/500 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 100 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 500/500 successful ║
║ Duration: 40.0s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 5738.1 ms | P50: 6673.6 ms ║
║ P95: 7537.6 ms | P99: 7656.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 5738.1 ms | P95: 7537.6 ms ║
║ P99: 7656.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 1160.4 tok/s ║
║ Per Request: 14.9 tok/s (avg) ║
║ Per Request: 17.2 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 12.40% ║
║ Power: 260.1W ║
║ Efficiency: 4.46 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

✓ GPU telemetry timeline saved: gpu_telemetry_timeline.png
✓ Comprehensive analysis saved: comprehensive_benchmark.png

╔═══════════════════════════════════════════════════════════════╗
║ COMPARATIVE ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ BASELINE: 10 users ║
║ Throughput: 154.9 tok/s ║
║ MFU: 1.66% ║
╟───────────────────────────────────────────────────────────────╢
║ 50 users vs 10 users ║
║ Throughput: 660.7 tok/s (+326.6%) ║
║ MFU: 7.06% (+326.6%) ║
║ TTFT P95: 6036 ms ║
╟───────────────────────────────────────────────────────────────╢
║ 100 users vs 10 users ║
║ Throughput: 1160.4 tok/s (+649.3%) ║
║ MFU: 12.40% (+649.3%) ║
║ TTFT P95: 7538 ms ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║ SATURATION ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ 📊 MATHEMATICAL PREDICTION (Michaelis-Menten Curve Fit) ║
║ Theoretical Max: 4620.9 tok/s ║
║ Half-Saturation (K): 298.5 users ║
║ Fit Quality: Excellent (R² = 1.000) ║
║ Current Status: 25.1% of theoretical max ║
║ ║
║ Predicted 95% Saturation: 5670 users ║
║ Expected Throughput: 4389.8 tok/s ║
║ ║
║ 💡 RECOMMENDATION: System has significant headroom. ║
║ Test up to 5670 users to approach theoretical limit. ║
╟───────────────────────────────────────────────────────────────╢
║ ✓ NO EMPIRICAL SATURATION (< 10% gain threshold) ║
║ All test points show healthy scaling. ║
║ ║
║ 💡 CURRENT BEST EFFICIENCY: 10 users ║
║ (Highest throughput per user: 15.5 tok/s/user) ║
╟───────────────────────────────────────────────────────────────╢
║ MARGINAL GAINS BREAKDOWN: ║
║ 10 → 50 users: +326.6% (+ 506 tok/s) ║
║ 50 → 100 users: + 75.6% (+ 500 tok/s) ║
╟───────────────────────────────────────────────────────────────╢
║ ⚠ DIMINISHING RETURNS DETECTED: ║
║ At 100 users: 10.0 tok/s per additional user ║
║ (Below 10.0 tok/s threshold) ║
╚═══════════════════════════════════════════════════════════════╝

KEY INSIGHTS:
• Best MFU: 12.40% at 100 users
• Best Throughput: 1160.4 tok/s at 100 users
• MFU explains TRUE GPU utilization (not kernel occupancy)
• System can likely handle more load - consider testing 150+ users

⚠ Remember: 149.7 TFLOPS is theoretical peak.
Inference is memory-bound. 15-25% MFU is excellent.

✓ Benchmark complete!
✓ Dataset used: huggingface
✓ HF Dataset: tatsu-lab/alpaca
