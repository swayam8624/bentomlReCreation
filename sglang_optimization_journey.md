# First Try

python3 -m sglang.launch_server \
 --model-path /workspace/models/llama-8b-hf \
 --host 0.0.0.0 \
 --port 8888 \
 --mem-fraction-static 0.95 \
 --context-length 8192 \
 --max-running-requests 256 \
 --schedule-policy lpm \
 --chunked-prefill-size 8192 \
 --max-prefill-tokens 16384 \
 --attention-backend flashinfer

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
║ Duration: 25.3s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 3525.5 ms | P50: 4072.0 ms ║
║ P95: 4218.8 ms | P99: 4241.5 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 3525.6 ms | P95: 4218.8 ms ║
║ P99: 4241.6 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 216.1 tok/s ║
║ Per Request: 30.8 tok/s (avg) ║
║ Per Request: 30.8 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 2.31% ║
║ Power: 246.7W ║
║ Efficiency: 0.88 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 50 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 250/250 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 50 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 250/250 successful ║
║ Duration: 30.0s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 4529.5 ms | P50: 4871.8 ms ║
║ P95: 5341.7 ms | P99: 5425.9 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 4529.5 ms | P95: 5341.8 ms ║
║ P99: 5425.9 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 966.8 tok/s ║
║ Per Request: 25.4 tok/s (avg) ║
║ Per Request: 25.3 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 10.33% ║
║ Power: 251.0W ║
║ Efficiency: 3.85 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 100 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 500/500 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 100 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 500/500 successful ║
║ Duration: 32.9s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 5164.7 ms | P50: 5509.0 ms ║
║ P95: 6364.4 ms | P99: 6538.6 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 5164.7 ms | P95: 6364.4 ms ║
║ P99: 6538.6 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 1762.8 tok/s ║
║ Per Request: 22.3 tok/s (avg) ║
║ Per Request: 22.4 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 18.84% ║
║ Power: 255.7W ║
║ Efficiency: 6.89 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

✓ GPU telemetry timeline saved: gpu_telemetry_timeline.png
✓ Comprehensive analysis saved: comprehensive_benchmark.png

╔═══════════════════════════════════════════════════════════════╗
║ COMPARATIVE ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ BASELINE: 10 users ║
║ Throughput: 216.1 tok/s ║
║ MFU: 2.31% ║
╟───────────────────────────────────────────────────────────────╢
║ 50 users vs 10 users ║
║ Throughput: 966.8 tok/s (+347.3%) ║
║ MFU: 10.33% (+347.3%) ║
║ TTFT P95: 5342 ms ║
╟───────────────────────────────────────────────────────────────╢
║ 100 users vs 10 users ║
║ Throughput: 1762.8 tok/s (+715.6%) ║
║ MFU: 18.84% (+715.6%) ║
║ TTFT P95: 6364 ms ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║ SATURATION ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ 📊 MATHEMATICAL PREDICTION (Michaelis-Menten Curve Fit) ║
║ Theoretical Max: 9692.5 tok/s ║
║ Half-Saturation (K): 450.0 users ║
║ Fit Quality: Excellent (R² = 1.000) ║
║ Current Status: 18.2% of theoretical max ║
║ ║
║ Predicted 95% Saturation: 8550 users ║
║ Expected Throughput: 9207.8 tok/s ║
║ ║
║ 💡 RECOMMENDATION: System has significant headroom. ║
║ Test up to 8550 users to approach theoretical limit. ║
╟───────────────────────────────────────────────────────────────╢
║ ✓ NO EMPIRICAL SATURATION (< 10% gain threshold) ║
║ All test points show healthy scaling. ║
║ ║
║ 💡 CURRENT BEST EFFICIENCY: 10 users ║
║ (Highest throughput per user: 21.6 tok/s/user) ║
╟───────────────────────────────────────────────────────────────╢
║ MARGINAL GAINS BREAKDOWN: ║
║ 10 → 50 users: +347.3% (+ 751 tok/s) ║
║ 50 → 100 users: + 82.3% (+ 796 tok/s) ║
╚═══════════════════════════════════════════════════════════════╝

KEY INSIGHTS:
• Best MFU: 18.84% at 100 users
• Best Throughput: 1762.8 tok/s at 100 users
• MFU explains TRUE GPU utilization (not kernel occupancy)
• System can likely handle more load - consider testing 150+ users

⚠ Remember: 149.7 TFLOPS is theoretical peak.
Inference is memory-bound. 15-25% MFU is excellent.

# Second Try

python3 -m sglang.launch_server \
 --model-path /workspace/models/llama-8b-hf \
 --host 0.0.0.0 \
 --port 8888 \
 --mem-fraction-static 0.90 \
 --context-length 8192 \
 --max-running-requests 256 \
 --schedule-policy fcfs \
 --attention-backend flashinfer

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
║ Duration: 25.8s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 3659.0 ms | P50: 4059.4 ms ║
║ P95: 4176.7 ms | P99: 4212.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 3659.0 ms | P95: 4176.7 ms ║
║ P99: 4212.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 221.8 tok/s ║
║ Per Request: 31.1 tok/s (avg) ║
║ Per Request: 31.1 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 2.37% ║
║ Power: 245.9W ║
║ Efficiency: 0.90 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 50 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 250/250 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 50 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 250/250 successful ║
║ Duration: 30.1s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 4479.0 ms | P50: 4844.5 ms ║
║ P95: 5500.5 ms | P99: 5527.4 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 4479.1 ms | P95: 5500.5 ms ║
║ P99: 5527.4 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 943.5 tok/s ║
║ Per Request: 25.1 tok/s (avg) ║
║ Per Request: 25.2 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 10.08% ║
║ Power: 250.6W ║
║ Efficiency: 3.76 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║ LOAD TEST: 100 Concurrent Users  
╚══════════════════════════════════════════════════════════╝
✓ Complete: 500/500 successful

╔═══════════════════════════════════════════════════════════════╗
║ RESULTS: 100 Concurrent Users ║
╠═══════════════════════════════════════════════════════════════╣
║ Requests: 500/500 successful ║
║ Duration: 34.4s ║
╟───────────────────────────────────────────────────────────────╢
║ TIME TO FIRST TOKEN (TTFT) ║
║ Avg: 5218.2 ms | P50: 5736.0 ms ║
║ P95: 6766.9 ms | P99: 6941.1 ms ║
╟───────────────────────────────────────────────────────────────╢
║ TOTAL LATENCY ║
║ Avg: 5218.2 ms | P95: 6766.9 ms ║
║ P99: 6941.2 ms ║
╟───────────────────────────────────────────────────────────────╢
║ THROUGHPUT ║
║ Aggregate: 1621.7 tok/s ║
║ Per Request: 21.3 tok/s (avg) ║
║ Per Request: 21.1 tok/s (p50) ║
╟───────────────────────────────────────────────────────────────╢
║ EFFICIENCY (Real GPU Utilization) ║
║ MFU: 17.33% ║
║ Power: 255.8W ║
║ Efficiency: 6.34 tok/W ║
╚═══════════════════════════════════════════════════════════════╝

✓ GPU telemetry timeline saved: gpu_telemetry_timeline.png
✓ Comprehensive analysis saved: comprehensive_benchmark.png

╔═══════════════════════════════════════════════════════════════╗
║ COMPARATIVE ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ BASELINE: 10 users ║
║ Throughput: 221.8 tok/s ║
║ MFU: 2.37% ║
╟───────────────────────────────────────────────────────────────╢
║ 50 users vs 10 users ║
║ Throughput: 943.5 tok/s (+325.4%) ║
║ MFU: 10.08% (+325.4%) ║
║ TTFT P95: 5500 ms ║
╟───────────────────────────────────────────────────────────────╢
║ 100 users vs 10 users ║
║ Throughput: 1621.7 tok/s (+631.2%) ║
║ MFU: 17.33% (+631.2%) ║
║ TTFT P95: 6767 ms ║
╚═══════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════╗
║ SATURATION ANALYSIS ║
╠═══════════════════════════════════════════════════════════════╣
║ 📊 MATHEMATICAL PREDICTION (Michaelis-Menten Curve Fit) ║
║ Theoretical Max: 5688.0 tok/s ║
║ Half-Saturation (K): 250.9 users ║
║ Fit Quality: Excellent (R² = 1.000) ║
║ Current Status: 28.5% of theoretical max ║
║ ║
║ Predicted 95% Saturation: 4766 users ║
║ Expected Throughput: 5403.6 tok/s ║
║ ║
║ 💡 RECOMMENDATION: System has significant headroom. ║
║ Test up to 4766 users to approach theoretical limit. ║
╟───────────────────────────────────────────────────────────────╢
║ ✓ NO EMPIRICAL SATURATION (< 10% gain threshold) ║
║ All test points show healthy scaling. ║
║ ║
║ 💡 CURRENT BEST EFFICIENCY: 10 users ║
║ (Highest throughput per user: 22.2 tok/s/user) ║
╟───────────────────────────────────────────────────────────────╢
║ MARGINAL GAINS BREAKDOWN: ║
║ 10 → 50 users: +325.4% (+ 722 tok/s) ║
║ 50 → 100 users: + 71.9% (+ 678 tok/s) ║
╚═══════════════════════════════════════════════════════════════╝

KEY INSIGHTS:
• Best MFU: 17.33% at 100 users
• Best Throughput: 1621.7 tok/s at 100 users
• MFU explains TRUE GPU utilization (not kernel occupancy)
• System can likely handle more load - consider testing 150+ users

⚠ Remember: 149.7 TFLOPS is theoretical peak.
Inference is memory-bound. 15-25% MFU is excellent.
