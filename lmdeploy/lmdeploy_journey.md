# First Attempt without Dataset

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
