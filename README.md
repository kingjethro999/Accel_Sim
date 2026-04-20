# accel-sim 🚀

**accel-sim** is a high-performance, CPU-based estimation tool designed to predict the latency and memory footprint of Transformer-based workloads on hardware accelerators (GPUs, TPUs). 

By hooking into the PyTorch computation graph, `accel-sim` applies an analytical **Roofline Model** to estimate performance without requiring access to the physical hardware.

## 🌟 Real-World Scenarios

### 1. Cost-Benefit Analysis for Cloud Renting
Before spinning up an expensive A100 or H100 cluster, use `accel-sim` to determine if your model actually benefits from the extra TFLOPS. If your workload is memory-bandwidth bound, a cheaper V100 might offer nearly identical performance.

### 2. OOM (Out-of-Memory) Prediction
Predict exactly at what sequence length or batch size your model will crash on a 16GB V100 vs. a 40GB A100. This is critical for production deployment planning and avoiding runtime failures.

### 3. CI/CD Performance Guardrails
Integrate `accel-sim` into your GitHub Actions or Jenkins pipelines. Since it runs entirely on CPU, you can catch performance regressions (e.g., a change that accidentally doubles the FLOPs of your attention layer) before they are merged, without needing GPU runners.

### 4. Hardware-Aware Model Design
Identify which specific ops (e.g., Softmax vs. Matmul) are your primary bottlenecks. This helps you decide where to apply optimizations like operator fusion, quantization, or specialized kernels like FlashAttention.

### 5. Custom Hardware Evaluation
Designing a new AI chip? Define a custom `DeviceProfile` with your theoretical TFLOPS and memory bandwidth to see how state-of-the-art models like Llama-3 or GPT-4 would perform on your architecture.

---

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/accel-sim.git
cd accel-sim

# Create a virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## 🚀 Usage

### 1. Simulate a Model
Estimate performance for a specific model script and device:
```bash
accel-sim simulate examples/gpt_block.py --device v100
```

### 2. Compare Across Devices
Benchmark your model across multiple hardware generations:
```bash
accel-sim compare examples/gpt_block.py --devices v100,a100,h100
```

### 3. List Device Profiles
See the specifications used for estimation:
```bash
accel-sim devices
```

---

## 🏗 Architecture

1.  **Capture**: Uses `torch.fx` and `make_fx` to extract a detailed Aten-level computation graph.
2.  **Cost Model**: Applies a Roofline Model: `time = max(flops / peak_tflops, bytes / peak_bandwidth)`.
3.  **Simulator**: Walks the graph sequentially, tracking live activation memory and cumulative latency.
4.  **Reporter**: Generates a detailed breakdown of bottlenecks and OOM warnings.

## 📊 Example Output
```text
Device: V100 (125.0 TFLOPS, 900.0 GB/s, 16.0 GB)

Total latency:   42.30 ms
Peak memory:      9.10 GB  ✓ (fits in device memory)

Top bottlenecks:
  1. attn_qk_matmul       27.60 ms  (65.2%)
  2. mlp_fc1_matmul        8.50 ms  (20.1%)
  3. attn_softmax          3.20 ms   (7.6%)
```

## 📜 License
MIT
