# accel-sim 🚀

**accel-sim** is a high-performance, CPU-based estimation tool designed to predict the latency and memory footprint of Transformer-based workloads on hardware accelerators (GPUs, TPUs). 

By hooking into the PyTorch computation graph, `accel-sim` applies an analytical **Roofline Model** to estimate performance without requiring access to the physical hardware.

---

## 💡 Why This Exists & How It Helps

As a Python developer who’s spent the better part of a decade (8 years and counting) wrestling with deep learning deployments, I’ve hit the same wall more times than I care to admit: **The GPU Scarcity Wall.**

We’ve all been there:
- Waiting in a SLURM queue for 4 hours just to find out your batch size is 2MB too large for a V100.
- Paying $4/hour for an A100 instance only to realize your model is actually IO-bound and would run just as fast on a much cheaper T4.
- Wondering if that new "optimization" actually makes things faster or if it just moves the bottleneck somewhere else.

`accel-sim` was created to solve these problems **locally, on your CPU**. It allows you to simulate the performance characteristics of high-end hardware without actually owning or renting it. Think of it as a "digital twin" for your model's execution.

### The Background Story
I sat down to build this project out of a mix of curiosity and pure boredom. I was tired of the "trial and error" approach to hardware selection. I wanted a way to look at a PyTorch graph and mathematically determine where the time was going. By leveraging `torch.fx`, I realized I could trace the operations, apply a Roofline model (FLOPs/Bandwidth), and get a surprisingly accurate estimation of execution time and memory usage—all while my laptop fan stays silent.

---

## 🎯 Use Case: The "Developer's Dilemma"

Imagine you are a Python developer tasked with deploying a new Transformer-based LLM. You have three choices for your production cluster: V100, A100, or the latest H100.

**The Old Way:**
1. Provision one of each instance.
2. Port your code.
3. Run benchmarks.
4. Burn $100+ in cloud credits before you've even started.

**The `accel-sim` Way:**
1. Run `accel-sim compare my_model.py --devices v100,a100,h100` on your laptop.
2. See a detailed breakdown:
    - *V100:* Memory Bandwidth Bound (70% utilization).
    - *A100:* Compute Bound (90% utilization).
    - *H100:* Overkill (20% utilization).
3. **Decision:** You go with the A100 because it’s the sweet spot for your specific sequence length, saving your company thousands in unnecessary hardware costs.


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

## 🚀 Deep Dive: Usage & Workflow

`accel-sim` is designed to be both a developer's local companion and a CI/CD gatekeeper. To get 100% out of the tool, you need to understand how it captures your model and how it calculates its "costs."

### 1. Preparing Your Model Script
The simulator doesn't just run a `.pth` file; it needs to trace the computation graph. To do this, it expects a Python script that exposes a specific entry point: `get_model()`.

**Example: `transformer_block.py`**
```python
import torch
import torch.nn as nn

class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(512, 2048)
        self.lin2 = nn.Linear(2048, 512)
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        return self.norm(x + self.lin2(torch.relu(self.lin1(x))))

def get_model():
    model = MyBlock()
    # Define example inputs (Batch size, Seq length, Hidden dim)
    example_inputs = (torch.randn(32, 128, 512),)
    return model, example_inputs
```

### 2. Command Line Interface (CLI)

#### `accel-sim simulate`
This is your bread and butter. It runs a full trace and generates a bottleneck report.
```bash
accel-sim simulate path/to/script.py --device a100
```
- **What happens behind the scenes:** The tool loads your script, calls `get_model()`, and then passes the model to `torch.fx.experimental.proxy_tensor.make_fx`. This captures the model at the **Aten level** (the low-level ops that actually run on the GPU).
- **The Output:** You'll see a line-by-line breakdown of latency. If a layer is particularly slow, it will show as a "Top Bottleneck."

#### `accel-sim compare`
Useful for capacity planning and budget management.
```bash
accel-sim compare path/to/script.py --devices v100,a100,h100,tpu-v4
```
- **Hardware Selection:** This command helps you see if your model hits diminishing returns. For example, if your sequence length is small, an H100 might be no faster than an A100 because the latency is dominated by memory overhead rather than raw compute.

#### `accel-sim devices`
View the ground-truth specifications used by the engine.
```bash
accel-sim devices
```
Each device profile includes:
- **Compute TFLOPS:** The theoretical peak for float16/bfloat16.
- **Memory Bandwidth (GB/s):** The speed at which data moves from HBM to the cores.
- **Memory Limit (GB):** The hard cap for Out-of-Memory (OOM) checks.

### 3. Understanding the Estimation Engine

How can a CPU script predict GPU performance? We use a **Roofline Model**, a standard architectural analysis tool.

#### The Math
For every operation captured in the graph, we calculate:
1. **Compute Time:** `Total_FLOPs / Device_Peak_TFLOPS`
2. **Memory Time:** `Total_Bytes_Accessed / Device_Peak_Bandwidth`
3. **Estimated Latency:** `max(Compute_Time, Memory_Time)`

This logic assumes that compute and memory access can overlap perfectly (which is true for well-optimized kernels like Matmul). If `Memory Time > Compute Time`, the operation is **Memory-Bound**. If `Compute Time > Memory Time`, it is **Compute-Bound**.

#### Memory Tracking
`accel-sim` doesn't just look at total model size. It simulates the **execution walk**:
- It tracks the size of input/output tensors for every node.
- It calculates the "Peak Live Memory"—the maximum memory used by activations and weights at any single point in time.
- This allows it to predict OOMs that only happen halfway through a forward pass.

### 4. Advanced: Integration & Automation

#### CI/CD Regressions
Add `accel-sim` to your PR pipeline. If a developer accidentally adds a layer that increases latency by 20% on an A100, the CI can fail before you ever waste a single cent on an actual GPU benchmark.

#### Bottleneck Hunting
The output report highlights exactly which op type is the culprit.
- **High Matmul Latency?** Your hidden dimension is likely too large for the device's cores.
- **High Softmax/Elementwise Latency?** You are likely memory-bound. Consider operator fusion or quantization.

### 5. Customizing Device Profiles
If you are working with proprietary hardware or a new cloud instance type, you can define your own profiles. Currently, these are stored in `accel_sim/devices/profiles.py`, but you can also pass them via the API.

```python
from accel_sim.devices.profiles import DeviceProfile

# Define a theoretical "AI Chip 2027"
my_chip = DeviceProfile(
    name="future_chip",
    compute_tflops=1000.0,    # 1 PetaFLOP
    memory_bandwidth=5000.0, # 5 TB/s
    memory_limit_gb=128.0
)
```

### 6. Interpreting the Report
When you run a simulation, the report provides three critical numbers:

1. **Total Latency (ms):** The sum of all non-overlapping execution times. Note that this is an *optimistic* lower bound. Real-world kernel overhead (CUDA launch times, etc.) typically adds 5-10%.
2. **Peak Memory (GB):** The absolute maximum memory footprint. If this exceeds the device limit, the status will flip to **OOM**. This includes both static weights and dynamic activations.
3. **Bottleneck Percentage:** Calculated as `Layer_Time / Total_Time`. Focus your optimization efforts on layers that take up >10% of the total execution time.

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

## 👨‍💻 About the Author

I am an **8-year seasoned Python developer** and hardware enthusiast. I created `accel-sim` during a streak of boredom as a way to bridge the gap between high-level PyTorch code and low-level hardware performance. When I'm not tracing computation graphs, I'm usually building CLI tools or optimizing things that probably don't need to be optimized.

## 📜 License
MIT
