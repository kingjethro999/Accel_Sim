# CLAUDE.md — CPU Simulator for Accelerator Workloads

## Project Overview

A CPU-based performance estimation tool that predicts how transformer workloads (attention, MLP layers, etc.) would behave on hardware accelerators — without needing access to one. The tool hooks into PyTorch, extracts a computation graph, runs it through an analytical cost model, and outputs latency and memory estimates for a target device profile.

This is **not** a cycle-accurate hardware simulator. It is a directionally correct estimation layer that runs on CPU and produces actionable numbers.

---

## Goals

- Estimate latency and peak memory of transformer workloads on virtual accelerator profiles (V100, A100, TPU-like configs)
- Identify per-op bottlenecks as a percentage of total compute
- Be usable as a CLI tool and as a Python library
- Support PyTorch models out of the box via `torch.fx` tracing
- Keep the cost model transparent and easy to extend

---

## Repository Layout

```
accel-sim/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── accel_sim/
│   ├── __init__.py
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── tracer.py          # torch.fx tracing + shape extraction
│   │   └── ir.py              # Graph / Node IR definitions
│   ├── cost_model/
│   │   ├── __init__.py
│   │   ├── ops.py             # Per-op cost functions (matmul, softmax, etc.)
│   │   └── memory.py          # Activation + buffer memory estimates
│   ├── devices/
│   │   ├── __init__.py
│   │   └── profiles.py        # Device spec dataclasses (V100, A100, ...)
│   ├── engine/
│   │   ├── __init__.py
│   │   └── simulator.py       # Graph walker + accumulation logic
│   ├── output/
│   │   ├── __init__.py
│   │   └── report.py          # SimResult dataclass + formatting
│   └── cli.py                 # Entry point: `accel-sim simulate`
├── tests/
│   ├── test_tracer.py
│   ├── test_cost_model.py
│   ├── test_simulator.py
│   └── fixtures/
│       └── tiny_transformer.py
└── examples/
    ├── gpt_block.py
    └── latency_vs_seqlen.py
```

---

## Architecture

### 1. Frontend — Model Capture (`accel_sim/capture/`)

**Entry point:** `tracer.py`

Use `torch.fx.symbolic_trace` to trace the model. Walk the resulting `fx.Graph` and extract each node into the project's own IR.

Key responsibilities:
- Accept a `nn.Module` and example inputs
- Call `torch.fx.symbolic_trace(model)`
- Iterate `graph.nodes` and map each to an `IRNode`
- Propagate concrete tensor shapes using `torch.fx.passes.shape_prop.ShapeProp`

```python
# tracer.py interface
def trace_model(model: nn.Module, example_inputs: tuple) -> IRGraph:
    ...
```

**IR definitions:** `ir.py`

```python
@dataclass
class IRNode:
    op_type: str              # "matmul", "softmax", "add", "layernorm", ...
    input_shapes: list[tuple]
    output_shapes: list[tuple]
    attributes: dict          # e.g. {"heads": 8, "seq_len": 512}
    fx_node_name: str         # original fx node name for traceability

@dataclass
class IRGraph:
    nodes: list[IRNode]
    # edges are implicit: sequential order for MVP
```

**Op type normalization:** map `torch.ops.aten.*` names to canonical strings:
- `aten.mm`, `aten.bmm`, `aten.matmul` → `"matmul"`
- `aten.softmax` → `"softmax"`
- `aten.add`, `aten.mul`, `aten.relu`, `aten.gelu` → `"elementwise"`
- `aten.layer_norm` → `"layernorm"`
- `aten.embedding` → `"embedding"`

Unknown ops should map to `"unknown"` with a warning — never silently drop nodes.

---

### 2. Cost Model (`accel_sim/cost_model/`)

**This is the core of the project.** Keep formulas simple, documented, and easy to override.

#### `ops.py` — Compute Cost Functions

Each function takes an `IRNode` and a `DeviceProfile` and returns `(flops: float, compute_time_ms: float)`.

**Matmul** `(M, K) × (K, N)`:
```
flops = 2 * M * N * K
time  = flops / (device.compute_tflops * 1e12) * 1000  # ms
```
For batched matmul `(B, M, K) × (B, K, N)`: multiply by B.

**Attention** (broken into sub-ops — do not treat as a single node):
- `QKᵀ`:  `(B, H, S, D) × (B, H, D, S)` → matmul formula
- Softmax: see below
- `Attn × V`: `(B, H, S, S) × (B, H, S, D)` → matmul formula

**Softmax** over shape `(..., N)`:
```
flops ≈ 5 * N * prod(other_dims)   # exp, sum, div, max, sub
time  = flops / (device.compute_tflops * 1e12) * 1000
```

**Elementwise** (add, mul, relu, gelu, etc.):
```
flops = prod(output_shape)    # ~1 FLOP per element as a floor
time  = flops / (device.compute_tflops * 1e12) * 1000
```
For `gelu`, use a multiplier of 8 (it is not a trivial op).

**LayerNorm** over shape `(B, S, H)`:
```
flops ≈ 8 * B * S * H    # mean, variance, normalize, scale, shift
time  = flops / (device.compute_tflops * 1e12) * 1000
```

**Embedding** lookup `(B, S)` → `(B, S, D)`:
```
# Compute is trivial; this is memory-bound
bytes_read = B * S * D * dtype_bytes
time = bytes_read / (device.memory_bandwidth * 1e9) * 1000
```

**Unknown ops:** return `(0, 0)` with a logged warning — do not crash.

All cost functions should be **roofline-aware**: for compute-light ops (low arithmetic intensity), use bandwidth as the bottleneck rather than TFLOPS. Arithmetic intensity = flops / bytes_accessed.

```python
def roofline_time_ms(flops: float, bytes_accessed: float, device: DeviceProfile) -> float:
    compute_time = flops / (device.compute_tflops * 1e12) * 1000
    memory_time  = bytes_accessed / (device.memory_bandwidth * 1e9) * 1000
    return max(compute_time, memory_time)
```

#### `memory.py` — Memory Estimates

Track two things per node:
1. **Output activation size** = `prod(output_shape) * dtype_bytes`
2. **Temporary buffer size** = op-specific (e.g. attention score matrix `B * H * S * S * 4`)

```python
@dataclass
class MemoryEstimate:
    activation_bytes: int
    temp_buffer_bytes: int

def estimate_memory(node: IRNode, dtype_bytes: int = 2) -> MemoryEstimate:
    ...
```

Peak memory is tracked by the simulator engine, not by individual op functions.

---

### 3. Device Profiles (`accel_sim/devices/profiles.py`)

```python
@dataclass
class DeviceProfile:
    name: str
    compute_tflops: float      # FP16 peak TFLOPS
    memory_bandwidth: float    # GB/s
    memory_limit_gb: float
    dtype_bytes: int = 2       # default FP16

# Built-in profiles
V100  = DeviceProfile("V100",  compute_tflops=125,  memory_bandwidth=900,  memory_limit_gb=16)
A100  = DeviceProfile("A100",  compute_tflops=312,  memory_bandwidth=2000, memory_limit_gb=40)
H100  = DeviceProfile("H100",  compute_tflops=989,  memory_bandwidth=3350, memory_limit_gb=80)
TPU_V4 = DeviceProfile("TPU-v4", compute_tflops=275, memory_bandwidth=1200, memory_limit_gb=32)
```

Users can instantiate `DeviceProfile` directly to define custom accelerators.

---

### 4. Simulator Engine (`accel_sim/engine/simulator.py`)

Walk the `IRGraph` sequentially (no parallelism in MVP). For each node:

1. Call the appropriate cost function → `(flops, time_ms)`
2. Call `estimate_memory` → `MemoryEstimate`
3. Accumulate total latency
4. Track live memory (activation + temp buffer of current node; temp buffers are freed after the node)
5. Record per-node results for bottleneck reporting

```python
@dataclass
class NodeResult:
    node: IRNode
    flops: float
    time_ms: float
    activation_bytes: int
    temp_buffer_bytes: int

@dataclass
class SimResult:
    device: DeviceProfile
    total_latency_ms: float
    peak_memory_gb: float
    node_results: list[NodeResult]
    memory_oom: bool           # True if peak_memory_gb > device.memory_limit_gb

def simulate(graph: IRGraph, device: DeviceProfile) -> SimResult:
    ...
```

Peak memory tracking: maintain a running sum of all live activations (all outputs not yet consumed by later nodes). Use a simple liveness model for MVP: all activations are live until the end of the graph.

---

### 5. Output & Reporting (`accel_sim/output/report.py`)

```python
def format_report(result: SimResult) -> str:
    ...
```

Output format:

```
Device: V100 (125 TFLOPS, 900 GB/s, 16 GB)

Total latency:   42.3 ms
Peak memory:      9.1 GB  ✓ (fits in device memory)

Top bottlenecks:
  1. attn_qk_matmul       27.6 ms  (65.2%)
  2. mlp_fc1_matmul        8.5 ms  (20.1%)
  3. attn_softmax          3.2 ms   (7.6%)
  4. mlp_fc2_matmul        1.8 ms   (4.3%)
     ... (remaining ops <3%)

Op breakdown:
  matmul         38.1 ms  (90.1%)
  softmax         3.2 ms   (7.6%)
  layernorm       0.9 ms   (2.1%)
  elementwise     0.1 ms   (0.2%)
```

If `memory_oom` is True, print a clear warning:
```
⚠  OOM WARNING: Peak memory 18.4 GB exceeds device limit of 16.0 GB
```

---

### 6. CLI (`accel_sim/cli.py`)

```bash
accel-sim simulate path/to/model_script.py --device v100
accel-sim simulate path/to/model_script.py --device a100 --seq-len 2048
accel-sim compare  path/to/model_script.py --devices v100,a100,h100
accel-sim devices                          # list available device profiles
```

The model script must expose a `get_model()` function returning `(model, example_inputs)`.

Use `argparse` or `click`. Keep the CLI thin — it just calls `trace_model` → `simulate` → `format_report`.

---

## Key Implementation Rules

**Never crash on unsupported ops.** Log a warning, return zero cost, and continue. The graph traversal must always complete.

**Shapes must be concrete integers.** If `ShapeProp` returns symbolic shapes, raise a clear error with instructions to pass concrete `example_inputs`.

**Cost functions are pure.** They take `(IRNode, DeviceProfile)` and return numbers. No side effects, no global state.

**The IR is the contract.** Nothing downstream of `ir.py` imports from `torch`. Cost model, engine, and CLI only see `IRNode`/`IRGraph` + `DeviceProfile`.

**Roofline, not just TFLOPS.** Every compute estimate must account for memory bandwidth via `roofline_time_ms`. Ignoring bandwidth makes attention estimates wildly optimistic.

**No compiler optimizations modeled in MVP.** Assume no fusion, no tiling optimization, no tensor cores utilization bonus. This gives a conservative (pessimistic) lower bound on speed, which is safer than false optimism.

---

## Testing Strategy

### Unit tests — `tests/test_cost_model.py`

- Matmul with known shapes: verify FLOPs formula manually
- Roofline: test that bandwidth-bound ops pick memory time over compute time
- OOM detection: feed a graph with large activations, assert `memory_oom = True`

### Integration tests — `tests/test_simulator.py`

- Trace `fixtures/tiny_transformer.py` (a 2-layer GPT block, hidden=64, heads=4)
- Assert `total_latency_ms > 0`
- Assert `peak_memory_gb > 0`
- Assert bottleneck node is a matmul
- Run on multiple devices and assert A100 result is faster than V100

### Tracer tests — `tests/test_tracer.py`

- Trace `nn.Linear`, `nn.MultiheadAttention` — assert expected op types appear in graph
- Assert no nodes have `None` output shapes after tracing

### Regression baseline

Once estimates are stable, snapshot the output for `tiny_transformer.py` on V100 at seq_len=512 and check in as a golden file. Any change to cost model formulas must update the golden file intentionally.

---

## What This Is Not

- Not cycle-accurate. Do not claim ±5% accuracy.
- Not a compiler. Does not model fusion, kernel dispatch, or operator scheduling.
- Not a profiler. Does not execute on any GPU hardware.
- Not a general ML framework simulator. Scope is transformer workloads only.

Estimates should be presented as "directionally correct order-of-magnitude projections." A 2× error is acceptable. A 10× error for a common op is a bug.

---

## Extension Points (future, not MVP)

- **Operator fusion:** detect fusable op pairs (e.g. matmul + bias + relu) and apply a fusion discount to memory traffic
- **FlashAttention cost model:** model tiled attention separately — it has fundamentally different memory behavior
- **Batch size sweep:** simulate across a range of batch sizes and plot throughput vs latency tradeoff
- **Auto-suggestions:** if a matmul is bandwidth-bound, suggest quantization; if an op exceeds memory, suggest activation checkpointing
- **JAX/XLA frontend:** accept HLO graphs in addition to torch.fx
- **Export to CSV/JSON:** for integration with external dashboards

---

## Dependencies

```toml
[project]
dependencies = [
    "torch>=2.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]
```

No heavy dependencies. `torch` is required for tracing. Everything else is stdlib + dataclasses.

---

## Glossary

| Term | Meaning |
|---|---|
| FLOPs | Floating point operations (not per second) |
| TFLOPS | Tera floating point operations **per second** — device throughput |
| Roofline | Model that takes the max of compute time and memory time |
| Arithmetic intensity | FLOPs / bytes accessed — determines if an op is compute- or memory-bound |
| IR | Intermediate representation — the project's device-agnostic graph format |
| OOM | Out of memory — peak activation exceeds device memory limit |
| ShapeProp | `torch.fx.passes.shape_prop.ShapeProp` — propagates concrete shapes through an fx graph |