from dataclasses import dataclass

@dataclass
class DeviceProfile:
    name: str
    compute_tflops: float      # FP16 peak TFLOPS
    memory_bandwidth: float    # GB/s
    memory_limit_gb: float
    dtype_bytes: int = 2       # default FP16

# Built-in profiles
V100  = DeviceProfile("V100",  compute_tflops=125.0,  memory_bandwidth=900.0,  memory_limit_gb=16.0)
A100  = DeviceProfile("A100",  compute_tflops=312.0,  memory_bandwidth=2000.0, memory_limit_gb=40.0)
H100  = DeviceProfile("H100",  compute_tflops=989.0,  memory_bandwidth=3350.0, memory_limit_gb=80.0)
TPU_V4 = DeviceProfile("TPU-v4", compute_tflops=275.0, memory_bandwidth=1200.0, memory_limit_gb=32.0)

PROFILES = {
    "v100": V100,
    "a100": A100,
    "h100": H100,
    "tpu-v4": TPU_V4
}
