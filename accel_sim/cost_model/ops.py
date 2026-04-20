import math
from typing import Tuple
from ..capture.ir import IRNode
from ..devices.profiles import DeviceProfile

def prod(dims):
    p = 1
    for d in dims:
        p *= d
    return p

def roofline_time_ms(flops: float, bytes_accessed: float, device: DeviceProfile) -> float:
    """Calculate time using the roofline model: max(compute_time, memory_time)."""
    compute_time = flops / (device.compute_tflops * 1e12) * 1000
    memory_time  = bytes_accessed / (device.memory_bandwidth * 1e9) * 1000
    return max(compute_time, memory_time)

def estimate_matmul(node: IRNode, device: DeviceProfile) -> Tuple[float, float]:
    """Estimate cost for matmul: (B, M, K) x (B, K, N) -> (B, M, N)."""
    if not node.input_shapes or len(node.input_shapes) < 2:
        return 0.0, 0.0
    
    s1 = node.input_shapes[0]
    s2 = node.input_shapes[1]
    
    # Handle different matmul shapes
    # (M, K) x (K, N)
    if len(s1) == 2 and len(s2) == 2:
        B, M, K, N = 1, s1[0], s1[1], s2[1]
    # (B, M, K) x (B, K, N)
    elif len(s1) == 3 and len(s2) == 3:
        B, M, K, N = s1[0], s1[1], s1[2], s2[2]
    # (B, H, M, K) x (B, H, K, N)
    elif len(s1) == 4 and len(s2) == 4:
        B, M, K, N = s1[0] * s1[1], s1[2], s1[3], s2[3]
    else:
        # Fallback for weird shapes
        B = 1
        M = s1[-2] if len(s1) >= 2 else 1
        K = s1[-1]
        N = s2[-1]
        if len(s1) > 2:
            B = prod(s1[:-2])

    flops = 2 * B * M * N * K
    bytes_accessed = (B * M * K + B * K * N + B * M * N) * device.dtype_bytes
    
    time_ms = roofline_time_ms(flops, bytes_accessed, device)
    return float(flops), float(time_ms)

def estimate_softmax(node: IRNode, device: DeviceProfile) -> Tuple[float, float]:
    """Estimate cost for softmax."""
    if not node.output_shapes:
        return 0.0, 0.0
    
    shape = node.output_shapes[0]
    num_elements = prod(shape)
    # The last dimension size
    N = shape[-1] if shape else 1
    
    flops = 5 * num_elements # exp, sum, div, max, sub
    bytes_accessed = (num_elements * 2) * device.dtype_bytes # read + write
    
    time_ms = roofline_time_ms(flops, bytes_accessed, device)
    return float(flops), float(time_ms)

def estimate_elementwise(node: IRNode, device: DeviceProfile) -> Tuple[float, float]:
    """Estimate cost for elementwise ops."""
    if not node.output_shapes:
        return 0.0, 0.0
    
    num_elements = prod(node.output_shapes[0])
    
    multiplier = 1.0
    if node.attributes.get("subtype") == "gelu":
        multiplier = 8.0
        
    flops = num_elements * multiplier
    
    # bytes: inputs + output
    # Most elementwise ops are binary (2 inputs) or unary (1 input)
    num_inputs = len(node.input_shapes)
    bytes_accessed = (num_inputs + 1) * num_elements * device.dtype_bytes
    
    time_ms = roofline_time_ms(flops, bytes_accessed, device)
    return float(flops), float(time_ms)

def estimate_layernorm(node: IRNode, device: DeviceProfile) -> Tuple[float, float]:
    """Estimate cost for LayerNorm."""
    if not node.output_shapes:
        return 0.0, 0.0
    
    num_elements = prod(node.output_shapes[0])
    
    flops = 8 * num_elements # mean, variance, normalize, scale, shift
    bytes_accessed = (num_elements * 2) * device.dtype_bytes # read + write (approx)
    
    time_ms = roofline_time_ms(flops, bytes_accessed, device)
    return float(flops), float(time_ms)

def estimate_embedding(node: IRNode, device: DeviceProfile) -> Tuple[float, float]:
    """Estimate cost for Embedding lookup."""
    if not node.output_shapes:
        return 0.0, 0.0
    
    num_elements = prod(node.output_shapes[0])
    
    # Compute is trivial; this is memory-bound
    flops = 0.0
    bytes_read = num_elements * device.dtype_bytes
    
    time_ms = bytes_read / (device.memory_bandwidth * 1e9) * 1000
    return float(flops), float(time_ms)

def estimate_op_cost(node: IRNode, device: DeviceProfile) -> Tuple[float, float]:
    """Dispatch to the correct estimate function."""
    handlers = {
        "matmul": estimate_matmul,
        "softmax": estimate_softmax,
        "elementwise": estimate_elementwise,
        "layernorm": estimate_layernorm,
        "embedding": estimate_embedding,
    }
    
    handler = handlers.get(node.op_type)
    if handler:
        return handler(node, device)
    
    return 0.0, 0.0
