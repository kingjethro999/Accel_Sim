from dataclasses import dataclass
from typing import List
from ..capture.ir import IRGraph, IRNode
from ..devices.profiles import DeviceProfile
from ..cost_model.ops import estimate_op_cost
from ..cost_model.memory import estimate_memory, MemoryEstimate

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
    node_results: List[NodeResult]
    memory_oom: bool

def simulate(graph: IRGraph, device: DeviceProfile) -> SimResult:
    """Simulate graph execution on a device profile."""
    total_latency_ms = 0.0
    accumulated_activation_bytes = 0
    max_memory_bytes = 0
    node_results = []
    
    for node in graph.nodes:
        # 1. Compute cost
        flops, time_ms = estimate_op_cost(node, device)
        total_latency_ms += time_ms
        
        # 2. Memory cost
        mem = estimate_memory(node, device.dtype_bytes)
        
        # 3. Peak memory tracking
        # Activations stay live until end of graph per MVP rules
        accumulated_activation_bytes += mem.activation_bytes
        
        # Temp buffers are freed after node
        memory_at_node = accumulated_activation_bytes + mem.temp_buffer_bytes
        if memory_at_node > max_memory_bytes:
            max_memory_bytes = memory_at_node
            
        node_results.append(NodeResult(
            node=node,
            flops=flops,
            time_ms=time_ms,
            activation_bytes=mem.activation_bytes,
            temp_buffer_bytes=mem.temp_buffer_bytes
        ))
        
    peak_memory_gb = max_memory_bytes / (1024**3)
    memory_oom = peak_memory_gb > device.memory_limit_gb
    
    return SimResult(
        device=device,
        total_latency_ms=total_latency_ms,
        peak_memory_gb=peak_memory_gb,
        node_results=node_results,
        memory_oom=memory_oom
    )
