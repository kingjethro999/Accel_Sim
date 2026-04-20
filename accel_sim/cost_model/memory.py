from dataclasses import dataclass
from ..capture.ir import IRNode

@dataclass
class MemoryEstimate:
    activation_bytes: int
    temp_buffer_bytes: int

def prod(dims):
    p = 1
    for d in dims:
        p *= d
    return p

def estimate_memory(node: IRNode, dtype_bytes: int = 2) -> MemoryEstimate:
    """Estimate activation and temporary buffer memory for a node."""
    activation_bytes = 0
    if node.output_shapes:
        activation_bytes = prod(node.output_shapes[0]) * dtype_bytes
        
    temp_buffer_bytes = 0
    
    # Special cases for temp buffers
    if node.op_type == "softmax":
        # Softmax often needs a temporary copy of the input or scores
        if node.input_shapes:
            temp_buffer_bytes = prod(node.input_shapes[0]) * 4 # FP32 for softmax stability
            
    # Note: In a real transformer, the B*H*S*S matrix is the output of the QK matmul
    # which we already count in activation_bytes for that node.
    # The spec mentions "e.g. attention score matrix B * H * S * S * 4" as a temp buffer.
    # If we are at the softmax node, that matrix is its input.
    
    return MemoryEstimate(
        activation_bytes=activation_bytes,
        temp_buffer_bytes=temp_buffer_bytes
    )
