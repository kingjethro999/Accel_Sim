from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class IRNode:
    op_type: str              # "matmul", "softmax", "elementwise", "layernorm", "embedding", "unknown"
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    attributes: Dict[str, Any] = field(default_factory=dict)
    fx_node_name: str = ""

@dataclass
class IRGraph:
    nodes: List[IRNode]
    # Edges are implicit for now (sequential order)
