import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp
import logging

from .ir import IRNode, IRGraph

logger = logging.getLogger(__name__)

from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import ShapeProp

def normalize_op_name(node: torch.fx.Node) -> str:
    """Map aten ops to canonical IR op types."""
    target = node.target
    if not hasattr(target, "__name__") and not isinstance(target, str):
        target_str = str(target)
    else:
        target_str = target.__name__ if hasattr(target, "__name__") else str(target)
    
    target_str = target_str.lower()
    
    # Split by '.' to handle aten.add.Tensor etc.
    parts = target_str.split('.')
    
    # Skip no-cost ops early (exact match on parts)
    nocost_ops = {"view", "reshape", "permute", "transpose", "contiguous", "slice", "expand", "clone", "detach", "to", "squeeze", "unsqueeze", "t", "alias"}
    if any(p in nocost_ops for p in parts):
        return "nocost"

    # Common matmul patterns in aten
    matmul_ops = {"mm", "bmm", "matmul", "addmm"}
    if any(p in matmul_ops for p in parts):
        return "matmul"
    
    # Softmax
    if any("softmax" in p for p in parts):
        return "softmax"
    
    # Elementwise
    elementwise_ops = {"add", "mul", "sub", "div", "relu", "gelu", "silu", "tanh", "sigmoid", "pow", "where", "gt", "lt", "eq", "exp", "sqrt", "neg", "abs"}
    if any(p in elementwise_ops for p in parts):
        return "elementwise"
    
    # LayerNorm
    if any(x in target_str for x in ["layer_norm", "native_layer_norm"]):
        return "layernorm"
    
    # Embedding
    if "embedding" in target_str:
        return "embedding"
    
    return "unknown"

def get_shape(tensor_meta):
    if tensor_meta is None:
        return None
    if hasattr(tensor_meta, 'shape'):
        return tuple(tensor_meta.shape)
    if isinstance(tensor_meta, dict) and 'shape' in tensor_meta:
        return tuple(tensor_meta['shape'])
    return None

def trace_model(model: nn.Module, example_inputs: tuple) -> IRGraph:
    """Trace a model and convert it to IRGraph."""
    # 1. Trace the model using make_fx to get aten ops
    # make_fx produces a flattened graph of aten operations
    # We need to wrap the model call in a function for make_fx
    def model_call(*args):
        return model(*args)

    try:
        traced = make_fx(model_call)(*example_inputs)
    except Exception as e:
        logger.warning(f"make_fx failed: {e}. Falling back to symbolic_trace.")
        traced = torch.fx.symbolic_trace(model)

    # 2. Propagate shapes (make_fx usually has them, but let's be sure)
    # Actually make_fx already has metadata.
    # But we can run ShapeProp just in case or if we fell back.
    if not any('tensor_meta' in node.meta for node in traced.graph.nodes):
        ShapeProp(traced).propagate(*example_inputs)

    ir_nodes = []
    
    for node in traced.graph.nodes:
        if node.op in ["placeholder", "output", "get_attr"]:
            continue
            
        op_type = normalize_op_name(node)
        
        if op_type == "nocost":
            continue
            
        if op_type == "unknown":
            logger.warning(f"Unknown op type for node: {node.name} (op: {node.op}, target: {node.target})")

        # Extract shapes
        input_shapes = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                # make_fx stores val in meta
                meta = node.meta.get('val', None) if arg == node else arg.meta.get('val', None)
                if hasattr(meta, 'shape'):
                    input_shapes.append(tuple(meta.shape))
                else:
                    # Fallback to tensor_meta
                    meta = arg.meta.get('tensor_meta', None)
                    shape = get_shape(meta)
                    if shape:
                        input_shapes.append(shape)
        
        # Output shape
        output_meta = node.meta.get('val', None)
        if hasattr(output_meta, 'shape'):
            output_shapes = [tuple(output_meta.shape)]
        else:
            output_meta = node.meta.get('tensor_meta', None)
            output_shape = get_shape(output_meta)
            output_shapes = [output_shape] if output_shape else []

        attributes = {}
        if "gelu" in str(node.target).lower():
            attributes["subtype"] = "gelu"
        
        ir_node = IRNode(
            op_type=op_type,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            attributes=attributes,
            fx_node_name=node.name
        )
        ir_nodes.append(ir_node)

    return IRGraph(nodes=ir_nodes)
