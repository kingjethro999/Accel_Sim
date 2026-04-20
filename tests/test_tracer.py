import torch
import torch.nn as nn
from accel_sim.capture.tracer import trace_model

def test_trace_linear():
    model = nn.Linear(128, 256)
    example_inputs = (torch.randn(1, 128),)
    graph = trace_model(model, example_inputs)
    
    # Linear should contain a matmul (or addmm mapped to matmul)
    op_types = [node.op_type for node in graph.nodes]
    assert "matmul" in op_types
    
    # Check shapes
    matmul_node = next(n for n in graph.nodes if n.op_type == "matmul")
    assert matmul_node.output_shapes[0] == (1, 256)

def test_trace_layernorm():
    model = nn.LayerNorm(128)
    example_inputs = (torch.randn(1, 128),)
    graph = trace_model(model, example_inputs)
    
    op_types = [node.op_type for node in graph.nodes]
    assert "layernorm" in op_types
