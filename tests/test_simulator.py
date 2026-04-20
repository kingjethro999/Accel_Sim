import torch
from accel_sim.capture.tracer import trace_model
from accel_sim.engine.simulator import simulate
from accel_sim.devices.profiles import V100, A100
from tests.fixtures.tiny_transformer import get_model

def test_simulator_tiny_transformer():
    model, example_inputs = get_model()
    graph = trace_model(model, example_inputs)
    
    result_v100 = simulate(graph, V100)
    result_a100 = simulate(graph, A100)
    
    assert result_v100.total_latency_ms > 0
    assert result_v100.peak_memory_gb > 0
    assert not result_v100.memory_oom # Tiny transformer should fit in 16GB
    
    # A100 should be faster
    assert result_a100.total_latency_ms < result_v100.total_latency_ms

def test_oom_detection():
    # Hidden size 16384, very large
    from dataclasses import replace
    tiny_v100 = replace(V100, memory_limit_gb=0.001) # 1MB limit
    
    model, example_inputs = get_model()
    graph = trace_model(model, example_inputs)
    
    result = simulate(graph, tiny_v100)
    assert result.memory_oom
