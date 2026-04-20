import pytest
from accel_sim.capture.ir import IRNode
from accel_sim.devices.profiles import V100, A100
from accel_sim.cost_model.ops import estimate_matmul, roofline_time_ms

def test_matmul_flops():
    # (128, 256) x (256, 512)
    node = IRNode(
        op_type="matmul",
        input_shapes=[(128, 256), (256, 512)],
        output_shapes=[(128, 512)]
    )
    flops, time = estimate_matmul(node, V100)
    
    # Formula: 2 * M * N * K
    expected_flops = 2 * 128 * 512 * 256
    assert flops == expected_flops

def test_roofline_v100():
    # V100: 125 TFLOPS, 900 GB/s
    # Case 1: Compute bound
    # 1e12 flops, 1e6 bytes -> intensity 1e6
    time = roofline_time_ms(1e12, 1e6, V100)
    compute_time = 1e12 / (125 * 1e12) * 1000 # 8 ms
    assert time == pytest.approx(compute_time)
    
    # Case 2: Memory bound
    # 1e6 flops, 1e9 bytes -> intensity 1e-3
    time = roofline_time_ms(1e6, 1e9, V100)
    memory_time = 1e9 / (900 * 1e9) * 1000 # 1.11 ms
    assert time == pytest.approx(memory_time)

def test_compare_devices():
    node = IRNode(
        op_type="matmul",
        input_shapes=[(1024, 1024), (1024, 1024)],
        output_shapes=[(1024, 1024)]
    )
    _, time_v100 = estimate_matmul(node, V100)
    _, time_a100 = estimate_matmul(node, A100)
    
    # A100 should be faster than V100
    assert time_a100 < time_v100
