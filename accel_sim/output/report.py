from typing import Dict
from ..engine.simulator import SimResult

def format_report(result: SimResult) -> str:
    """Format the simulation result into a human-readable report."""
    device = result.device
    
    lines = []
    lines.append(f"Device: {device.name} ({device.compute_tflops} TFLOPS, {device.memory_bandwidth} GB/s, {device.memory_limit_gb} GB)")
    lines.append("")
    
    status_char = "✓" if not result.memory_oom else "⚠"
    status_msg = "(fits in device memory)" if not result.memory_oom else "OOM WARNING: exceeds device limit"
    
    lines.append(f"Total latency:   {result.total_latency_ms:.2f} ms")
    lines.append(f"Peak memory:      {result.peak_memory_gb:.2f} GB  {status_char} {status_msg}")
    lines.append("")
    
    # Bottlenecks
    lines.append("Top bottlenecks:")
    sorted_nodes = sorted(result.node_results, key=lambda x: x.time_ms, reverse=True)
    
    total_time = result.total_latency_ms if result.total_latency_ms > 0 else 1.0
    
    # Filter nodes with significant contribution
    for i, res in enumerate(sorted_nodes[:10]):
        pct = (res.time_ms / total_time) * 100
        if pct < 1.0 and i > 4:
            break
        name = res.node.fx_node_name or res.node.op_type
        lines.append(f"  {i+1}. {name:<20} {res.time_ms:>7.2f} ms  ({pct:>5.1f}%)")
    
    if len(sorted_nodes) > 10:
        remaining_time = sum(n.time_ms for n in sorted_nodes[10:])
        remaining_pct = (remaining_time / total_time) * 100
        if remaining_pct > 0.1:
            lines.append(f"     ... (remaining ops {remaining_pct:>4.1f}%)")
    
    lines.append("")
    
    # Op breakdown
    lines.append("Op breakdown:")
    op_totals: Dict[str, float] = {}
    for res in result.node_results:
        op_totals[res.node.op_type] = op_totals.get(res.node.op_type, 0.0) + res.time_ms
        
    sorted_ops = sorted(op_totals.items(), key=lambda x: x[1], reverse=True)
    for op_type, op_time in sorted_ops:
        pct = (op_time / total_time) * 100
        lines.append(f"  {op_type:<14} {op_time:>7.2f} ms  ({pct:>5.1f}%)")
        
    if result.memory_oom:
        lines.append("")
        lines.append(f"⚠  OOM WARNING: Peak memory {result.peak_memory_gb:.2f} GB exceeds device limit of {device.memory_limit_gb:.2f} GB")
        
    return "\n".join(lines)
