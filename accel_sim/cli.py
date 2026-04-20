import click
import importlib.util
import os
import sys
from .capture.tracer import trace_model
from .engine.simulator import simulate
from .devices.profiles import PROFILES
from .output.report import format_report

def load_model_from_script(script_path):
    """Load model and example inputs from a user-provided script."""
    if not os.path.exists(script_path):
        click.echo(f"Error: Script {script_path} not found.", err=True)
        sys.exit(1)
        
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        click.echo(f"Error executing script: {e}", err=True)
        sys.exit(1)
        
    if not hasattr(module, "get_model"):
        click.echo("Error: Script must define a `get_model()` function.", err=True)
        sys.exit(1)
        
    return module.get_model()

@click.group()
def main():
    """accel-sim: Accelerator performance estimation tool."""
    pass

@main.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--device", default="v100", help="Device profile to use (v100, a100, h100, tpu-v4)")
def simulate_cmd(script_path, device):
    """Simulate a model's performance on a specific device."""
    device_name = device.lower()
    if device_name not in PROFILES:
        click.echo(f"Error: Unknown device profile '{device}'. Available: {', '.join(PROFILES.keys())}", err=True)
        sys.exit(1)
        
    profile = PROFILES[device_name]
    
    click.echo(f"Tracing model from {script_path}...")
    model, example_inputs = load_model_from_script(script_path)
    
    graph = trace_model(model, example_inputs)
    
    click.echo("Running simulation...")
    result = simulate(graph, profile)
    
    click.echo("-" * 40)
    click.echo(format_report(result))

@main.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--devices", default="v100,a100,h100", help="Comma-separated device profiles to compare")
def compare(script_path, devices):
    """Compare a model's performance across multiple devices."""
    device_names = [d.strip().lower() for d in devices.split(",")]
    
    model, example_inputs = load_model_from_script(script_path)
    graph = trace_model(model, example_inputs)
    
    click.echo(f"Comparing performance across {len(device_names)} devices...")
    click.echo(f"{'Device':<10} | {'Latency (ms)':<15} | {'Peak Mem (GB)':<15} | {'Status'}")
    click.echo("-" * 60)
    
    for name in device_names:
        if name not in PROFILES:
            click.echo(f"{name:<10} | {'UNKNOWN':<15} | {'-':<15} | -")
            continue
            
        profile = PROFILES[name]
        result = simulate(graph, profile)
        status = "OK" if not result.memory_oom else "OOM"
        
        click.echo(f"{name.upper():<10} | {result.total_latency_ms:>15.2f} | {result.peak_memory_gb:>15.2f} | {status}")

@main.command()
def devices():
    """List available device profiles."""
    click.echo("Available Device Profiles:")
    click.echo(f"{'Name':<10} | {'TFLOPS':>10} | {'Bandwidth':>15} | {'Memory (GB)':>15}")
    click.echo("-" * 60)
    for name, p in PROFILES.items():
        click.echo(f"{name.upper():<10} | {p.compute_tflops:>10.1f} | {p.memory_bandwidth:>15.1f} | {p.memory_limit_gb:>15.1f}")

if __name__ == "__main__":
    main()
