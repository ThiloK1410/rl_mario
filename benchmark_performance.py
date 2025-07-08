#!/usr/bin/env python3
"""
Performance benchmark script to compare original vs optimized training pipelines.
Measures key metrics:
- Experience collection rate (experiences/second)
- Training throughput (batches/second)
- GPU utilization
- Lock contention statistics
- Memory usage
"""

import time
import torch
import psutil
import threading
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Performance monitoring class
class PerformanceMonitor:
    def __init__(self, name):
        self.name = name
        self.metrics = {
            'experience_rate': deque(maxlen=100),
            'batch_rate': deque(maxlen=100),
            'gpu_utilization': deque(maxlen=100),
            'gpu_memory': deque(maxlen=100),
            'cpu_percent': deque(maxlen=100),
            'lock_wait_time': deque(maxlen=100),
            'epoch_times': deque(maxlen=100),
        }
        self.start_time = None
        self.experience_count = 0
        self.batch_count = 0
        self.lock_waits = 0
        self.lock_wait_total = 0
        
    def start(self):
        self.start_time = time.time()
        
    def record_experiences(self, count):
        self.experience_count += count
        elapsed = time.time() - self.start_time
        rate = self.experience_count / elapsed
        self.metrics['experience_rate'].append(rate)
        
    def record_batch(self):
        self.batch_count += 1
        elapsed = time.time() - self.start_time
        rate = self.batch_count / elapsed
        self.metrics['batch_rate'].append(rate)
        
    def record_lock_wait(self, wait_time):
        self.lock_waits += 1
        self.lock_wait_total += wait_time
        self.metrics['lock_wait_time'].append(wait_time * 1000)  # Convert to ms
        
    def record_epoch_time(self, epoch_time):
        self.metrics['epoch_times'].append(epoch_time)
        
    def record_system_stats(self):
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.metrics['cpu_percent'].append(cpu_percent)
        
        # GPU stats if available
        if torch.cuda.is_available():
            # GPU utilization
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.metrics['gpu_utilization'].append(util.gpu)
                
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_percent = (mem_info.used / mem_info.total) * 100
                self.metrics['gpu_memory'].append(mem_percent)
            except:
                self.metrics['gpu_utilization'].append(0)
                self.metrics['gpu_memory'].append(0)
                
    def get_summary(self):
        elapsed = time.time() - self.start_time
        return {
            'name': self.name,
            'total_time': elapsed,
            'total_experiences': self.experience_count,
            'total_batches': self.batch_count,
            'avg_experience_rate': np.mean(self.metrics['experience_rate']) if self.metrics['experience_rate'] else 0,
            'avg_batch_rate': np.mean(self.metrics['batch_rate']) if self.metrics['batch_rate'] else 0,
            'avg_gpu_utilization': np.mean(self.metrics['gpu_utilization']) if self.metrics['gpu_utilization'] else 0,
            'avg_gpu_memory': np.mean(self.metrics['gpu_memory']) if self.metrics['gpu_memory'] else 0,
            'avg_cpu_percent': np.mean(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else 0,
            'avg_lock_wait_ms': np.mean(self.metrics['lock_wait_time']) if self.metrics['lock_wait_time'] else 0,
            'total_lock_waits': self.lock_waits,
            'avg_epoch_time': np.mean(self.metrics['epoch_times']) if self.metrics['epoch_times'] else 0,
        }


def benchmark_original_pipeline(num_epochs=10):
    """Benchmark the original training pipeline."""
    print("\n" + "="*80)
    print("BENCHMARKING ORIGINAL PIPELINE")
    print("="*80)
    
    monitor = PerformanceMonitor("Original Pipeline")
    
    # Import original components
    from mario_rl import main as original_main
    from mario_rl import ExperienceProcessor
    
    # Monkey patch to add monitoring
    original_process_batch = ExperienceProcessor._process_batch
    def monitored_process_batch(self):
        result = original_process_batch(self)
        monitor.record_experiences(result)
        monitor.record_batch()
        return result
    ExperienceProcessor._process_batch = monitored_process_batch
    
    # Run benchmark
    monitor.start()
    
    # Monitor system stats in background
    stop_monitoring = threading.Event()
    def monitor_thread():
        while not stop_monitoring.is_set():
            monitor.record_system_stats()
            time.sleep(1)
    
    monitoring_thread = threading.Thread(target=monitor_thread, daemon=True)
    monitoring_thread.start()
    
    # Run training for specified epochs
    try:
        # Modify config temporarily for benchmark
        import config
        original_num_epochs = config.NUM_EPOCHS
        config.NUM_EPOCHS = num_epochs
        
        # Run training
        original_main()
        
        # Restore config
        config.NUM_EPOCHS = original_num_epochs
        
    finally:
        stop_monitoring.set()
        monitoring_thread.join()
    
    return monitor.get_summary()


def benchmark_optimized_pipeline(num_epochs=10):
    """Benchmark the optimized training pipeline."""
    print("\n" + "="*80)
    print("BENCHMARKING OPTIMIZED PIPELINE")
    print("="*80)
    
    monitor = PerformanceMonitor("Optimized Pipeline")
    
    # Import optimized components with optimized config
    os.environ['TRAINING_CONFIG'] = 'config_optimized'
    from mario_rl_optimized import main as optimized_main
    from mario_rl_optimized import LockFreeExperienceProcessor
    
    # Monkey patch to add monitoring
    original_process_batch = LockFreeExperienceProcessor._process_batch
    def monitored_process_batch(self, experiences):
        original_process_batch(self, experiences)
        monitor.record_experiences(len(experiences))
        monitor.record_batch()
    LockFreeExperienceProcessor._process_batch = monitored_process_batch
    
    # Run benchmark
    monitor.start()
    
    # Monitor system stats in background
    stop_monitoring = threading.Event()
    def monitor_thread():
        while not stop_monitoring.is_set():
            monitor.record_system_stats()
            time.sleep(1)
    
    monitoring_thread = threading.Thread(target=monitor_thread, daemon=True)
    monitoring_thread.start()
    
    # Run training for specified epochs
    try:
        # Use optimized config
        import config_optimized
        original_num_epochs = config_optimized.NUM_EPOCHS
        config_optimized.NUM_EPOCHS = num_epochs
        
        # Run training
        optimized_main()
        
        # Restore config
        config_optimized.NUM_EPOCHS = original_num_epochs
        
    finally:
        stop_monitoring.set()
        monitoring_thread.join()
    
    return monitor.get_summary()


def plot_comparison(original_stats, optimized_stats):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Pipeline Performance Comparison', fontsize=16)
    
    # Prepare data
    metrics = ['avg_experience_rate', 'avg_batch_rate', 'avg_gpu_utilization', 
               'avg_cpu_percent', 'avg_lock_wait_ms', 'avg_epoch_time']
    labels = ['Experience Rate\n(exp/s)', 'Batch Rate\n(batches/s)', 'GPU Utilization\n(%)',
              'CPU Usage\n(%)', 'Lock Wait Time\n(ms)', 'Epoch Time\n(seconds)']
    
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 3, idx % 3]
        
        original_val = original_stats.get(metric, 0)
        optimized_val = optimized_stats.get(metric, 0)
        
        bars = ax.bar(['Original', 'Optimized'], [original_val, optimized_val],
                      color=['#3498db', '#2ecc71'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_ylabel(label)
        ax.set_title(metric.replace('_', ' ').title())
        
        # Calculate and show improvement
        if original_val > 0:
            improvement = ((optimized_val - original_val) / original_val) * 100
            ax.text(0.5, 0.95, f'{improvement:+.1f}%', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'performance_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results(original_stats, optimized_stats):
    """Save benchmark results to JSON."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'original': original_stats,
        'optimized': optimized_stats,
        'improvements': {}
    }
    
    # Calculate improvements
    for metric in ['avg_experience_rate', 'avg_batch_rate', 'avg_gpu_utilization']:
        original_val = original_stats.get(metric, 0)
        optimized_val = optimized_stats.get(metric, 0)
        if original_val > 0:
            improvement = ((optimized_val - original_val) / original_val) * 100
            results['improvements'][metric] = improvement
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'benchmark_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")


def print_comparison_table(original_stats, optimized_stats):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<30} {'Original':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-"*80)
    
    metrics = [
        ('Experience Rate (exp/s)', 'avg_experience_rate', '{:.1f}'),
        ('Batch Rate (batches/s)', 'avg_batch_rate', '{:.2f}'),
        ('GPU Utilization (%)', 'avg_gpu_utilization', '{:.1f}'),
        ('GPU Memory (%)', 'avg_gpu_memory', '{:.1f}'),
        ('CPU Usage (%)', 'avg_cpu_percent', '{:.1f}'),
        ('Lock Wait Time (ms)', 'avg_lock_wait_ms', '{:.2f}'),
        ('Epoch Time (s)', 'avg_epoch_time', '{:.2f}'),
        ('Total Lock Waits', 'total_lock_waits', '{:d}'),
    ]
    
    for label, key, fmt in metrics:
        original_val = original_stats.get(key, 0)
        optimized_val = optimized_stats.get(key, 0)
        
        if isinstance(original_val, float) or isinstance(optimized_val, float):
            if original_val > 0:
                if key in ['avg_lock_wait_ms', 'avg_epoch_time']:  # Lower is better
                    improvement = ((original_val - optimized_val) / original_val) * 100
                else:  # Higher is better
                    improvement = ((optimized_val - original_val) / original_val) * 100
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement_str = "N/A"
        else:
            improvement_str = "N/A"
        
        print(f"{label:<30} {fmt.format(original_val):>15} {fmt.format(optimized_val):>15} {improvement_str:>15}")
    
    print("="*80)


def main():
    """Run the benchmark comparison."""
    print("Mario RL Training Pipeline Performance Benchmark")
    print("=" * 80)
    
    # Check if running in suitable environment
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Results may not be representative.")
    
    # Number of epochs to benchmark
    num_epochs = 5  # Adjust based on your needs
    
    print(f"\nBenchmarking both pipelines for {num_epochs} epochs...")
    print("This may take some time...\n")
    
    # Run benchmarks
    try:
        # Benchmark original pipeline
        original_stats = benchmark_original_pipeline(num_epochs)
        
        # Brief pause between benchmarks
        time.sleep(5)
        
        # Benchmark optimized pipeline
        optimized_stats = benchmark_optimized_pipeline(num_epochs)
        
        # Display results
        print_comparison_table(original_stats, optimized_stats)
        
        # Save results
        save_results(original_stats, optimized_stats)
        
        # Create visualization
        plot_comparison(original_stats, optimized_stats)
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()