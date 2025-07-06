#!/usr/bin/env python3
"""
Script to manage TensorBoard experiments for Mario RL training
"""

import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def list_experiments(runs_dir="runs"):
    """List all available experiments"""
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        print(f"No experiments found. Directory {runs_dir} does not exist.")
        return []
    
    def has_event_files_recursive(directory):
        """Check if directory has any TensorBoard event files (recursively)"""
        if not directory.exists() or not directory.is_dir():
            return False
        
        # Check current directory
        for file in directory.glob('*'):
            if file.is_file() and file.name.startswith('events.out.tfevents'):
                return True
        
        # Check subdirectories recursively
        for subdir in directory.iterdir():
            if subdir.is_dir() and has_event_files_recursive(subdir):
                return True
        
        return False
    
    experiments = []
    for exp_dir in runs_path.iterdir():
        if exp_dir.is_dir():
            # Get creation time
            creation_time = datetime.fromtimestamp(exp_dir.stat().st_ctime)
            
            # Check if it has TensorBoard event files (recursively)
            has_events = has_event_files_recursive(exp_dir)
            
            experiments.append({
                'name': exp_dir.name,
                'path': exp_dir,
                'created': creation_time,
                'has_events': has_events
            })
    
    # Sort by creation time (newest first)
    experiments.sort(key=lambda x: x['created'], reverse=True)
    
    return experiments


def display_experiments(experiments):
    """Display experiments in a formatted table"""
    if not experiments:
        print("No experiments found.")
        return
    
    print("\n" + "="*80)
    print("TENSORBOARD EXPERIMENTS")
    print("="*80)
    print(f"{'#':>3} {'Name':<40} {'Created':<20} {'Status':<10}")
    print("-" * 80)
    
    for i, exp in enumerate(experiments, 1):
        status = "✓ Ready" if exp['has_events'] else "✗ No data"
        created_str = exp['created'].strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:>3} {exp['name']:<40} {created_str:<20} {status:<10}")
    
    print("="*80)


def remove_experiment(experiment_name, runs_dir="runs"):
    """Remove a specific experiment"""
    runs_path = Path(runs_dir)
    exp_path = runs_path / experiment_name
    
    if not exp_path.exists():
        print(f"Experiment '{experiment_name}' not found.")
        return False
    
    if not exp_path.is_dir():
        print(f"'{experiment_name}' is not a directory.")
        return False
    
    # Confirm deletion
    confirm = input(f"Are you sure you want to delete experiment '{experiment_name}'? (y/N): ")
    if confirm.lower() != 'y':
        print("Deletion cancelled.")
        return False
    
    try:
        shutil.rmtree(exp_path)
        print(f"Successfully removed experiment '{experiment_name}'.")
        return True
    except Exception as e:
        print(f"Error removing experiment: {e}")
        return False


def cleanup_empty_experiments(runs_dir="runs"):
    """Remove experiments with no data and nested empty directories, including TensorBoard hparams subdirs"""
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        print(f"Directory {runs_dir} does not exist.")
        return
    
    removed_count = 0
    hparams_subdir_count = 0
    
    def has_event_files(directory):
        """Check if directory has any TensorBoard event files"""
        if not directory.exists() or not directory.is_dir():
            return False
        
        # Check current directory
        for file in directory.glob('*'):
            if file.is_file() and file.name.startswith('events.out.tfevents'):
                return True
        
        # Check subdirectories recursively
        for subdir in directory.iterdir():
            if subdir.is_dir() and has_event_files(subdir):
                return True
        
        return False
    
    def is_hparams_subdir(directory):
        """Check if this is a TensorBoard hparams subdirectory (numeric name)"""
        try:
            float(directory.name)
            return True
        except ValueError:
            return False
    
    def remove_empty_subdirs(directory):
        """Remove empty subdirectories recursively"""
        if not directory.exists() or not directory.is_dir():
            return
        
        # First, clean up subdirectories
        for subdir in directory.iterdir():
            if subdir.is_dir():
                remove_empty_subdirs(subdir)
                # Remove if now empty
                try:
                    if not any(subdir.iterdir()):
                        subdir.rmdir()
                        print(f"Removed empty subdirectory: {subdir}")
                except OSError:
                    pass  # Directory not empty or other issue
    
    # Process all experiment directories
    for exp_dir in runs_path.iterdir():
        if exp_dir.is_dir():
            # Special handling for TensorBoard hparams subdirectories
            for subdir in list(exp_dir.iterdir()):
                if subdir.is_dir() and is_hparams_subdir(subdir):
                    # Remove numeric subdirectories created by add_hparams
                    try:
                        shutil.rmtree(subdir)
                        print(f"Removed TensorBoard hparams subdirectory: {subdir}")
                        hparams_subdir_count += 1
                    except Exception as e:
                        print(f"Error removing hparams subdir {subdir}: {e}")
            
            # First clean up empty subdirectories
            remove_empty_subdirs(exp_dir)
            
            # Check if the entire experiment directory should be removed
            if not has_event_files(exp_dir):
                try:
                    shutil.rmtree(exp_dir)
                    print(f"Removed empty experiment: {exp_dir.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {exp_dir.name}: {e}")
    
    print(f"Cleaned up {removed_count} empty experiments and {hparams_subdir_count} TensorBoard hparams subdirectories.")


def interactive_remove(runs_dir="runs"):
    """Interactive experiment removal"""
    experiments = list_experiments(runs_dir)
    
    if not experiments:
        return
    
    display_experiments(experiments)
    
    try:
        choice = input("\nEnter experiment number to remove (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            exp_index = int(choice) - 1
            if 0 <= exp_index < len(experiments):
                experiment = experiments[exp_index]
                remove_experiment(experiment['name'], runs_dir)
            else:
                print("Invalid experiment number.")
        except ValueError:
            print("Please enter a valid number.")
    
    except (EOFError, KeyboardInterrupt):
        print("\nOperation cancelled.")


def get_tensorboard_command(runs_dir="runs"):
    """Get the TensorBoard command to run"""
    return f"tensorboard --logdir={runs_dir}"


def main():
    parser = argparse.ArgumentParser(description='Manage TensorBoard experiments for Mario RL')
    parser.add_argument('--runs-dir', default='runs', help='Directory containing experiment runs')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove an experiment')
    remove_parser.add_argument('experiment_name', help='Name of experiment to remove')
    
    # Interactive remove command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive experiment management')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove empty experiments')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start TensorBoard')
    start_parser.add_argument('--port', default=6006, type=int, help='Port to run TensorBoard on')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        experiments = list_experiments(args.runs_dir)
        display_experiments(experiments)
        print(f"\nTo start TensorBoard: {get_tensorboard_command(args.runs_dir)}")
    
    elif args.command == 'remove':
        remove_experiment(args.experiment_name, args.runs_dir)
    
    elif args.command == 'interactive':
        interactive_remove(args.runs_dir)
    
    elif args.command == 'cleanup':
        cleanup_empty_experiments(args.runs_dir)
    
    elif args.command == 'start':
        import subprocess
        cmd = f"tensorboard --logdir={args.runs_dir} --port={args.port}"
        print(f"Starting TensorBoard: {cmd}")
        subprocess.run(cmd, shell=True)
    
    else:
        # Default: show experiments and TensorBoard command
        experiments = list_experiments(args.runs_dir)
        display_experiments(experiments)
        print(f"\nCommands:")
        print(f"  List experiments:     python {__file__} list")
        print(f"  Remove experiment:    python {__file__} remove <experiment_name>")
        print(f"  Interactive mode:     python {__file__} interactive")
        print(f"  Cleanup empty:        python {__file__} cleanup")
        print(f"  Start TensorBoard:    python {__file__} start")
        print(f"  Manual TensorBoard:   {get_tensorboard_command(args.runs_dir)}")


if __name__ == "__main__":
    main() 