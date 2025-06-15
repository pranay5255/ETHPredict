#!/usr/bin/env python3
"""
Setup and Test Script for ETHPredict CLI Runner

This script helps set up the environment and test the CLI runner functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd):
    """Run a command and return its output."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        print(f"Success: {result.stdout}")
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn', 
        'typer', 'rich', 'yaml', 'optuna'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    return True

def check_config_files():
    """Check if configuration files exist."""
    print("\nChecking configuration files...")
    
    config_files = [
        'configs/training.yml',
        'configs/market_making.yml', 
        'configs/backtest.yml'
    ]
    
    missing = []
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file}")
        else:
            print(f"✗ {config_file}")
            missing.append(config_file)
    
    if missing:
        print(f"\nMissing config files: {missing}")
        return False
    
    return True

def test_cli_commands():
    """Test basic CLI commands."""
    print("\nTesting CLI commands...")
    
    # Test help command
    if not run_command("python runner.py --help"):
        print("CLI help command failed")
        return False
    
    # Test info command
    if not run_command("python runner.py info"):
        print("CLI info command failed")
        return False
    
    print("\nCLI basic commands working!")
    return True

def create_sample_experiment():
    """Create and run a sample experiment."""
    print("\nRunning sample experiment...")
    
    # Run a quick test experiment
    cmd = "python runner.py run --id test_setup --tag sample"
    print(f"Running sample experiment: {cmd}")
    
    # Note: This might fail if training modules aren't fully implemented
    # That's expected at this stage
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Sample experiment completed successfully!")
        return True
    else:
        print("Sample experiment failed (this may be expected if training modules need implementation)")
        print(f"Error: {result.stderr}")
        return False

def main():
    """Main setup and test function."""
    print("=" * 60)
    print("ETHPredict CLI Runner Setup and Test")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("runner.py").exists():
        print("Error: runner.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    success = True
    
    # Check dependencies
    if not check_dependencies():
        success = False
    
    # Check config files
    if not check_config_files():
        success = False
    
    # Test CLI commands
    if success and not test_cli_commands():
        success = False
    
    # Create sample experiment (optional)
    if success:
        create_sample_experiment()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Setup and basic tests completed successfully!")
        print("\nYou can now use the CLI runner:")
        print("  python runner.py info")
        print("  python runner.py run")
        print("  python runner.py grid --trials 10")
        print("  python runner.py report results/")
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main() 