import os
import sys
import subprocess

def run_training(config_path=None):
    """
    Run the DF-GAN training on a single GPU
    
    Args:
        config_path: Path to the configuration file (optional)
    """
    train_script = os.path.join('code', 'src', 'train.py')
    
    # Build the command
    cmd = ['python', train_script]
    
    # Add config file if specified
    if config_path:
        cmd.extend(['--cfg', config_path])
    
    # Run the command
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    # If a config file is provided as an argument, use it
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_training(config_path)
