import torch
import platform
import subprocess

def get_gpu_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {platform.python_version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        # Show details for each GPU
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("No CUDA devices available")

    # Try to get NVIDIA driver version on Windows
    if platform.system() == "Windows":
        try:
            nvidia_smi_output = subprocess.check_output("nvidia-smi", shell=True).decode()
            for line in nvidia_smi_output.split('\n'):
                if "Driver Version" in line:
                    print(f"\nNVIDIA Driver: {line.strip()}")
                    break
        except:
            print("\nNVIDIA driver information not available")

if __name__ == "__main__":
    get_gpu_info()
    
    # Test a small tensor operation on GPU if available
    if torch.cuda.is_available():
        print("\nRunning a simple GPU test...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"Matrix multiplication result shape: {z.shape}")
        print("GPU test completed successfully!")