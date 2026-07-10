import time
import subprocess
import sys

print("Monitoring system for CUDA GPU availability...")
print("Waiting for both PyTorch installation and NVIDIA CUDA Toolkit installation to finish...")

while True:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nCUDA IS READY! Found GPU: {torch.cuda.get_device_name(0)}")
            break
    except Exception:
        pass
    time.sleep(10)

print("Starting training manager automatically...")
subprocess.run([sys.executable, "train_manager.py"])
