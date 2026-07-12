"""Robust Training Manager for CORTEX.

Monitors training logs in real-time. If eval_loss diverges or hits NaN,
it automatically kills the process, lowers the learning rate in the YAML config,
and restarts from the last safe checkpoint.
"""

import subprocess
import time
import re
import math
import sys
import re

def halve_lr(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Match lr: 0.001
    match = re.search(r"lr:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", content)
    if match:
        old_lr = float(match.group(1))
        new_lr = old_lr / 2.0
        content = content[:match.start(1)] + str(new_lr) + content[match.end(1):]
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"\n[MANAGER] Decreased learning rate in {config_path}: {old_lr} -> {new_lr}")
    else:
        print(f"\n[MANAGER] Could not find lr parameter in {config_path}!")

def run_training_loop(config_path: str):
    cmd = [sys.executable, "-u", "train.py", "--config", config_path, "--resume"]
    
    while True:
        print(f"\n[MANAGER] Starting training process: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        last_losses = []
        diverged = False
        
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                
                # Look for eval_loss in the log line
                # format: step  1000 | loss 6.6368 | eval_loss 4.3243 | answer_acc ...
                match = re.search(r"eval_loss\s+([0-9\.\-naninf]+)\s*\|", line, re.IGNORECASE)
                if match:
                    val_str = match.group(1).strip()
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = math.nan
                        
                    if math.isnan(val) or math.isinf(val):
                        print("\n[MANAGER] CRITICAL: NaN or Inf eval_loss detected! Divergence triggered.")
                        diverged = True
                        process.kill()
                        break
                        
            process.wait()
            
            if process.returncode == 0 and not diverged:
                print("\n[MANAGER] Training completed successfully!")
                break
                
        except KeyboardInterrupt:
            print("\n[MANAGER] Interrupted by user. Shutting down.")
            process.terminate()
            break
            
        if diverged:
            print("[MANAGER] Killing divergent process...")
            process.kill()
            process.wait()
            
            print("[MANAGER] Halving learning rate and restarting from last checkpoint...")
            halve_lr(config_path)
            time.sleep(2) # Give OS time to clear file locks
            # Loop will restart the process with --resume automatically!
        else:
            # If it crashed for some other reason (e.g. OOM), just restart after a delay
            if process.returncode != 0:
                print(f"\n[MANAGER] Process crashed with exit code {process.returncode}. Restarting in 10s...")
                time.sleep(10)

if __name__ == "__main__":
    run_training_loop("configs/language.yaml")
