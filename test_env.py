import os
import sys
import torch
from accelerate import Accelerator

print(f"[{os.getpid()}] Starting test_env.py. LOCAL_RANK={os.environ.get('LOCAL_RANK')}, RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

accelerator = Accelerator()
print(f"[{os.getpid()}] Accelerator initialized. process_index={accelerator.process_index}, num_processes={accelerator.num_processes}, device={accelerator.device}")
