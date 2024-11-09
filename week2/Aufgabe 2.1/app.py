import lightning as L
import torch

print(f"Is Cuda available: {torch.cuda.is_available()}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.get_device_name(device=device))