import torch

print(torch.cuda.is_available())
print("device name", torch.cuda.get_device_name(0))