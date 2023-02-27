import torch
import torch.optim as optim

scaler = torch.cuda.amp.GradScaler()##这个写到训练的最前面

optimizer = optim
model = torch

optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
scaler.step(optimizer)
scaler.update()