

import time
import os
import torch
import torch.distributed as dist
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

start_time = time.time()

class Part1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 500), nn.ReLU())

    def forward(self, x):
        return self.seq(x)

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    dist.init_process_group(backend='gloo', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)

    transform = transforms.ToTensor()
    dataset = datasets.MNIST('/tmp/mnist', download=True, train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)    

    model = Part1()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()

    for step, (images, labels) in enumerate(dataloader):
        if step >= 1800:
            break
        output = model(images)
        dist.send(output, dst=1)
        dist.send(labels, dst=1)
        
        grad_output = torch.zeros_like(output)
        dist.recv(grad_output, src=1)

        output.backward(grad_output)
        optimizer.step()
        optimizer.zero_grad()
        print(f"[Part1] Step {step} done")
        dist.barrier()
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"[Part1] Total training time: {total_time:.2f} seconds")

    torch.save(model.state_dict(), f'/workspace/models/part1.pth')

if __name__ == '__main__':
    main()
    



