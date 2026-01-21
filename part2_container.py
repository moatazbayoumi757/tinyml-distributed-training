import os
import torch
import torch.distributed as dist
from torch import nn
import time


start_time = time.time()

class Part2(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(500, 100), nn.ReLU(), nn.Linear(100, 10))

    def forward(self, x):
        return self.seq(x)

def main():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    dist.init_process_group(backend='gloo', init_method=f'tcp://{master_addr}:{master_port}', rank=rank, world_size=world_size)

    model = Part2()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()

    for step in range(1800):
        input_tensor = torch.zeros(32, 500, requires_grad=True)
        labels = torch.zeros(32, dtype=torch.long)
        
        dist.recv(input_tensor, src=0)
        dist.recv(labels, src=0)
        
        output = model(input_tensor)

        loss = criterion(output, labels)
        loss.backward()
        dist.send(input_tensor.grad, dst=0)
        optimizer.step()
        optimizer.zero_grad()
        print(f"[Part2] Step {step} done, loss: {loss}")
        dist.barrier()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[Part1] Total training time: {total_time:.2f} seconds")



    torch.save(model.state_dict(), f'/workspace/models/part2.pth')
    
    print(f"Final loss: {loss}")


if __name__ == '__main__':
    main()
