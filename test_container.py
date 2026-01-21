import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score

# docker compose run --rm part1 python test_container.py


class Part1(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 500), nn.ReLU())

    def forward(self, x):
        return self.seq(x)

class Part2(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(500, 100), nn.ReLU(), nn.Linear(100, 10))

    def forward(self, x):
        return self.seq(x)

def test():
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST('/tmp/mnist', download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32)

    part1 = Part1()
    part2 = Part2()
    part1.load_state_dict(torch.load("/workspace/models/part1.pth"))
    part2.load_state_dict(torch.load("/workspace/models/part2.pth"))
    part1.eval()
    part2.eval()

    correct = 0
    total = 0
    num_classes = 10
    all_preds = torch.tensor([], dtype=torch.long)
    all_labels = torch.tensor([], dtype=torch.long)

    with torch.no_grad():
        for images, labels in test_loader:
            x = part1(images)
            out = part2(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds = torch.cat((all_preds, preds))
            all_labels = torch.cat((all_labels, labels))

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    f1 = multiclass_f1_score(all_preds, all_labels, num_classes=num_classes, average='macro')

    print(f'F1 Score: {f1:.4f}')

if __name__ == '__main__':
    test()


