import sys
import torch
import torchcsprng as csprng
import time

from torch import nn, optim
from torch.nn.functional import *
from torchvision import datasets, transforms
from opacus import PrivacyEngine

def test_aes(device: str):
    key = torch.empty(16, dtype=torch.uint8, device=device).random_(0, 256)

    for i in range(19, 24):
        size_bytes = (1 << i) * 16
        initial = torch.empty(size_bytes // 4, dtype=torch.float32, device=device).normal_(-24.0, 42.0)

        dev = torch.device(device)
        encrypted = torch.empty(size_bytes // 8, dtype=torch.int64, device=device)
        decrypted = torch.empty_like(initial)

        start = time.time()
        csprng.encrypt(initial, encrypted, key, "aes128", "ecb")
        torch.cuda.synchronize()
        end = time.time()
        csprng.decrypt(encrypted, decrypted, key, "aes128", "ecb")
        torch.cuda.synchronize()
        assert(decrypted == initial).all()

        if i > 19:
            print(f'2^{i}: {dev} {(end-start)*1000:.2f} ms')

def test_csprng(device: str):
    urandom_gen = csprng.create_random_device_generator('/dev/urandom')
    for i in range(19, 24):
        start = time.time()
        torch.randn(1 << i, device=device, generator=urandom_gen)
        end = time.time()

        if i > 19:
            print(f'2^{i}: {dev} {(end-start)*1000:.2f} ms')

def test_dp(device: str):
    class Net(nn.Module):      
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 10)
            
        def forward(self, x):
            x = max_pool2d(relu(self.conv1(x)), (2,2))
            x = max_pool2d(relu(self.conv2(x)), 2)    
            x = x.view(x.size(0), -1)
            x = relu(self.fc1(x))
            x = self.fc2(x)
            
            return x
    
    dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f'Current CUDA device {torch.cuda.current_device()}\n')
    else:
        sys.exit('No CUDA device found')

    fname = ['aes', 'csprng', 'dp']
    for name, test_func in zip(fname, [test_aes, test_csprng, test_dp]):
        print(f'Testing {name}')
        for dev in ['cpu', 'cuda']:
            print(f'running on {dev}')
            test_func(dev)