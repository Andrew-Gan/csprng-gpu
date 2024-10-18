import sys
import torch
import torchcsprng as csprng
import time

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from opacus import PrivacyEngine

training_set = datasets.FashionMNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
validation_set = datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor(), download=True)

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
        csprng.decrypt(encrypted, decrypted, key, "aes128", "ecb")
        torch.cuda.synchronize()

        if i > 19:
            print(f'2^{i}: {dev} {(time.time()-start)*1000:.2f} ms')

        assert(decrypted == initial).all()

def test_csprng(device: str):
    urandom_gen = csprng.create_random_device_generator('/dev/urandom')
    for i in range(19, 24):
        start = time.time()
        torch.randn(1 << i, device=device, generator=urandom_gen)

        if i > 19:
            print(f'2^{i}: {dev} {(time.time()-start)*1000:.2f} ms')

def test_dp_train(device: str):
    class GarmentClassifier(nn.Module):
        def __init__(self):
            super(GarmentClassifier, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = GarmentClassifier()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=8, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=8, shuffle=False)

    privacy_engine = PrivacyEngine(secure_mode=True)
    model, optimizer, training_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=training_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()
    model.train(True)
    running_loss = 0.
    for i, data in enumerate(training_loader):
        if i >= 200:
            break
        data = [d.to(device) for d in data]
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            running_loss = 0.

    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            if i >= 20:
                break
            vdata = [d.to(device) for d in vdata]
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    print(f'{dev} {(time.time()-start)*1000:.2f} ms')

if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f'Current CUDA device {torch.cuda.current_device()}\n')
    else:
        sys.exit('No CUDA device found')

    fname = ['aes', 'csprng', 'dp']
    for name, test_func in zip(fname, [test_aes, test_csprng, test_dp_train]):
        print(f'Testing {name}')
        for dev in ['cpu', 'cuda']:
            print(f'running on {dev}')
            test_func(dev)
