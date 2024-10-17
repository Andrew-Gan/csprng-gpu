import sys
import torch
import torchcsprng as csprng
import time

def test_aes(device: str):
    key = torch.empty(16, dtype=torch.uint8, device=device).random_(0, 256)

    for i in range(19, 24):
        if i > 19:
            print('2^' + str(i) + ' blocks')
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
            print("%s %.2f ms" % (dev, (end - start) * 1000))

def test_prng(device: str):
    urandom_gen = csprng.create_random_device_generator('/dev/urandom')
    for i in range(20, 25):
        start = time.time()
        torch.randn(1 << i, device='cpu', generator=urandom_gen)
        mid = time.time()
        torch.randn(1 << i, device='cuda', generator=urandom_gen)
        end = time.time()

        if i > 20:
            print("cpu %.2f ms" % ((mid - start) * 1000))
            print("cuda %.2f ms" % ((end - mid) * 1000))

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Current CUDA device', torch.cuda.current_device())
    else:
        sys.exit('No CUDA device found')
    for dev in ['cpu', 'cuda']:
        print('using', dev)
        test_aes(dev)