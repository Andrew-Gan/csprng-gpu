FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y bash g++ python3 python3-pip
RUN pip3 install torch
ENV TORCH_CUDA_ARCH_LIST=8.6
ENV FORCE_CUDA=1

COPY ./csprng /home/csprng
WORKDIR /home/csprng

RUN python3 setup.py install

COPY test.py /home
WORKDIR /home

CMD python3 test.py