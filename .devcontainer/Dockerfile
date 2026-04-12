FROM nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04

# config user and bash completion
RUN apt-get update && apt-get install -y sudo bash-completion \
    && echo "ubuntu ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

RUN apt-get update && apt-get install -y build-essential ninja-build cmake git clangd
