FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG USERNAME=dockeruser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y software-properties-common && \
    apt-add-repository ppa:fish-shell/release-3 && \
    apt install -y build-essential wget curl tar ripgrep git tmux fish time cmake

RUN mkdir -p /tmp/nvim_download && \
    curl -sL https://github.com/neovim/neovim/releases/download/stable/nvim-linux64.tar.gz \
        | tar xzf - -C /tmp/nvim_download && \
    mv /tmp/nvim_download/nvim-linux64/bin/nvim /usr/local/bin/nvim && \
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
        | bash -s -- --to /usr/local/bin


# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /usr/bin/fish \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Setup dotfiles
ADD https://api.github.com/repos/andylolu2/dotfiles/git/refs/heads/main /tmp/version.json
RUN git clone https://github.com/andylolu2/dotfiles $HOME/.dotfiles && \
    $HOME/.dotfiles/main.fish
