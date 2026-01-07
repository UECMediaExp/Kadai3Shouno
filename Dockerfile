# ベースイメージはとりあえず 3.13 の slim で．(2025年末版)
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# 必要パッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    sudo \
    python3 \
    python3-pip \
    python3-venv \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 一般ユーザを vscode に
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# bash as default shell
RUN chsh -s /bin/bash ${USERNAME}
COPY --chown=${USERNAME}:${USERNAME} .bashrc /home/${USERNAME}/.bashrc
COPY --chown=${USERNAME}:${USERNAME} .tmux.conf /home/${USERNAME}/.tmux.conf

# install Python library
#RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
COPY --chown=${USERNAME}:${USERNAME} requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# config workspace
USER $USERNAME
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
WORKDIR /workspace


