FROM nvidia/cuda:12.0.1-base-ubuntu22.04

# Install dependencies
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt update && apt install -y pipx python3-pip git ffmpeg libsm6 libxext6
RUN pipx ensurepath
RUN pipx install dvc[s3]
ENV PATH=/root/.local/bin:$PATH

RUN pip3 install torch opencv-python numpy albumentations torchvision python-telegram-bot \
    fastai torchvision boto3 wandb

# Clone the repository
RUN git clone https://github.com/dfbakin/colorizer.git colorizer
WORKDIR /colorizer

# Pull checkpoints and install dependencies
RUN dvc pull checkpoints/baseline.dvc

# Set the entrypoint
CMD ["sh", "-c", "python3 deploy_tg_bot.py $TELEGRAM_BOT_TOKEN"]
