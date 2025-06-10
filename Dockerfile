FROM ubuntu:24.04

RUN apt update && apt install -y pipx python3-pip git
RUN pipx ensurepath
RUN pipx install poetry dvc[s3]
ENV PATH=/root/.local/bin:$PATH

RUN apt update && apt install -y ffmpeg libsm6 libxext6

RUN git clone https://github.com/dfbakin/colorizer.git colorizer
WORKDIR /colorizer

RUN git checkout deploy-tg-bot

RUN poetry install --no-root
RUN dvc pull checkpoints/baseline.dvc

WORKDIR /colorizer
# docker run -e TELEGRAM_BOT_TOKEN=your_token_here your_image

CMD ["poetry", "run", "python", "deploy_tg_bot.py", "$TELEGRAM_BOT_TOKEN"]