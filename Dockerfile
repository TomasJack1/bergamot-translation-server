FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*  # 清理缓存减小镜像体积 

WORKDIR /bergamot-translation-server

COPY ./requirements.txt /bergamot-translation-server/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /bergamot-translation-server/requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/"

COPY ./ /bergamot-translation-server

CMD [ "fastapi", "run" , "main.py", "--port", "8080"]