FROM python:3.9.16-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

COPY requirements.txt /usr/src/

WORKDIR /usr/src

RUN apt update -y && apt upgrade -y
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "pca-stne.py"]