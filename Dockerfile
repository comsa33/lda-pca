FROM python:3.9.16-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

COPY requirements.txt /usr/src/

WORKDIR /usr/src

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install fontconfig && apt-get install unzip
RUN curl -o nanumfont.zip http://cdn.naver.com/naver/NanumFont/fontfiles/NanumFont_TTF_ALL.zip 
RUN unzip -d /usr/share/fonts/nanum nanumfont.zip
RUN fc-cache -f -v
RUN pip install -r requirements.txt

COPY . /usr/src/LDA-PCA
WORKDIR /usr/src/LDA-PCA/

EXPOSE 8501

CMD ["streamlit", "run", "pca-tsne.py"]