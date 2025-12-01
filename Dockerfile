FROM ghcr.io/osgeo/gdal:ubuntu-full-3.7.3

# Don't use old pygeos
ENV USE_PYGEOS=0

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libpq-dev \
    ca-certificates \
    build-essential \
    libcairo2-dev\
    # odc-algo now requires rust
    rustc \
    cargo \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

RUN pip3 install --upgrade pip setuptools wheel
ADD requirements.txt /tmp/requirements.txt
RUN pip3 --no-cache-dir install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip3 --no-cache-dir install -r /tmp/requirements.txt

ADD . /code

WORKDIR /code
