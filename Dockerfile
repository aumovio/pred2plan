FROM nvcr.io/nvidia/pytorch:24.04-py3 AS base

# # Install rust packaging systems as necessary for av2
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# ENV PATH="$HOME/.cargo/bin:${PATH}"

# Install opencv, dos2unix
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Berlin"

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y build-essential \
    make \
    git \
    tmux \
    dos2unix \
    python3-opencv \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    curl \
    wget \
    tk-dev \
    ffmpeg \
    gfortran \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libmetis-dev \
    gdb \
    clang \
    && apt-get purge -y imagemagick imagemagick-6-common \
    && apt-get clean && apt-get autoclean && apt-get autoremove 
RUN mkdir /app && chgrp users /app

RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz \
    && tar -xzf Python-3.11.9.tgz \
    && cd Python-3.11.9 \
    && ./configure --enable-optimizations \
    && make install

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# # Install rust version compatible to av2
# RUN rustup default nightly-2024-02-04

# Install HSL solvers for casadi for use of faster Ma57 linear solver
COPY ThirdParty-HSL /ThirdParty-HSL
RUN cd ./ThirdParty-HSL && ./configure --prefix=/usr/local && \
    make -j"$(nproc)" && \
    make install && \
    make clean || true
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local-lib.conf && ldconfig && \
    bash -lc '\
      set -e; \
      cd /usr/local/lib; \
      # create libhsl.so -> libcoinhsl.so and mirror versioned symlinks if present
      if [ -e libcoinhsl.so ]; then ln -sfn libcoinhsl.so libhsl.so; fi; \
      for v in libcoinhsl.so.*; do \
        [ -e "$v" ] || continue; \
        ln -sfn "$v" "libhsl.so.${v##*.so.}"; \
      done \
    '
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH 
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH 

# install requirements for app
COPY --chown=root:users requirements.txt /app/requirements.txt
COPY --chown=root:users requirements_nuplan.txt /app/requirements_nuplan.txt
COPY --chown=root:users requirements_no_deps.txt /app/requirements_no_deps.txt
RUN pip install --upgrade setuptools wheel pip && \
    pip install --no-input -r /app/requirements_nuplan.txt --no-cache-dir && \
    pip install --no-input -r /app/requirements.txt --no-cache-dir && \
	pip install --no-input -r /app/requirements_no_deps.txt --no-deps --no-cache-dir && \
    pip install torch_geometric && \
    pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
    
FROM base AS build

# Install project libs & scripts
# COPY --chown=root:users docker /app/docker
COPY --chown=root:users entrypoint /entrypoint
RUN dos2unix /entrypoint

COPY --chown=root:users . /app/pred2plan
WORKDIR /app

RUN chmod o+rx /entrypoint
RUN chmod -R o+rx /app

ENV PYTHONPATH /app:$PYTHONPATH

CMD ["/entrypoint"]
