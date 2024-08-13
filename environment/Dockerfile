# hash:sha256:695107804f7d2d84577bbfb22f9a2c8fc03decac7e035e2cafa6e8a9d3a40a5e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    caveclient==5.24.0 \
    cloud-volume==10.2.2 \
    ipywidgets==8.1.3 \
    k3d==2.16.1 \
    matplotlib==3.9.0 \
    nglui==3.3.7 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    skeleton-plot==0.0.10

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
