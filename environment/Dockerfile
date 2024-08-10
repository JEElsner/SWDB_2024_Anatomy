# hash:sha256:6c97d1dbc5465bdec73485c93afcb509b1b3d54d599636d40ec96adec7daeaaa
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    brainglobe_atlasapi \
    caveclient==5.24.0 \
    cloud-volume==10.2.2 \
    ipywidgets==8.1.3 \
    k3d==2.16.1 \
    matplotlib==3.9.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    skeleton-plot==0.0.10

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
