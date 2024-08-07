# hash:sha256:b1a1d287f7b378b17127b6a85004e6b108a021db131f9ee0ca080ed1a59431f1
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
    numpy==1.26.4 \
    pandas==2.2.2 \
    skeleton-plot==0.0.10
