# hash:sha256:ae55f0687155e01008cc441889ea164b46864881b1a9b456448d2f90734dcc7f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/py-r:python3.10.12-R4.2.3-IRkernel-ubuntu22.04

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

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

COPY postInstall /
RUN /postInstall