# hash:sha256:fcd0b103c18596e360b53a2e667b86361c088e655c1253c7ec1053b8c6752423
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG API_KEY
ARG API_SECRET
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    caveclient \
    cloud-volume \
    ipywidgets \
    k3d \
    matplotlib \
    numpy \
    pandas \
    skeleton-plot
