FROM ghcr.io/pyo3/maturin

WORKDIR /
RUN yum install -y openssl-devel

WORKDIR /io
ENTRYPOINT ["/usr/bin/maturin"]
