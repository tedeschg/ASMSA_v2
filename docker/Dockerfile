# CUDA-enabled GROMACS 2026.3 with the PLUMED 2.11 PyTorch module.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG JOBS=8
ARG GMX_GPU=CUDA
ARG CUDA_ARCH=75
ARG LIBTORCH_VERSION=2.5.1
ARG PLUMED_REF=master
ARG GMX_VERSION=2026.3
ARG GMX_PATCH=gromacs-2026.0

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CC=/usr/bin/gcc-12 \
    CXX=/usr/bin/g++-12 \
    CUDAHOSTCXX=/usr/bin/g++-12

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc-12 g++-12 \
        git wget unzip ca-certificates patch xxd jq \
        python3 python3-pip python-is-python3 \
        libfftw3-dev liblapack-dev libblas-dev zlib1g-dev libgsl-dev \
        libhwloc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir cmake

WORKDIR /software

# PLUMED's PyTorch module uses the same LibTorch release as the Python package.
RUN mkdir "libtorch-${LIBTORCH_VERSION}" \
    && cd "libtorch-${LIBTORCH_VERSION}" \
    && wget -q -O libtorch.zip \
       "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip" \
    && unzip -q libtorch.zip \
    && rm libtorch.zip

ENV LIBTORCH=/software/libtorch-${LIBTORCH_VERSION}/libtorch

RUN git clone https://github.com/plumed/plumed2.git /software/plumed2 \
    && cd /software/plumed2 \
    && git checkout ${PLUMED_REF} \
    && ./configure --enable-modules=+pytorch \
                   --enable-libtorch \
                   --prefix=/software/plumed-install \
                   CXXFLAGS="-std=c++17" \
                   LDFLAGS="-L${LIBTORCH}/lib -Wl,-rpath,${LIBTORCH}/lib" \
                   CPPFLAGS="-I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include" \
    && make -j${JOBS} \
    && make install \
    && cd / \
    && rm -rf /software/plumed2

ENV PLUMED_PREFIX=/software/plumed-install
ENV PATH=${PLUMED_PREFIX}/bin:${PATH} \
    LD_LIBRARY_PATH=${PLUMED_PREFIX}/lib:${LD_LIBRARY_PATH} \
    PKG_CONFIG_PATH=${PLUMED_PREFIX}/lib/pkgconfig \
    PLUMED_KERNEL=${PLUMED_PREFIX}/lib/libplumedKernel.so

RUN plumed --no-mpi info --version \
    && plumed --no-mpi config show | grep -E "has libtorch on" \
    && plumed --no-mpi config show | grep -E "module pytorch on" \
    && ldd "${PLUMED_KERNEL}" | grep -E "libtorch_cpu|libc10" \
    && ! ldd "${PLUMED_KERNEL}" | grep "not found"

RUN wget -q "https://ftp.gromacs.org/gromacs/gromacs-${GMX_VERSION}.tar.gz" \
    && tar xzf "gromacs-${GMX_VERSION}.tar.gz" \
    && rm "gromacs-${GMX_VERSION}.tar.gz" \
    && cd "/software/gromacs-${GMX_VERSION}" \
    && plumed patch -p --runtime -e "${GMX_PATCH}"

RUN set -e; \
    cd "/software/gromacs-${GMX_VERSION}" && mkdir build && cd build; \
    if [ "${GMX_GPU}" = "CUDA" ]; then \
        GPU_FLAGS="-DGMX_GPU=CUDA -DCMAKE_CUDA_HOST_COMPILER=${CXX} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"; \
    else \
        GPU_FLAGS="-DGMX_GPU=OFF"; \
    fi; \
    cmake .. \
        -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DREGRESSIONTEST_DOWNLOAD=OFF \
        -DGMX_MPI=OFF \
        ${GPU_FLAGS} \
        -DCMAKE_INSTALL_PREFIX=/software/gromacs-${GMX_VERSION}-plumed; \
    make -j${JOBS}; \
    make install; \
    cd / && rm -rf "/software/gromacs-${GMX_VERSION}"

ENV GMX_PREFIX=/software/gromacs-${GMX_VERSION}-plumed
ENV PATH=${GMX_PREFIX}/bin:${PATH} \
    LD_LIBRARY_PATH=${GMX_PREFIX}/lib:${LD_LIBRARY_PATH} \
    GMXDATA=${GMX_PREFIX}/share/gromacs

RUN gmx --version | grep -i -E "GROMACS version|GPU support" \
    && gmx mdrun -h 2>&1 | grep -i -E "^ *-plumed"

# Install Python after the simulation libraries so its CUDA libraries are not
# shadowed by PLUMED's separate CPU-only LibTorch runtime.
RUN pip install --no-cache-dir \
        torch==${LIBTORCH_VERSION} --index-url https://download.pytorch.org/whl/cu124

COPY . /opt/asmsa_v2
RUN pip install --no-cache-dir '/opt/asmsa_v2[full]'

RUN install -m 755 /opt/asmsa_v2/docker/entrypoint.sh /usr/local/bin/entrypoint.sh \
    && echo 'source ${GMX_PREFIX}/bin/GMXRC' >> /etc/bash.bashrc

WORKDIR /workspace
EXPOSE 8888

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
