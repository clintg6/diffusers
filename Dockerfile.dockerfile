ARG BASE_IMAGE="rocm/vllm-dev:base"

FROM $BASE_IMAGE

RUN echo "Base image is $BASE_IMAGE"


ENV LLVM_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
ENV PATH=$PATH:/opt/rocm/bin:/libtorch/bin:
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/:/libtorch/lib:
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/libtorch/include:/libtorch/include/torch/csrc/api/include/:/opt/rocm/include/:
ENV USE_ROCM="1"
ENV MAX_JOBS="200"
ENV USE_CK_FLASH_ATTENTION="1"

### Mount Point ###
# When launching the container, mounts to APP_MOUNT
ARG APP_MOUNT=/app
VOLUME [ ${APP_MOUNT} ]
WORKDIR ${APP_MOUNT}

RUN pip uninstall -y torch torchvision flash-attn
# A commit to fix the output scaling factor issue in _scaled_mm
# Not yet in 2.5.0-rc1
ARG PYTORCH_VISION_BRANCH="v0.20.1"
ARG PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
#RUN --mount=type=bind,from=export_hipblaslt,src=/,target=/install \
#if ls /install/*.deb; then \
#    apt-get purge -y hipblaslt \
#    && dpkg -i /install/*.deb \
#    && sed -i 's/, hipblaslt-dev \(.*\), hipcub-dev/, hipcub-dev/g' /var/lib/dpkg/status \
#    && sed -i 's/, hipblaslt \(.*\), hipfft/, hipfft/g' /var/lib/dpkg/status; \
#fi
RUN git clone ${PYTORCH_REPO} pytorch \
    && cd pytorch && git submodule update --init --recursive \
    && python tools/amd_build/build_amd.py \
    && CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl \
    && cd .. \
    && git clone ${PYTORCH_VISION_REPO} vision \
    && cd vision \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl

RUN python3 -m pip install --upgrade huggingface-hub[cli] diffusers ipython

ENV TOKENIZERS_PARALLELISM=false
ENV HIP_FORCE_DEV_KERNARG=1

WORKDIR /app
