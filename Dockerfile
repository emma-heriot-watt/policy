ARG BUILDER_IMAGE_NAME
ARG BASE_IMAGE_NAME

# ---------------------------------- Builder --------------------------------- #
# hadolint ignore=DL3006
FROM ${BUILDER_IMAGE_NAME} as builder

ARG TORCH_VERSION_SUFFIX=""

WORKDIR ${PYSETUP_PATH}/repo

COPY . ${PYSETUP_PATH}/repo

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN poetry install --only main,web \
	&& TORCH_VERSION="$(pip show torch | grep Version | cut -d ':' -f2 | xargs)${TORCH_VERSION_SUFFIX}" \
	&& TORCHVISION_VERSION="$(pip show torchvision | grep Version | cut -d ':' -f2 | xargs)${TORCH_VERSION_SUFFIX}" \
	&& pip install --no-cache-dir torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" -f https://download.pytorch.org/whl/torch_stable.html

# ---------------------------------- Runner ---------------------------------- #
# hadolint ignore=DL3006
FROM ${BASE_IMAGE_NAME} as runner

COPY --from=builder ${PYSETUP_PATH} ${PYSETUP_PATH}

WORKDIR ${PYSETUP_PATH}/repo

# Set the PYTHONPATH
ENV PYTHONPATH='./src'

ENTRYPOINT ["/bin/bash"]
