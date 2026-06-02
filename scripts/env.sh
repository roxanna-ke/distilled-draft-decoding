#!/usr/bin/env bash
# Shared runtime environment for RunAI scripts.
# Prefers the vLLM 0.22.0 venv created under /scratch over the course image's
# preinstalled Python/vLLM 0.11.0. Override KDSD_VENV or KDSD_PYTHON if needed.

_kdsd_default_venv="/scratch/venvs/kdsd-vllm"
KDSD_VENV="${KDSD_VENV:-${_kdsd_default_venv}}"

if [[ -x "${KDSD_VENV}/bin/python" ]]; then
  export VIRTUAL_ENV="${KDSD_VENV}"
  export PATH="${KDSD_VENV}/bin:${PATH}"
  export KDSD_PYTHON="${KDSD_PYTHON:-${KDSD_VENV}/bin/python}"

  _kdsd_site="${KDSD_VENV}/lib/python3.12/site-packages"
  _kdsd_ld_path="${_kdsd_site}/nvidia/cu13/lib:${_kdsd_site}/nvidia/cu13/cccl/lib:${_kdsd_site}/nvidia/nvjitlink/lib:${_kdsd_site}/nvidia/cuda_nvrtc/lib:${_kdsd_site}/nvidia/cuda_runtime/lib:${_kdsd_site}/nvidia/cuda_cupti/lib:${_kdsd_site}/nvidia/cublas/lib:${_kdsd_site}/nvidia/cusparse/lib:${_kdsd_site}/nvidia/cufft/lib:${_kdsd_site}/nvidia/cudnn/lib:${_kdsd_site}/nvidia/cusolver/lib:${_kdsd_site}/nvidia/nccl/lib:${_kdsd_site}/nvidia/nvshmem/lib:${_kdsd_site}/nvidia/nvtx/lib:${_kdsd_site}/nvidia/curand/lib:${_kdsd_site}/nvidia/cufile/lib:${_kdsd_site}/nvidia/cusparselt/lib:${_kdsd_site}/torch/lib"
  export LD_LIBRARY_PATH="${_kdsd_ld_path}:${LD_LIBRARY_PATH:-}"
else
  export KDSD_PYTHON="${KDSD_PYTHON:-python}"
  echo "WARNING: KDSD venv not found at ${KDSD_VENV}; using ${KDSD_PYTHON}" >&2
fi

export PYTHON="${KDSD_PYTHON}"
