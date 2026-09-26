#!/usr/bin/env bash
# The body every Kernels/<domain>/build.sh shares: toolchain resolution, compilation, the PTX ISA check and the
# install step. A domain script declares its kernel lists, sources this file and calls build_all "$@".
#
#   ./build.sh                    # compile + install every kernel in the domain's lists into ../../Ptx
#   ./build.sh --no-install       # compile only (artifacts stay beside the .cu)
#   ./build.sh --arch sm_120a     # also build the domain's ARCH_VARIANTS for that arch, as <kernel>.sm120.ptx
#   ./build.sh --install-tuned    # also install the TUNED kernels (see lm/build.sh)
#
# Lists a domain script may set before build_all:
#   KERNELS        compiled for sm_80, the fleet baseline; PTX is JIT-forward-compatible from there
#   KERNELS_SM75   compiled for sm_75 (GGUF dequant keeps Turing)
#   KERNELS_SM80   compiled for sm_80 (alongside KERNELS_SM75; a domain uses one style or the other)
#   ARCH_VARIANTS  kernels with arch-conditional device code; --arch builds each as a suffixed variant that
#                  CudaKernels.PtxPath loads only on that exact compute capability
#   TUNED          kernels whose shipped PTX is hand-tuned and not overwritten unless --install-tuned is passed
#
# Uses nvcc when it is on PATH, else the committed ../nvrtc_compile frontend (needs $CUDA_LIB and the CUDA
# headers, $CUDA_INC). The emitted PTX must say ".version 9.0": a newer toolchain emits 9.3, which the fleet
# driver refuses to JIT (TROUBLESHOOTING §CUDA/PTX).

set -euo pipefail

PTX_OUT="${THIS_DIR}/../../Ptx"
NVRTC="${THIS_DIR}/../nvrtc_compile"
CUDA_LIB="${CUDA_LIB:-${HOME}/.local/lib/cuda13}"
# The headers must be a COMPLETE set: mma.h (attention) pulls crt/mma.h, which the lib-adjacent include dir
# lacks. Prefer the first candidate that has it, so a partial set degrades to a clear error rather than a stale PTX.
if [[ -z "${CUDA_INC:-}" ]]; then
    for _cand in "${HOME}/.local/cuda-tools/nvidia/cu13/include" "${CUDA_LIB}/include"; do
        if [[ -f "${_cand}/crt/mma.h" ]]; then CUDA_INC="$_cand"; break; fi
    done
    CUDA_INC="${CUDA_INC:-${CUDA_LIB}/include}"
fi

KERNELS=("${KERNELS[@]:-}")
KERNELS_SM75=("${KERNELS_SM75[@]:-}")
KERNELS_SM80=("${KERNELS_SM80[@]:-}")
ARCH_VARIANTS=("${ARCH_VARIANTS[@]:-}")
TUNED=("${TUNED[@]:-}")

# compile_one <kernel> <arch>   arch is the bare number nvcc's sm_ and nvrtc's compute_ take (80, 75, 120a);
# a family-specific arch (trailing letter) lands as <kernel>.sm<digits>.ptx, the baseline as <kernel>.ptx.
compile_one() {
    local kernel="$1" arch="$2"
    local src="${THIS_DIR}/${kernel}.cu"
    local suffix=""
    if [[ "$arch" =~ [a-z]$ ]]; then suffix=".sm${arch%[a-z]}"; fi
    local ptx="${THIS_DIR}/${kernel}${suffix}.ptx"
    if [[ ! -f "$src" ]]; then
        echo "missing source: $src" >&2
        exit 1
    fi
    if command -v nvcc >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_${arch} ${kernel}.cu"
        nvcc -ptx -arch="sm_${arch}" "$src" -o "$ptx"
    else
        if [[ ! -x "$NVRTC" ]]; then
            echo "no nvcc on PATH and no nvrtc helper — build it: cc -O2 -o $NVRTC ${NVRTC}.c -ldl" >&2
            exit 1
        fi
        echo "[$(date +%H:%M:%S)] nvrtc_compile compute_${arch} ${kernel}.cu"
        # The kernel's own directory is an include path too: NVRTC compiles from a string and resolves
        # #include "x.cuh" only through the paths it is given.
        LD_LIBRARY_PATH="$CUDA_LIB" "$NVRTC" "$src" "$ptx" "compute_${arch}" "$CUDA_INC" "${THIS_DIR}"
    fi
    if ! head -20 "$ptx" | grep -q '^\.version 9\.0$'; then
        echo "ERROR: ${kernel}${suffix}.ptx is not PTX ISA 9.0 (driver JIT ceiling) — check toolchain pin." >&2
        exit 1
    fi
    if $INSTALL; then
        local is_tuned=false
        for t in "${TUNED[@]}"; do [[ -n "$t" && "$kernel" == "$t" ]] && is_tuned=true; done
        if $is_tuned && ! $INSTALL_TUNED; then
            echo "  · ${kernel}: compiled but NOT installed (hand-tuned; pass --install-tuned with a perf run)"
        else
            cp "$ptx" "${PTX_OUT}/${kernel}${suffix}.ptx"
            echo "  → ${PTX_OUT}/${kernel}${suffix}.ptx"
        fi
    fi
}

build_all() {
    INSTALL=true
    INSTALL_TUNED=false
    local extra_arch=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-install)    INSTALL=false; shift ;;
            --install-tuned) INSTALL_TUNED=true; shift ;;
            --arch)          extra_arch="${2#sm_}"; shift 2 ;;
            *) echo "unknown argument: $1" >&2; exit 2 ;;
        esac
    done
    for kernel in "${KERNELS[@]}";      do [[ -n "$kernel" ]] && compile_one "$kernel" 80; done
    for kernel in "${KERNELS_SM75[@]}"; do [[ -n "$kernel" ]] && compile_one "$kernel" 75; done
    for kernel in "${KERNELS_SM80[@]}"; do [[ -n "$kernel" ]] && compile_one "$kernel" 80; done
    if [[ -n "$extra_arch" ]]; then
        if [[ ! "$extra_arch" =~ [a-z]$ ]]; then
            echo "ERROR: --arch takes a family-specific arch (sm_120a, sm_100a): plain sm_${extra_arch} PTX would replace nothing and load nowhere the baseline does not." >&2
            exit 2
        fi
        for kernel in "${ARCH_VARIANTS[@]}"; do [[ -n "$kernel" ]] && compile_one "$kernel" "$extra_arch"; done
    fi
    echo "done."
}
