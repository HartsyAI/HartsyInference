#!/usr/bin/env bash
# The body every Kernels/<domain>/build.sh shares: toolchain resolution, compilation, the PTX ISA check and the
# install step. A domain script declares its kernel lists, sources this file and calls build_all "$@".
#
#   ./build.sh                    # compile + install every kernel in the domain's lists into ../../Ptx
#   ./build.sh --no-install       # compile only (artifacts stay beside the .cu)
#   ./build.sh --arch sm_120a     # also build the domain's ARCH_VARIANTS for that arch, as <kernel>.sm120.ptx
#
# Lists a domain script may set before build_all:
#   KERNELS        compiled for sm_80, the fleet baseline; PTX is JIT-forward-compatible from there
#   KERNELS_SM75   compiled for sm_75 (GGUF dequant keeps Turing)
#   KERNELS_SM80   compiled for sm_80 (alongside KERNELS_SM75; a domain uses one style or the other)
#   ARCH_VARIANTS  kernels with arch-conditional device code; --arch builds each as a suffixed variant that
#                  CudaKernels.PtxPath loads only on that exact compute capability
#   NVCC_BUILT     kernels whose shipped PTX is CUDA 13.0.88 nvcc output rather than nvrtc output
#
# Each kernel has one compiler. nvcc and nvrtc of the same release stamp the same PTX header but still emit different
# PTX (instruction selection, register numbering), so taking whichever is on PATH would rewrite benchmarked kernels.
# NVCC_BUILT kernels use $HARTSY_NVCC, else nvcc on PATH, else ~/.local/cuda-tools-13.0; without the 13.0.88 release
# at ISA 9.0 they are reported UNVERIFIED and left as shipped. Every other kernel uses the committed ../nvrtc_compile
# frontend (needs $CUDA_LIB and the CUDA headers, $CUDA_INC). The emitted PTX must say ".version 9.0": a newer
# toolchain emits 9.3, which the fleet driver refuses to JIT (TROUBLESHOOTING §CUDA/PTX).

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
NVCC="${HARTSY_NVCC:-$(command -v nvcc || true)}"
# The pinned pip --target install the workstation keeps, when nothing else names an nvcc.
if [[ -z "$NVCC" && -x "${HOME}/.local/cuda-tools-13.0/nvidia/cu13/bin/nvcc" ]]; then
    NVCC="${HOME}/.local/cuda-tools-13.0/nvidia/cu13/bin/nvcc"
fi
# NVCC_BUILT kernels are this release's output; another nvcc release emits different PTX even at ISA 9.0.
NVCC_RELEASE="release 13.0, V13.0.88"
# A pip-installed nvcc can ship without CCCL, which cuda_fp16.h includes (<nv/target>); borrow the nvrtc header set's.
NVCC_FLAGS=()
if [[ -n "$NVCC" ]]; then
    _nvcc_inc="$(dirname "$NVCC")/../include"
    if [[ ! -e "${_nvcc_inc}/nv/target" && ! -e "${_nvcc_inc}/cccl/nv/target" ]]; then
        for _cand in "${CUDA_INC}/cccl" "${CUDA_INC}"; do
            if [[ -e "${_cand}/nv/target" ]]; then NVCC_FLAGS=("-I${_cand}"); break; fi
        done
    fi
fi

KERNELS=("${KERNELS[@]:-}")
KERNELS_SM75=("${KERNELS_SM75[@]:-}")
KERNELS_SM80=("${KERNELS_SM80[@]:-}")
ARCH_VARIANTS=("${ARCH_VARIANTS[@]:-}")
NVCC_BUILT=("${NVCC_BUILT[@]:-}")

# PTX ISA version a file declares in its header, and whether it is at or below the fleet driver's JIT ceiling (9.0).
ptx_version() { head -20 "$1" | sed -n 's/^\.version \([0-9.]*\)$/\1/p' | head -1; }
ptx_isa_ok() { local v; v="$(ptx_version "$1")"; [[ -n "$v" && "$(printf '%s\n' "$v" 9.0 | sort -V | tail -1)" == "9.0" ]]; }

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
    if listed "$kernel" "${NVCC_BUILT[@]}"; then
        # Never an nvrtc stand-in, and an UNVERIFIED kernel leaves any earlier intermediate as it was.
        if [[ -z "$NVCC" ]]; then
            echo "UNVERIFIED ${kernel}${suffix}: nvcc-built and no nvcc found; set HARTSY_NVCC to the CUDA 13.0.88 nvcc"
            return 0
        fi
        echo "[$(date +%H:%M:%S)] nvcc -ptx -arch=sm_${arch} ${kernel}.cu"
        local tmp
        tmp="$(mktemp --suffix=.ptx)"   # GNU mktemp; the PTX build runs on Linux
        if ! "$NVCC" -ptx -arch="sm_${arch}" ${NVCC_FLAGS[@]+"${NVCC_FLAGS[@]}"} "$src" -o "$tmp"; then
            rm -f "$tmp"
            exit 1
        fi
        if ! ptx_isa_ok "$tmp" || ! grep -qF "// Cuda compilation tools, ${NVCC_RELEASE}" "$tmp"; then
            echo "UNVERIFIED ${kernel}${suffix}: ${NVCC} emits PTX ISA $(ptx_version "$tmp") from" \
                "\"$(grep -m1 -o 'release [0-9.]*, V[0-9.]*' "$tmp" || true)\"; nvcc-built kernels need CUDA 13.0.88 nvcc and nvvm"
            rm -f "$tmp"
            return 0
        fi
        cat "$tmp" > "$ptx"
        rm -f "$tmp"
    else
        if [[ ! -x "$NVRTC" ]]; then
            if command -v cc >/dev/null 2>&1; then
                cc -O2 -o "$NVRTC" "${NVRTC}.c" -ldl
            else
                echo "no usable nvrtc helper at $NVRTC and no cc to build it" >&2
                exit 1
            fi
        fi
        echo "[$(date +%H:%M:%S)] nvrtc_compile compute_${arch} ${kernel}.cu"
        # The kernel's own directory is an include path too: NVRTC compiles from a string and resolves
        # #include "x.cuh" only through the paths it is given.
        LD_LIBRARY_PATH="$CUDA_LIB" "$NVRTC" "$src" "$ptx" "compute_${arch}" "$CUDA_INC" "${THIS_DIR}"
        if ! ptx_isa_ok "$ptx"; then
            echo "ERROR: ${kernel}${suffix}.ptx is PTX ISA $(ptx_version "$ptx") (driver JIT ceiling is 9.0) — check toolchain pin." >&2
            exit 1
        fi
    fi
    if $INSTALL; then
        cp "$ptx" "${PTX_OUT}/${kernel}${suffix}.ptx"
        echo "  → ${PTX_OUT}/${kernel}${suffix}.ptx"
    fi
}

# listed <kernel> <entry>...: whether the kernel is one of the (non-empty) entries.
listed() {
    local kernel="$1" entry
    shift
    for entry in "$@"; do [[ -n "$entry" && "$entry" == "$kernel" ]] && return 0; done
    return 1
}

# want <kernel>: true when no kernel names were given on the command line, or when this one is named.
want() {
    [[ ${#ONLY[@]} -eq 0 ]] && return 0
    listed "$1" "${ONLY[@]}"
}

build_all() {
    INSTALL=true
    local ONLY=()
    local extra_arch=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-install)    INSTALL=false; shift ;;
            --arch)          extra_arch="${2#sm_}"; shift 2 ;;
            -*) echo "unknown argument: $1" >&2; exit 2 ;;
            *)  ONLY+=("$1"); shift ;;
        esac
    done
    for kernel in "${KERNELS[@]}";      do [[ -n "$kernel" ]] && want "$kernel" && compile_one "$kernel" 80; done
    for kernel in "${KERNELS_SM75[@]}"; do [[ -n "$kernel" ]] && want "$kernel" && compile_one "$kernel" 75; done
    for kernel in "${KERNELS_SM80[@]}"; do [[ -n "$kernel" ]] && want "$kernel" && compile_one "$kernel" 80; done
    if [[ -n "$extra_arch" ]]; then
        if [[ ! "$extra_arch" =~ [a-z]$ ]]; then
            echo "ERROR: --arch takes a family-specific arch (sm_120a, sm_100a): plain sm_${extra_arch} PTX would replace nothing and load nowhere the baseline does not." >&2
            exit 2
        fi
        for kernel in "${ARCH_VARIANTS[@]}"; do [[ -n "$kernel" ]] && compile_one "$kernel" "$extra_arch"; done
    fi
    "${THIS_DIR}/../check_ptx_isa.sh" >/dev/null || { echo "ERROR: a shipped PTX file is not ISA 9.0; run check_ptx_isa.sh" >&2; exit 1; }
    echo "done."
}
