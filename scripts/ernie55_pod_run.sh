#!/usr/bin/env bash
# ERNIE-5.5-Flash-Base real-weight run on one rented CUDA pod. Runbook: docs/Checklists/ERNIE55_POD_RUNBOOK.md.
#
#   scripts/ernie55_pod_run.sh preflight    # tools, repo commit, output folder, environment log
#   scripts/ernie55_pod_run.sh download     # 288 GB from baidu/ERNIE-5.5-Flash-Base at the pinned revision; needs ERNIE_CONFIRM_LICENSE=1
#   scripts/ernie55_pod_run.sh verify       # file count, every size against the manifest; ERNIE_VERIFY=sha256 also hashes every file
#   scripts/ernie55_pod_run.sh gpu-check    # one GPU, enough free VRAM, no compute process running
#   scripts/ernie55_pod_run.sh ptx-check    # shipped PTX is ISA 9.0 and the harness's kernels are present
#   scripts/ernie55_pod_run.sh test         # the real-weight harness on CUDA, with peak VRAM sampled
#   scripts/ernie55_pod_run.sh all          # preflight, verify, gpu-check, ptx-check, test (never download)
#
# Environment: ERNIE_CKPT_DIR (checkpoint folder, required by verify and test), ERNIE_RUN_DIR (logs, default ./ernie55-run-<time>),
# ERNIE_MIN_FREE_MIB (default 24576), ERNIE_REAL_TOKENS (default 1,2,3,4,5), ERNIE_VERIFY (size or sha256; default size).
# Every stage stops the script on its first failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${REPO_ROOT}/scripts/ernie55-flash-base-028544b2.manifest"
REVISION="028544b2ffceea189b2d8768ed2b9d09b06f2343"
REPO_ID="baidu/ERNIE-5.5-Flash-Base"
EXPECTED_SHARDS=128
EXPECTED_BYTES=288448161526   # 128 shards + index + config.json; the manifest lists exactly these
MIN_FREE_MIB="${ERNIE_MIN_FREE_MIB:-24576}"
TOKENS="${ERNIE_REAL_TOKENS:-1,2,3,4,5}"
RUN_DIR="${ERNIE_RUN_DIR:-${PWD}/ernie55-run-$(date +%Y%m%d-%H%M%S)}"

fail() {
    echo "FAILED: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "$1 is not on PATH"
}

require_ckpt() {
    [[ -n "${ERNIE_CKPT_DIR:-}" ]] || fail "set ERNIE_CKPT_DIR to the checkpoint folder"
    [[ -d "$ERNIE_CKPT_DIR" ]] || fail "$ERNIE_CKPT_DIR is not a folder"
}

ensure_run_dir() {
    mkdir -p "$RUN_DIR"
}

stage_preflight() {
    require_cmd dotnet
    require_cmd nvidia-smi
    require_cmd stat
    require_cmd sha256sum
    ensure_run_dir
    local commit
    commit="$(git -C "$REPO_ROOT" rev-parse HEAD)" || fail "$REPO_ROOT is not a git checkout"
    {
        echo "repo commit: ${commit}"
        echo "dotnet: $(dotnet --version)"
        echo "nvidia-smi:"
        nvidia-smi
        echo "gpu:"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
        if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -1; else echo "nvcc: not on PATH (not needed to run the test)"; fi
    } | tee "${RUN_DIR}/environment.txt"
    echo "preflight ok; run folder ${RUN_DIR}"
}

stage_download() {
    [[ "${ERNIE_CONFIRM_LICENSE:-}" == "1" ]] || fail "download needs ERNIE_CONFIRM_LICENSE=1 after the license is confirmed (runbook section 3)"
    require_ckpt
    require_cmd hf
    mkdir -p "$ERNIE_CKPT_DIR"
    hf download "$REPO_ID" --revision "$REVISION" --local-dir "$ERNIE_CKPT_DIR" --include "*.json" --include "*.safetensors"
    stage_verify
}

stage_verify() {
    require_ckpt
    [[ -f "$MANIFEST" ]] || fail "manifest missing: $MANIFEST"
    local path sha size actual missing=0 mismatched=0 listed=0 total=0 sum=0
    while read -r sha size path; do
        [[ -z "$sha" || "$sha" == \#* ]] && continue
        listed=$((listed + 1))
        sum=$((sum + size))
        if [[ ! -f "${ERNIE_CKPT_DIR}/${path}" ]]; then
            echo "missing: ${path}"
            missing=$((missing + 1))
            continue
        fi
        actual="$(stat -c %s "${ERNIE_CKPT_DIR}/${path}")"
        total=$((total + actual))
        if [[ "$actual" != "$size" ]]; then
            echo "size mismatch: ${path} is ${actual}, the pin says ${size}"
            mismatched=$((mismatched + 1))
        fi
    done < "$MANIFEST"
    [[ "$missing" -eq 0 ]] || fail "${missing} file(s) missing"
    [[ "$mismatched" -eq 0 ]] || fail "${mismatched} file(s) differ in size"
    [[ "$sum" -eq "$EXPECTED_BYTES" ]] || fail "manifest sums to ${sum}, expected ${EXPECTED_BYTES}"
    [[ "$total" -eq "$EXPECTED_BYTES" ]] || fail "files total ${total} bytes, expected ${EXPECTED_BYTES}"

    local shards index_shards
    shards="$(find "$ERNIE_CKPT_DIR" -maxdepth 1 -name 'model-*-of-00128.safetensors' | wc -l)"
    [[ "$shards" -eq "$EXPECTED_SHARDS" ]] || fail "${shards} shard files in ${ERNIE_CKPT_DIR}, expected ${EXPECTED_SHARDS}"
    index_shards="$(grep -o 'model-[0-9]*-of-00128\.safetensors' "${ERNIE_CKPT_DIR}/model.safetensors.index.json" | sort -u | wc -l)"
    [[ "$index_shards" -eq "$EXPECTED_SHARDS" ]] || fail "the index names ${index_shards} shards, expected ${EXPECTED_SHARDS}"
    echo "sizes ok: ${listed} manifest files, ${total} bytes, ${shards} shards"

    if [[ "${ERNIE_VERIFY:-size}" == "sha256" ]]; then
        local list="${RUN_DIR:-${TMPDIR:-/tmp}}/sha256.list"
        ensure_run_dir
        awk '$1 != "-" && $1 !~ /^#/ { print $1 "  " $3 }' "$MANIFEST" > "$list"
        (cd "$ERNIE_CKPT_DIR" && sha256sum -c --quiet "$list") || fail "sha256 mismatch"
        echo "sha256 ok: $(wc -l < "$list") files"
    else
        echo "sha256 not checked (set ERNIE_VERIFY=sha256 for the full hash pass)"
    fi
}

stage_gpu_check() {
    require_cmd nvidia-smi
    local rows count free_mib name
    rows="$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits)"
    count="$(printf '%s\n' "$rows" | grep -c .)"
    [[ "$count" -eq 1 ]] || fail "${count} GPUs visible; the runbook is one GPU"
    name="${rows%%,*}"
    free_mib="$(printf '%s\n' "$rows" | awk -F', ' '{ print $2 }')"
    [[ "$free_mib" -ge "$MIN_FREE_MIB" ]] || fail "${free_mib} MiB free on ${name}; need ${MIN_FREE_MIB} MiB"
    local apps
    apps="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
    [[ -z "$apps" ]] || fail "another process holds the GPU: ${apps}"
    echo "GPU free: ${name}, ${free_mib} MiB free, no compute process"
}

stage_ptx_check() {
    local ptx_dir="${REPO_ROOT}/src/HartsyInference.Cuda/Ptx"
    "${REPO_ROOT}/src/HartsyInference.Cuda/Kernels/check_ptx_isa.sh" >/dev/null || fail "a shipped PTX file is not ISA 9.0"
    local needed
    for needed in fp8_block_linear.ptx mul_mat_vec_f16_bf16_f32.ptx lm_f32.ptx; do
        [[ -f "${ptx_dir}/${needed}" ]] || fail "missing ${needed} in ${ptx_dir}"
    done
    echo "ptx ok: ISA 9.0 throughout; targets: $(grep -h '^\.target' "${ptx_dir}/fp8_block_linear.ptx" "${ptx_dir}/lm_f32.ptx" | sort -u | tr '\n' ' ')"
}

stage_test() {
    require_ckpt
    require_cmd dotnet
    require_cmd nvidia-smi
    ensure_run_dir
    local log="${RUN_DIR}/ernie55_real_test.log" vram="${RUN_DIR}/vram_mib.log" sampler
    (
        while true; do
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
            sleep 0.5
        done
    ) > "$vram" 2>/dev/null &
    sampler=$!
    # shellcheck disable=SC2064
    trap "kill $sampler 2>/dev/null || true" EXIT

    cd "$REPO_ROOT"
    local status=0
    ERNIE_REAL_DIR="$ERNIE_CKPT_DIR" ERNIE_REAL_DEVICE=cuda ERNIE_REAL_TOKENS="$TOKENS" \
        dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests" \
        --logger "console;verbosity=normal" 2>&1 | tee "$log" || status=$?
    kill "$sampler" 2>/dev/null || true
    trap - EXIT
    [[ "$status" -eq 0 ]] || fail "dotnet test exited ${status}; see ${log}"

    # A skipped fact validated nothing: require both facts to pass, none skipped.
    grep -qE 'Failed: +0, +Passed: +2, +Skipped: +0,' "$log" || fail "the harness did not report 2 passed, 0 skipped, 0 failed; see ${log}"
    local timing
    timing="$(grep -m1 '^ERNIE real run on cuda:' "$log" || true)"
    [[ -n "$timing" ]] || fail "no timing line from the forward; see ${log}"
    local peak
    peak="$(sort -n "$vram" | tail -1)"
    {
        echo "commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        nvidia-smi | grep -o 'CUDA Version: [0-9.]*' || true
        echo "$timing"
        echo "peak VRAM used (MiB, sampled every 0.5 s): ${peak:-unknown}"
    } | tee "${RUN_DIR}/summary.txt"
}

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
}

case "${1:-}" in
    preflight) stage_preflight ;;
    download)  stage_download ;;
    verify)    stage_verify ;;
    gpu-check) stage_gpu_check ;;
    ptx-check) stage_ptx_check ;;
    test)      stage_test ;;
    all)
        stage_preflight
        stage_verify
        stage_gpu_check
        stage_ptx_check
        stage_test
        ;;
    -h|--help|"") usage ;;
    *) fail "unknown stage: $1" ;;
esac
