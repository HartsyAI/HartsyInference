#!/usr/bin/env bash
# Runs on a fresh pod: layout, dotnet, llama.cpp (pinned d81235049) build, engine clone, Strata harness clone, model downloads.
# Progress lines go to /root/baseline/logs/bootstrap.log; the last line is BOOTSTRAP_DONE (or BOOTSTRAP_FAILED <step>).
set -u
B=/root/baseline; L=$B/logs
mkdir -p $B/{bin,cfg,prompts,engine,run,results,logs,scripts} /root/models/moe
cp -r /root/podkit/scripts/. $B/scripts/ && cp -r /root/podkit/cfg/. $B/cfg/ && cp -r /root/podkit/prompts/. $B/prompts/
cp $B/scripts/libcuptitrace.so $B/scripts/ 2>/dev/null; chmod +x $B/scripts/*.sh
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
say(){ echo "$(date -u +%T) $*"; }
fail(){ say "BOOTSTRAP_FAILED $1"; exit 1; }
say "env: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) nproc=$(nproc) $(df -h / | tail -n1 | awk '{print $4}') free"
pip install -q --break-system-packages ninja psutil numpy 2>&1 | tail -n1

# 1. model downloads (background, sha256-checked) -------------------------------------------------------------------------
( D=/root/models/moe
  fetch(){ local url=$1 name=$2 bytes=$3 sha=$4
    [ -f $D/$name ] && [ "$(stat -c %s $D/$name)" = "$bytes" ] || curl -sSL --retry 5 --retry-delay 5 -o $D/$name "$url" || { echo "FAILED $name"; return; }
    got=$(sha256sum $D/$name | cut -d' ' -f1); [ "$got" = "$sha" ] && echo "OK $name" || echo "SHA-MISMATCH $name"; }
  fetch https://huggingface.co/bartowski/granite-3.0-3b-a800m-instruct-GGUF/resolve/main/granite-3.0-3b-a800m-instruct-Q4_K_M.gguf granite-3b-a800m-Q4_K_M.gguf 2059356000 aa6287bafd182d06a6d11b0d001458deec902e8c0522875fc20cde2f561ebf6d
  fetch https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf qwen3-30b-a3b-Q4_K_M.gguf 18556686912 9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48
  # NOT TheBloke's Mixtral: it stores one tensor per expert, which this llama.cpp (and the stacked-expert loader) rejects. mradermacher's stacks them.
  fetch https://huggingface.co/mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf mixtral-8x7b-instruct-v0.1-mradermacher-Q4_K_M.gguf 28448468384 9f8d778c2eb8fd7f8426d03496aec30c46cf4076e98758818cc73d020ed01d44
  fetch https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf olmoe-1b-7b-0924-instruct-q4_k_m.gguf 4213512672 8c310f1435a1222338fd2d3d974975be9cd908180b644bab0c2a94da1ac32f3f
  echo MODELS_DONE ) > $L/models.log 2>&1 &

# 2. dotnet + engine source ------------------------------------------------------------------------------------------------
( if [ ! -x /root/dotnet/dotnet ]; then curl -sSL https://dot.net/v1/dotnet-install.sh -o /root/dotnet-install.sh && bash /root/dotnet-install.sh --channel 10.0 --install-dir /root/dotnet > $L/dotnet-install.log 2>&1; fi
  mkdir -p $B/engine && cd $B/engine && rm -rf src-main && git clone -q https://github.com/HartsyAI/HartsyInference src-main && echo ENGINE_CLONED ) > $L/engine.log 2>&1 &

# 3. Strata (benchmark harness only) and llama.cpp -------------------------------------------------------------------------
cd $B
git clone -q https://github.com/Niko1221/Strata Strata && git -C Strata checkout -q fb58e0dbc8399662c0e47c76578c6e878b14f6cf || fail strata
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp llama.cpp || fail llama-clone
git -C llama.cpp checkout -q d81235049 || fail llama-checkout
say "llama.cpp at $(git -C llama.cpp rev-parse --short HEAD)"
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > $L/llamacpp-cmake.log 2>&1 || fail llama-cmake
cmake --build llama.cpp/build -j "$(nproc)" --target llama-server llama-cli llama-bench > $L/llamacpp-build.log 2>&1 || fail llama-build
say "llama.cpp built"
wait
# The engine clone and the model downloads ran in the background: a bare `wait` reports success whatever they did, so check their markers.
grep -q ENGINE_CLONED $L/engine.log && [ -x /root/dotnet/dotnet ] && [ -d $B/engine/src-main/.git ] || fail engine
cat $L/models.log; cat $L/engine.log | tail -n 2
grep -q MODELS_DONE $L/models.log && ! grep -q -E "FAILED|MISMATCH|MISSING" $L/models.log || fail models
say "BOOTSTRAP_DONE"
