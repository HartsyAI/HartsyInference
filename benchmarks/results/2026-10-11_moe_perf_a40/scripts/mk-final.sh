#!/usr/bin/env bash
# Final comparison setup: one dispatcher serving llama.cpp and two HartsyInference configurations from the same build, plus per-model configs.
# mk-final.sh <bin dir name, e.g. perf8>
set -eu
B=/root/baseline
N=$1
bash $B/scripts/mk-variant.sh ${N}F $B/engine/bin-$N '{"numerics.graphDecode": true}' >/dev/null
bash $B/scripts/mk-variant.sh ${N}K $B/engine/bin-$N '{"numerics.graphDecode": true, "vram.kvF16": true}' >/dev/null
cat > $B/bin/bench-serve-final <<EOS
#!/usr/bin/env bash
set -eu
MODE=llama
ARGS=()
for a in "\$@"; do
  case "\$a" in
    --hartsy-engine) MODE=hartsy ;;
    --hartsy-engine-kvf16) MODE=hartsykv ;;
    *) ARGS+=("\$a") ;;
  esac
done
case \$MODE in
  hartsy) exec $B/bin/hartsy-serve-${N}F "\${ARGS[@]}" ;;
  hartsykv) exec $B/bin/hartsy-serve-${N}K "\${ARGS[@]}" ;;
esac
exec $B/llama.cpp/build/bin/llama-server "\${ARGS[@]}"
EOS
chmod +x $B/bin/bench-serve-final
python3 - <<PY
import json
for m in ["granite-3b","olmoe","qwen3-30b","mixtral"]:
    c=json.load(open("$B/cfg/%s.json"%m))
    c["llama_server"]="$B/bin/bench-serve-final"
    names=[x["name"] for x in c["configs"]]
    llama=[x for x in c["configs"] if x["name"]=="llama-cpp-gpu"][0]
    c["configs"]=[llama,
      {"name":"hartsy-cuda","kind":"llama","args":["--hartsy-engine"]},
      {"name":"hartsy-cuda-kvf16","kind":"llama","args":["--hartsy-engine-kvf16"]}]
    json.dump(c,open("$B/cfg/%s-final.json"%m,"w"),indent=2)
print("final configs written")
PY
