#!/usr/bin/env bash
# Create the harness wrapper, dispatcher and per-model configs for one engine build: mk-variant.sh <name> <bin dir>.
set -eu
NAME=$1; BIN=$2; SETTINGS=${3:-{\}}
B=/root/baseline
cat > $B/bin/hartsy-serve-$NAME <<EOS
#!/usr/bin/env bash
# Harness adapter for the HartsyInference API build in $BIN. Reads -m <gguf> --port <p> --host <h>; ignores the llama flags.
set -eu
GGUF=""; PORT=8080; HOST=127.0.0.1
while [ \$# -gt 0 ]; do
  case "\$1" in
    -m) GGUF="\$2"; shift 2 ;;
    --port) PORT="\$2"; shift 2 ;;
    --host) HOST="\$2"; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "\$GGUF" ] || { echo "hartsy-serve-$NAME: -m <gguf> is required" >&2; exit 2; }
mkdir -p $B/run
export HOME=$B/home-$NAME
WORK=\$(mktemp -d $B/run/hartsy-$NAME-XXXXXX)
ln -s "\$GGUF" "\$WORK/x"
cd "\$WORK"
export DOTNET_ROOT=/root/dotnet PATH=/root/dotnet:\$PATH ASPNETCORE_URLS=http://\$HOST:\$PORT DOTNET_CLI_HOME=/root
exec dotnet $BIN/HartsyInference.API.dll
EOS
cat > $B/bin/bench-serve-$NAME <<EOS
#!/usr/bin/env bash
set -eu
HARTSY=0
ARGS=()
for a in "\$@"; do
  if [ "\$a" = "--hartsy-engine" ]; then HARTSY=1; else ARGS+=("\$a"); fi
done
if [ "\$HARTSY" = 1 ]; then
  exec $B/bin/hartsy-serve-$NAME "\${ARGS[@]}"
fi
exec $B/llama.cpp/build/bin/llama-server "\${ARGS[@]}"
EOS
mkdir -p $B/home-$NAME/.config/hartsyinference
printf '{"settings": %s}\n' "$SETTINGS" > $B/home-$NAME/.config/hartsyinference/settings.json
chmod +x $B/bin/hartsy-serve-$NAME $B/bin/bench-serve-$NAME
for m in granite-3b olmoe qwen3-30b mixtral; do
  sed "s#\"llama_server\": \"[^\"]*\"#\"llama_server\": \"$B/bin/bench-serve-$NAME\"#" $B/cfg/$m.json > $B/cfg/$m-$NAME.json
done
echo "variant $NAME ready ($BIN)"
