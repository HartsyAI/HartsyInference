#!/usr/bin/env bash
# Build the HartsyInference API at a branch/commit of the shared clone into /root/baseline/engine/bin-<name>: pod-build-ref.sh <name> <ref>
export DOTNET_ROOT=/root/dotnet PATH=/root/dotnet:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1
NAME=$1; REF=$2
L=/root/baseline/logs/build-$NAME.status
echo building > $L
cd /root/baseline/engine/src-main
git fetch -q origin "$REF" || { echo failed:fetch > $L; exit 1; }
git checkout -q -f FETCH_HEAD || { echo failed:checkout > $L; exit 1; }
echo "commit $(git rev-parse --short HEAD) $(git log -1 --format=%s | cut -c1-80)" > /root/baseline/logs/build-$NAME.commit
dotnet publish src/HartsyInference.API/HartsyInference.API.csproj -c Release -o /root/baseline/engine/bin-$NAME > /root/baseline/logs/build-$NAME.log 2>&1 || { echo failed:publish > $L; exit 1; }
echo done > $L
