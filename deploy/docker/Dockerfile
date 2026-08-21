# Standalone HartsyInference.API, wake-word-listener deployment. Framework-dependent (not
# self-contained) since the aspnet runtime image already ships the .NET 10 runtime — a
# self-contained publish would just duplicate it into every layer for nothing.
#
# Build from the repo root:  docker build -f deploy/docker/Dockerfile -t hartsy-wake .

FROM mcr.microsoft.com/dotnet/sdk:10.0-noble AS build
WORKDIR /src

# Copy the whole repo — HartsyInference.API only references Core+Engine directly, but Engine
# pulls in every modality project transitively (see HartsyInference.API.csproj's own comment:
# "Deliberately thin... every modality package Engine pulls in transitively ships via Engine's
# own references"), so the solution needs to be present for restore to resolve project refs.
COPY . .

RUN dotnet publish src/HartsyInference.API/HartsyInference.API.csproj \
    -c Release -o /app/publish --no-self-contained

FROM mcr.microsoft.com/dotnet/aspnet:10.0-noble AS runtime
WORKDIR /app

# curl for the HEALTHCHECK below — the aspnet base image doesn't include it.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/publish .

# The base image already runs as the unprivileged 'app' user (UID 64198) by default —
# not overridden here, no reason for this process to run as root.

EXPOSE 5099
EXPOSE 10800

ENV ASPNETCORE_URLS=http://0.0.0.0:5099

# Liveness only — matches deploy/README.md's guidance: /health is unconditional 200 once the
# process is up, the right signal for "does this container need a restart." /ready (per-model
# loop liveness) is deliberately NOT used here; a 503 there means one generation model's loop
# died, not that the wake listener or the process itself is unhealthy, and restarting the whole
# container over that would be the exact overreaction the docs warn against.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5099/health || exit 1

ENTRYPOINT ["dotnet", "HartsyInference.API.dll"]
