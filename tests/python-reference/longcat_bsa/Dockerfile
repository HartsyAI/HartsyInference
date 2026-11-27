# The dumper's environment: torch 2.6.0 (cu124 wheels, which bring triton 3.2.0) and the C compiler triton needs for its driver.
FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 einops==0.8.0
