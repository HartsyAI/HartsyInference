"""Convert xiph's RNNoise model release into the rnnoise.safetensors that RnnoiseWeights loads.

Upstream ships the weights only inside rnnoise_data-<sha256>.tar.gz, the file xiph/rnnoise's download_model.sh
fetches from media.xiph.org; the sha256 is pinned in that repo's model_version file. The tarball holds:

    models/rnnoise10Ga_12.pth         float checkpoint behind src/rnnoise_data.c, the default C build   <- converted
    models/rnnoise10Gb_15.pth         float checkpoint behind src/rnnoise_data_little.c                 (unused)
    src/rnnoise_data{,_little}.{c,h}  the C tables; conv2 and the GRUs are int8 there, so they are not the source

The engine performs the same conversion in C# at install time (RnnoiseInstaller -> RnnoiseCheckpoint) and needs
none of this. This script is the offline reference: RnnoiseCheckpointTests compares the C# output with this one's
tensor by tensor. --check-c-tables also proves the checkpoint is the one the C library compiles in, by comparing
the three layers the C tables keep in float (conv1, dense_out, vad_dense) bit for bit.

    curl -O https://media.xiph.org/rnnoise/models/rnnoise_data-0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37.tar.gz
    pip install torch safetensors numpy
    python tools/convert_rnnoise.py rnnoise_data-0a8755f8...b37.tar.gz Models/audio/wake/denoise/rnnoise.safetensors \
        --check-c-tables

License: BSD-3-Clause (xiph/rnnoise COPYING), which covers the model data.
"""
import argparse
import hashlib
import io
import re
import sys
import tarfile

import numpy as np
import torch
from safetensors.torch import save_file

SHA256 = "0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"
URL = f"https://media.xiph.org/rnnoise/models/rnnoise_data-{SHA256}.tar.gz"
MEMBER = "models/rnnoise10Ga_12.pth"
C_TABLES = "src/rnnoise_data.c"
LICENSE = "BSD-3-Clause"

# Names and shapes RnnoiseWeights binds: cond 128, GRU 384, 65 features in, 32 gains out.
GRU, GATES, CAT = 384, 3 * 384, 4 * 384
EXPECTED = {
    "conv1.weight": (128, 65, 3), "conv1.bias": (128,),
    "conv2.weight": (GRU, 128, 3), "conv2.bias": (GRU,),
    **{f"gru{i}.{name}": shape for i in (1, 2, 3) for name, shape in (
        ("weight_ih_l0", (GATES, GRU)), ("weight_hh_l0", (GATES, GRU)),
        ("bias_ih_l0", (GATES,)), ("bias_hh_l0", (GATES,)))},
    "dense_out.weight": (32, CAT), "dense_out.bias": (32,),
    "vad_dense.weight": (1, CAT), "vad_dense.bias": (1,),
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state_dict(data):
    # weights_only: the checkpoint is a dict of args, kwargs, an OrderedDict of tensors, loss and epoch, all of which
    # the restricted unpickler accepts, so nothing in it is executed.
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    if set(state) != set(EXPECTED):
        raise SystemExit(f"unexpected tensor set: missing {sorted(set(EXPECTED) - set(state))}, "
                         f"extra {sorted(set(state) - set(EXPECTED))}")
    for name, shape in EXPECTED.items():
        if tuple(state[name].shape) != shape or state[name].dtype != torch.float32:
            raise SystemExit(f"{name}: {tuple(state[name].shape)} {state[name].dtype}, expected {shape} float32")
    # Each GRU's four tensors are views of one flattened cuDNN buffer; clone so every tensor owns its storage.
    return {name: state[name].detach().clone().contiguous() for name in EXPECTED}


def c_float_array(source, name):
    match = re.search(rf"static const float {name}\[(\d+)\] = \{{(.*?)\}};", source, re.S)
    if match is None:
        raise SystemExit(f"{C_TABLES} has no float array {name}")
    values = np.array([float(v) for v in match.group(2).split(",") if v.strip()], dtype=np.float64)
    if len(values) != int(match.group(1)):
        raise SystemExit(f"{name}: parsed {len(values)} values, declared {match.group(1)}")
    return values.astype(np.float32)


def check_c_tables(state, source):
    # wexchange writes each float matrix as [inputs, outputs], row-major: a conv's (out, in, k) becomes (k, in, out)
    # and a linear layer's (out, in) becomes (in, out).
    layers = {
        "conv1_weights_float": state["conv1.weight"].numpy().transpose(2, 1, 0),
        "conv1_bias": state["conv1.bias"].numpy(),
        "dense_out_weights_float": state["dense_out.weight"].numpy().T,
        "dense_out_bias": state["dense_out.bias"].numpy(),
        "vad_dense_weights_float": state["vad_dense.weight"].numpy().T,
        "vad_dense_bias": state["vad_dense.bias"].numpy(),
    }
    for name, expected in layers.items():
        if not np.array_equal(c_float_array(source, name), expected.reshape(-1)):
            raise SystemExit(f"{name} in {C_TABLES} differs from {MEMBER}; this is not the default model")
        print(f"  {name}: identical to the checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tarball", help=f"rnnoise_data-{SHA256[:8]}...tar.gz from {URL}")
    parser.add_argument("output", help="rnnoise.safetensors to write")
    parser.add_argument("--check-c-tables", action="store_true",
                        help=f"also compare the float layers of {C_TABLES} with the checkpoint")
    args = parser.parse_args()

    actual = sha256_of(args.tarball)
    if actual != SHA256:
        raise SystemExit(f"{args.tarball} has sha256 {actual}, not the pinned release {SHA256}")
    with tarfile.open(args.tarball, "r:gz") as tar:
        state = load_state_dict(tar.extractfile(MEMBER).read())
        if args.check_c_tables:
            check_c_tables(state, tar.extractfile(C_TABLES).read().decode("ascii"))

    # Same keys as RnnoiseInstaller writes, so either file says where it came from in the same words.
    metadata = {
        "hartsy.component": "denoiser",
        "hartsy.converter": "tools/convert_rnnoise.py",
        "hartsy.license": LICENSE,
        "hartsy.source_url": URL,
        "hartsy.source_sha256": SHA256,
        "hartsy.source_member": MEMBER,
    }
    save_file(state, args.output, metadata=metadata)
    print(f"wrote {args.output}: {len(state)} tensors from {MEMBER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
