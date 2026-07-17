#!/usr/bin/env python3
"""Convert the InsightFace ArcFace recognition backbone (buffalo_l's w600k_r50.onnx,
IResNet-50 trained on WebFace600K) to a canonical safetensors layout for
HartsyInference.Vision/Face/ArcFaceModel.cs, and dump an onnxruntime parity reference.

The ONNX export already folds each block's bn2/bn3 and the downsample BN into the conv
biases; the surviving BatchNormalization nodes are each block's LEADING bn1 (applied to
the block input before conv1), the pre-flatten bn2, and the post-fc `features` BN1d.
This converter:
  - emits block bn1 as precomputed scale/shift (scale shaped [C,1,1,1] so the C# side can
    apply it via the existing Conv2dDepthwise op with a 1x1 kernel);
  - folds bn2 (per-channel affine on the 512x7x7 map before flatten) AND the features
    BN1d (after fc) into fc.weight/fc.bias — valid because both are affine;
  - flattens PReLU alphas from [C,1,1] to [C].

Output keys:
  conv1.weight [64,3,3,3] / conv1.bias / prelu1.alpha [64]
  layer{L}.{B}.bn1.scale [C,1,1,1] / layer{L}.{B}.bn1.shift [C]
  layer{L}.{B}.conv1.weight / .conv1.bias / .prelu.alpha
  layer{L}.{B}.conv2.weight / .conv2.bias            (stride 2 when B == 0)
  layer{L}.{B}.downsample.weight / .downsample.bias  (B == 0 only, 1x1 stride 2)
  fc.weight [512, 25088] / fc.bias [512]             (bn2 + features BN folded)

Usage:
  python convert_arcface_onnx.py <w600k_r50.onnx> <out_weights.safetensors> <out_reference.safetensors> [face_image.png]
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def convert(onnx_path: str):
    model = onnx.load(onnx_path)
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    out = {}

    nodes = list(g.node)
    # Stem: Conv_0 + PRelu_1.
    stem_conv = nodes[0]
    assert stem_conv.op_type == 'Conv'
    out['conv1.weight'] = init[stem_conv.input[1]].astype(np.float32)
    out['conv1.bias'] = init[stem_conv.input[2]].astype(np.float32)
    stem_prelu = nodes[1]
    assert stem_prelu.op_type == 'PRelu'
    out['prelu1.alpha'] = init[stem_prelu.input[1]].reshape(-1).astype(np.float32)

    i = 2
    while i < len(nodes):
        n = nodes[i]
        if n.op_type != 'BatchNormalization':
            raise RuntimeError(f'expected BatchNormalization at node {i}, got {n.op_type}')
        bn_name = n.input[1]  # e.g. layer1.0.bn1.weight
        eps = next((a.f for a in n.attribute if a.name == 'epsilon'), 1e-5)
        gamma = init[n.input[1]].astype(np.float64)
        beta = init[n.input[2]].astype(np.float64)
        mean = init[n.input[3]].astype(np.float64)
        var = init[n.input[4]].astype(np.float64)
        scale = gamma / np.sqrt(var + eps)
        shift = beta - mean * scale

        if bn_name.startswith('bn2.'):
            # Tail: bn2 -> Flatten -> Gemm(fc) -> features BN1d. Fold both BNs into fc.
            gemm = nodes[i + 2]
            assert gemm.op_type == 'Gemm'
            fc_w = init[gemm.input[1]].astype(np.float64)  # [512, 25088]
            fc_b = init[gemm.input[2]].astype(np.float64)
            ch = scale.shape[0]
            spatial = fc_w.shape[1] // ch
            w3 = fc_w.reshape(fc_w.shape[0], ch, spatial)
            fc_b = fc_b + (w3.sum(axis=2) @ shift)
            fc_w = (w3 * scale[None, :, None]).reshape(fc_w.shape[0], -1)
            feat_bn = nodes[i + 3]
            assert feat_bn.op_type == 'BatchNormalization' and feat_bn.input[1] == 'features.weight'
            feps = next((a.f for a in feat_bn.attribute if a.name == 'epsilon'), 1e-5)
            fs = init[feat_bn.input[1]].astype(np.float64) / np.sqrt(init[feat_bn.input[4]].astype(np.float64) + feps)
            fc_b = fs * (fc_b - init[feat_bn.input[3]].astype(np.float64)) + init[feat_bn.input[2]].astype(np.float64)
            fc_w = fc_w * fs[:, None]
            out['fc.weight'] = fc_w.astype(np.float32)
            out['fc.bias'] = fc_b.astype(np.float32)
            break

        # Block: bn1 -> conv1 -> prelu -> conv2 [-> downsample conv] -> Add.
        block = bn_name.rsplit('.bn1.', 1)[0]  # layerL.B
        out[f'{block}.bn1.scale'] = scale.astype(np.float32).reshape(-1, 1, 1, 1)
        out[f'{block}.bn1.shift'] = shift.astype(np.float32)
        c1 = nodes[i + 1]; pr = nodes[i + 2]; c2 = nodes[i + 3]
        assert c1.op_type == 'Conv' and pr.op_type == 'PRelu' and c2.op_type == 'Conv'
        out[f'{block}.conv1.weight'] = init[c1.input[1]].astype(np.float32)
        out[f'{block}.conv1.bias'] = init[c1.input[2]].astype(np.float32)
        out[f'{block}.prelu.alpha'] = init[pr.input[1]].reshape(-1).astype(np.float32)
        out[f'{block}.conv2.weight'] = init[c2.input[1]].astype(np.float32)
        out[f'{block}.conv2.bias'] = init[c2.input[2]].astype(np.float32)
        i += 4
        if nodes[i].op_type == 'Conv':  # downsample branch on the block input
            ds = nodes[i]
            out[f'{block}.downsample.weight'] = init[ds.input[1]].astype(np.float32)
            out[f'{block}.downsample.bias'] = init[ds.input[2]].astype(np.float32)
            i += 1
        assert nodes[i].op_type == 'Add'
        i += 1

    return out


def make_inputs(face_image: str | None):
    rng = np.random.default_rng(1234)
    inputs = {'input_random': rng.standard_normal((1, 3, 112, 112)).astype(np.float32)}
    if face_image:
        from PIL import Image
        img = Image.open(face_image).convert('RGB')
        side = min(img.size)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        crop = img.crop((left, top, left + side, top + side)).resize((112, 112), Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32)                      # HWC RGB, 0..255
        chw = ((arr - 127.5) / 127.5).transpose(2, 0, 1)[None]        # insightface ArcFaceONNX norm
        # ascontiguousarray is load-bearing: astype(order='K') would keep the transposed memory
        # layout, so onnxruntime and the serialized safetensors would see DIFFERENT buffers.
        inputs['input_face'] = np.ascontiguousarray(chw, dtype=np.float32)
        # A second, tighter crop as an extra sample.
        side2 = int(side * 0.7)
        left2 = (img.width - side2) // 2
        top2 = (img.height - side2) // 4
        crop2 = img.crop((left2, top2, left2 + side2, top2 + side2)).resize((112, 112), Image.BILINEAR)
        arr2 = np.asarray(crop2, dtype=np.float32)
        inputs['input_face2'] = np.ascontiguousarray(((arr2 - 127.5) / 127.5).transpose(2, 0, 1)[None], dtype=np.float32)
    return inputs


def main():
    onnx_path, weights_out, ref_out = sys.argv[1], sys.argv[2], sys.argv[3]
    face_image = sys.argv[4] if len(sys.argv) > 4 else None

    out = convert(onnx_path)
    save_file(out, weights_out)
    print(f'wrote {len(out)} tensors -> {weights_out}')

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    ref = {}
    for name, x in make_inputs(face_image).items():
        y = sess.run(None, {in_name: x})[0]
        ref[name] = x
        ref[name.replace('input', 'output')] = y.astype(np.float32)
        print(name, '-> embedding norm', float(np.linalg.norm(y)))
    save_file(ref, ref_out)
    print(f'wrote reference -> {ref_out}')


if __name__ == '__main__':
    main()
