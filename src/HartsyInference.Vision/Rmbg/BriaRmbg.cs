using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Rmbg;

/// <summary>BriaRMBG-1.4 (ISNet / U²-Net-style salient-object segmentation) for background removal. A nested-U
/// encoder–decoder of <see cref="RsuBlock"/> ReSidual-U blocks producing a per-pixel foreground probability at
/// 1024×1024. Used to isolate the subject for the image→3D pipelines (TripoSR / Hunyuan3D) so raw photos work
/// without an external tool. Pure conv/BN/ReLU/pool/upsample: <b>BatchNorm is folded into the preceding conv at
/// load</b>, and <b>dilated convs are realized as zero-inflated 3×3 kernels</b> (no dilation kernel needed).
/// <para>Runs at a fixed 1024² input (every pooling stage stays a clean /2), inference uses <c>sigmoid(d1)</c>
/// (the finest side output). Route matmuls/convs through <see cref="IBackend"/>.</para></summary>
public sealed unsafe class BriaRmbg
{
    private Tensor? _convInW, _convInB;                 // conv_in: 3→64, 3×3 stride 2 (plain conv, no BN/ReLU)
    private readonly RsuBlock[] _enc;                    // stage1..6
    private readonly RsuBlock[] _dec;                    // stage5d..1d
    private Tensor? _side1W, _side1B;                    // side1: 64→1, 3×3 (only d1 is used at inference)

    public BriaRmbg()
    {
        // stageN (encoder): RSU height L, (in, mid, out). stage5/6 are the dilated RSU-4F (L=4, dilated).
        _enc = [
            new RsuBlock(7, 64, 32, 64), new RsuBlock(6, 64, 32, 128), new RsuBlock(5, 128, 64, 256),
            new RsuBlock(4, 256, 128, 512), new RsuBlock(4, 512, 256, 512, dilated: true), new RsuBlock(4, 512, 256, 512, dilated: true),
        ];
        // stageNd (decoder): input is cat(up(prev), skip) = 2× the skip channels.
        _dec = [
            new RsuBlock(4, 1024, 256, 512, dilated: true), new RsuBlock(4, 1024, 128, 256), new RsuBlock(5, 512, 64, 128),
            new RsuBlock(6, 256, 32, 64), new RsuBlock(7, 128, 16, 64),
        ];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _convInW = F32(w["conv_in.weight"]); _convInB = F32(w["conv_in.bias"]);
        for (int i = 0; i < 6; i++) _enc[i].LoadWeights(w, $"stage{i + 1}");
        string[] decNames = ["stage5d", "stage4d", "stage3d", "stage2d", "stage1d"];
        for (int i = 0; i < 5; i++) _dec[i].LoadWeights(w, decNames[i]);
        _side1W = F32(w["side1.weight"]); _side1B = F32(w["side1.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_convInW, _convInB, _side1W, _side1B];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (RsuBlock b in _enc) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (RsuBlock b in _dec) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Segments preprocessed pixels <c>[1,3,1024,1024]</c> (ImageNet-free: (x/255−0.5)) into a foreground
    /// probability mask <c>[1,1,1024,1024]</c> = <c>sigmoid(d1)</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor pixels)
    {
        // conv_in (stride 2): 1024 → 512.
        Tensor hxin = RmbgOps.Conv(backend, pixels, _convInW!, _convInB!, stride: 2, pad: 1);

        // Encoder: hx1..hx6 with a /2 pool between stages.
        Tensor hx1 = _enc[0].Forward(backend, hxin); hxin.Dispose();
        Tensor p1 = RmbgOps.Pool2(backend, hx1);
        Tensor hx2 = _enc[1].Forward(backend, p1); p1.Dispose();
        Tensor p2 = RmbgOps.Pool2(backend, hx2);
        Tensor hx3 = _enc[2].Forward(backend, p2); p2.Dispose();
        Tensor p3 = RmbgOps.Pool2(backend, hx3);
        Tensor hx4 = _enc[3].Forward(backend, p3); p3.Dispose();
        Tensor p4 = RmbgOps.Pool2(backend, hx4);
        Tensor hx5 = _enc[4].Forward(backend, p4); p4.Dispose();
        Tensor p5 = RmbgOps.Pool2(backend, hx5);
        Tensor hx6 = _enc[5].Forward(backend, p5); p5.Dispose();

        // Decoder: upsample + concat(skip) + RSU.
        Tensor hx5d = DecodeStage(backend, _dec[0], hx6, hx5); hx6.Dispose(); hx5.Dispose();
        Tensor hx4d = DecodeStage(backend, _dec[1], hx5d, hx4); hx5d.Dispose(); hx4.Dispose();
        Tensor hx3d = DecodeStage(backend, _dec[2], hx4d, hx3); hx4d.Dispose(); hx3.Dispose();
        Tensor hx2d = DecodeStage(backend, _dec[3], hx3d, hx2); hx3d.Dispose(); hx2.Dispose();
        Tensor hx1d = DecodeStage(backend, _dec[4], hx2d, hx1); hx2d.Dispose(); hx1.Dispose();

        // d1 = sigmoid(upsample(side1(hx1d) → 1024)). hx1d is at 512 → 2× to 1024.
        Tensor d1 = RmbgOps.Conv(backend, hx1d, _side1W!, _side1B!, stride: 1, pad: 1, relu: false); hx1d.Dispose();
        Tensor up = RmbgOps.Upsample2x(d1); d1.Dispose();
        Tensor mask = new(up.Shape, DType.F32); backend.Sigmoid(mask, up); up.Dispose();
        return mask;
    }

    // up(prev) to the skip's spatial size (clean 2×), concat on channels, run the decoder RSU.
    private static Tensor DecodeStage(IBackend backend, RsuBlock stage, Tensor prev, Tensor skip)
    {
        int c = (int)prev.Shape[1], sh = (int)skip.Shape[2], sw = (int)skip.Shape[3];
        Tensor up = RmbgOps.Upsample2x(prev);
        Tensor cat = new(new TensorShape(1, c + (int)skip.Shape[1], sh, sw), DType.F32);
        backend.Concat(cat, [up, skip], 1); up.Dispose();
        Tensor outp = stage.Forward(backend, cat); cat.Dispose();
        return outp;
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
