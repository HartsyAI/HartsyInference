using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.DepthAnything;

/// <summary>DPT head of Depth-Anything-V2: four backbone taps are projected to per-branch widths, resized to a
/// 4×/2×/1×/½× feature pyramid, unified to <c>features</c> channels (3×3, no bias), fused coarse→fine through
/// residual-conv-unit fusion blocks (bilinear <c>align_corners=True</c> upsampling to the next skip's size), and
/// decoded by the output convs to a single-channel relative-depth map at <c>14·patchGrid</c> resolution.
/// Matches the official <c>DPTHead</c> with <c>use_bn=False</c>, <c>use_clstoken=False</c>. All math routes
/// through <see cref="IBackend"/>.</summary>
public sealed class DptHead
{
    private readonly int _inChannels;
    private readonly int _features;
    private readonly int[] _outChannels;

    private readonly Tensor?[] _projW = new Tensor?[4];
    private readonly Tensor?[] _projB = new Tensor?[4];
    private Tensor? _resize0W, _resize0B;   // ConvTranspose2d k4 s4
    private Tensor? _resize1W, _resize1B;   // ConvTranspose2d k2 s2
    private Tensor? _resize3W, _resize3B;   // Conv2d k3 s2 p1
    private readonly Tensor?[] _layerRnW = new Tensor?[4];
    private readonly FusionBlock[] _refinenets = new FusionBlock[4];
    private Tensor? _outConv1W, _outConv1B;
    private Tensor? _outConv2AW, _outConv2AB;
    private Tensor? _outConv2BW, _outConv2BB;

    public DptHead(int inChannels, int features, int[] outChannels)
    {
        if (outChannels.Length != 4)
            throw new ArgumentException($"DPT head expects 4 reassemble widths; got {outChannels.Length}.", nameof(outChannels));
        _inChannels = inChannels;
        _features = features;
        _outChannels = outChannels;
        for (int i = 0; i < 4; i++) _refinenets[i] = new FusionBlock();
    }

    /// <summary>Loads the official checkpoint's <c>depth_head.*</c> tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "depth_head")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < 4; i++)
        {
            _projW[i] = Dinov2VisionEncoder.F32(w[$"{p}projects.{i}.weight"]);
            _projB[i] = Dinov2VisionEncoder.F32(w[$"{p}projects.{i}.bias"]);
            _layerRnW[i] = Dinov2VisionEncoder.F32(w[$"{p}scratch.layer{i + 1}_rn.weight"]);
            _refinenets[i].LoadWeights(w, $"{p}scratch.refinenet{i + 1}");
        }
        _resize0W = Dinov2VisionEncoder.F32(w[$"{p}resize_layers.0.weight"]);
        _resize0B = Dinov2VisionEncoder.F32(w[$"{p}resize_layers.0.bias"]);
        _resize1W = Dinov2VisionEncoder.F32(w[$"{p}resize_layers.1.weight"]);
        _resize1B = Dinov2VisionEncoder.F32(w[$"{p}resize_layers.1.bias"]);
        _resize3W = Dinov2VisionEncoder.F32(w[$"{p}resize_layers.3.weight"]);
        _resize3B = Dinov2VisionEncoder.F32(w[$"{p}resize_layers.3.bias"]);
        _outConv1W = Dinov2VisionEncoder.F32(w[$"{p}scratch.output_conv1.weight"]);
        _outConv1B = Dinov2VisionEncoder.F32(w[$"{p}scratch.output_conv1.bias"]);
        _outConv2AW = Dinov2VisionEncoder.F32(w[$"{p}scratch.output_conv2.0.weight"]);
        _outConv2AB = Dinov2VisionEncoder.F32(w[$"{p}scratch.output_conv2.0.bias"]);
        _outConv2BW = Dinov2VisionEncoder.F32(w[$"{p}scratch.output_conv2.2.weight"]);
        _outConv2BB = Dinov2VisionEncoder.F32(w[$"{p}scratch.output_conv2.2.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] flat =
        [
            _projW[0], _projB[0], _projW[1], _projB[1], _projW[2], _projB[2], _projW[3], _projB[3],
            _resize0W, _resize0B, _resize1W, _resize1B, _resize3W, _resize3B,
            _layerRnW[0], _layerRnW[1], _layerRnW[2], _layerRnW[3],
            _outConv1W, _outConv1B, _outConv2AW, _outConv2AB, _outConv2BW, _outConv2BB,
        ];
        foreach (Tensor? t in flat) if (t is not null) yield return t;
        foreach (FusionBlock f in _refinenets) foreach (Tensor t in f.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes the four tapped patch-token features <c>[B, gridH·gridW, hidden]</c> into a
    /// relative-depth map <c>[B, 1, 14·gridH, 14·gridW]</c> (already ReLU'd, ≥ 0).
    /// <paramref name="tap"/> receives named intermediates for the parity harness.</summary>
    public Tensor Forward(IBackend backend, Tensor[] patchFeatures, int gridH, int gridW, Action<string, Tensor>? tap = null)
    {
        if (patchFeatures.Length != 4)
            throw new ArgumentException($"DPT head expects 4 tapped features; got {patchFeatures.Length}.", nameof(patchFeatures));
        int batch = (int)patchFeatures[0].Shape[0];
        int numPatches = gridH * gridW;

        // Reassemble: [B,N,C] → [B,C,gridH,gridW] → 1×1 project → per-branch resize (4×, 2×, 1×, ½×).
        Tensor[] layers = new Tensor[4];
        for (int i = 0; i < 4; i++)
        {
            Tensor feat = patchFeatures[i];
            if (feat.Shape[1] != numPatches || feat.Shape[2] != _inChannels)
                throw new ArgumentException($"Tap {i} must be [B,{numPatches},{_inChannels}]; got {feat.Shape}.");
            Tensor chw = new(new TensorShape(batch, _inChannels, numPatches), DType.F32);
            backend.Transpose2D(chw, feat, numPatches, _inChannels);
            Tensor grid = chw.Reshape(new TensorShape(batch, _inChannels, gridH, gridW));
            Tensor proj = new(new TensorShape(batch, _outChannels[i], gridH, gridW), DType.F32);
            backend.Conv2D(proj, grid, _projW[i]!, _projB[i]!, 1, 1, 0, 0);
            chw.Dispose();
            layers[i] = Resize(backend, proj, i, gridH, gridW);
            tap?.Invoke($"layer_{i + 1}", layers[i]);
        }

        // Unify channel widths (3×3, no bias) → the scratch.layerN_rn features.
        Tensor[] rn = new Tensor[4];
        for (int i = 0; i < 4; i++)
        {
            rn[i] = new Tensor(new TensorShape(batch, _features, layers[i].Shape[2], layers[i].Shape[3]), DType.F32);
            backend.Conv2D(rn[i], layers[i], _layerRnW[i]!, null, 1, 1, 1, 1);
            layers[i].Dispose();
            tap?.Invoke($"layer_{i + 1}_rn", rn[i]);
        }

        // Coarse→fine fusion. refinenet4 has no skip input; refinenet1 upsamples by exactly 2×.
        Tensor path = _refinenets[3].Forward(backend, rn[3], null, (int)rn[2].Shape[2], (int)rn[2].Shape[3], _features);
        tap?.Invoke("path_4", path);
        for (int i = 2; i >= 0; i--)
        {
            int outH = i > 0 ? (int)rn[i - 1].Shape[2] : (int)rn[0].Shape[2] * 2;
            int outW = i > 0 ? (int)rn[i - 1].Shape[3] : (int)rn[0].Shape[3] * 2;
            Tensor next = _refinenets[i].Forward(backend, path, rn[i], outH, outW, _features);
            path.Dispose();
            rn[i].Dispose();
            path = next;
            tap?.Invoke($"path_{i + 1}", path);
        }
        rn[3].Dispose();

        // Output convs: 3×3 → half features, bilinear to pixel resolution, 3×3 → 32, ReLU, 1×1 → 1, ReLU.
        int half = _features / 2;
        Tensor c1 = new(new TensorShape(batch, half, path.Shape[2], path.Shape[3]), DType.F32);
        backend.Conv2D(c1, path, _outConv1W!, _outConv1B!, 1, 1, 1, 1);
        path.Dispose();
        Tensor up = new(new TensorShape(batch, half, gridH * 14, gridW * 14), DType.F32);
        backend.InterpolateBilinear2D(up, c1, alignCorners: true);
        c1.Dispose();
        Tensor c2 = new(new TensorShape(batch, 32, up.Shape[2], up.Shape[3]), DType.F32);
        backend.Conv2D(c2, up, _outConv2AW!, _outConv2AB!, 1, 1, 1, 1);
        up.Dispose();
        Tensor c2r = new(c2.Shape, DType.F32);
        backend.LeakyRelu(c2r, c2, 0f);
        c2.Dispose();
        Tensor depth = new(new TensorShape(batch, 1, c2r.Shape[2], c2r.Shape[3]), DType.F32);
        backend.Conv2D(depth, c2r, _outConv2BW!, _outConv2BB!, 1, 1, 0, 0);
        c2r.Dispose();
        Tensor relu = new(depth.Shape, DType.F32);
        backend.LeakyRelu(relu, depth, 0f);
        depth.Dispose();
        return relu;
    }

    /// <summary>Per-branch resize: ConvTranspose 4×/2× for the two fine taps, identity for the third,
    /// strided 3×3 conv (½×) for the coarsest.</summary>
    private Tensor Resize(IBackend backend, Tensor proj, int branch, int gridH, int gridW)
    {
        switch (branch)
        {
            case 0:
            {
                Tensor o = new(new TensorShape(proj.Shape[0], _outChannels[0], gridH * 4, gridW * 4), DType.F32);
                backend.ConvTranspose2d(o, proj, _resize0W!, _resize0B!, 4, 4, 0, 0);
                proj.Dispose();
                return o;
            }
            case 1:
            {
                Tensor o = new(new TensorShape(proj.Shape[0], _outChannels[1], gridH * 2, gridW * 2), DType.F32);
                backend.ConvTranspose2d(o, proj, _resize1W!, _resize1B!, 2, 2, 0, 0);
                proj.Dispose();
                return o;
            }
            case 2:
                return proj;
            default:
            {
                int oh = (gridH + 2 - 3) / 2 + 1, ow = (gridW + 2 - 3) / 2 + 1;
                Tensor o = new(new TensorShape(proj.Shape[0], _outChannels[3], oh, ow), DType.F32);
                backend.Conv2D(o, proj, _resize3W!, _resize3B!, 2, 2, 1, 1);
                proj.Dispose();
                return o;
            }
        }
    }

    /// <summary>One DPT feature-fusion block: optional skip through a residual conv unit, a second residual
    /// conv unit, bilinear <c>align_corners=True</c> resize to the target size, then a 1×1 out conv.</summary>
    private sealed class FusionBlock
    {
        private Tensor? _outW, _outB;
        private Tensor? _rcu1C1W, _rcu1C1B, _rcu1C2W, _rcu1C2B;
        private Tensor? _rcu2C1W, _rcu2C1B, _rcu2C2W, _rcu2C2B;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _outW = Dinov2VisionEncoder.F32(w[$"{prefix}.out_conv.weight"]);
            _outB = Dinov2VisionEncoder.F32(w[$"{prefix}.out_conv.bias"]);
            _rcu1C1W = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit1.conv1.weight"]);
            _rcu1C1B = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit1.conv1.bias"]);
            _rcu1C2W = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit1.conv2.weight"]);
            _rcu1C2B = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit1.conv2.bias"]);
            _rcu2C1W = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit2.conv1.weight"]);
            _rcu2C1B = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit2.conv1.bias"]);
            _rcu2C2W = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit2.conv2.weight"]);
            _rcu2C2B = Dinov2VisionEncoder.F32(w[$"{prefix}.resConfUnit2.conv2.bias"]);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_outW, _outB, _rcu1C1W, _rcu1C1B, _rcu1C2W, _rcu1C2B, _rcu2C1W, _rcu2C1B, _rcu2C2W, _rcu2C2B];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor? skip, int outH, int outW, int features)
        {
            Tensor cur;
            if (skip is not null)
            {
                Tensor res = ResidualConvUnit(backend, skip, _rcu1C1W!, _rcu1C1B!, _rcu1C2W!, _rcu1C2B!);
                cur = new Tensor(x.Shape, DType.F32);
                backend.Add(cur, x, res);
                res.Dispose();
            }
            else
            {
                cur = x;
            }

            Tensor r2 = ResidualConvUnit(backend, cur, _rcu2C1W!, _rcu2C1B!, _rcu2C2W!, _rcu2C2B!);
            if (!ReferenceEquals(cur, x)) cur.Dispose();
            Tensor up = new(new TensorShape(r2.Shape[0], features, outH, outW), DType.F32);
            backend.InterpolateBilinear2D(up, r2, alignCorners: true);
            r2.Dispose();
            Tensor outp = new(up.Shape, DType.F32);
            backend.Conv2D(outp, up, _outW!, _outB!, 1, 1, 0, 0);
            up.Dispose();
            return outp;
        }

        /// <summary><c>x + conv2(relu(conv1(relu(x))))</c> — both convs 3×3, pad 1.</summary>
        private static Tensor ResidualConvUnit(IBackend backend, Tensor x, Tensor c1W, Tensor c1B, Tensor c2W, Tensor c2B)
        {
            Tensor a = new(x.Shape, DType.F32);
            backend.LeakyRelu(a, x, 0f);
            Tensor c1 = new(x.Shape, DType.F32);
            backend.Conv2D(c1, a, c1W, c1B, 1, 1, 1, 1);
            a.Dispose();
            Tensor a2 = new(x.Shape, DType.F32);
            backend.LeakyRelu(a2, c1, 0f);
            c1.Dispose();
            Tensor c2 = new(x.Shape, DType.F32);
            backend.Conv2D(c2, a2, c2W, c2B, 1, 1, 1, 1);
            a2.Dispose();
            Tensor outp = new(x.Shape, DType.F32);
            backend.Add(outp, c2, x);
            c2.Dispose();
            return outp;
        }
    }
}
