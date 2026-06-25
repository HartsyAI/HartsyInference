using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Upscale;

/// <summary>Configuration for an RRDBNet (ESRGAN / Real-ESRGAN) super-resolution generator.</summary>
public sealed record RealEsrganConfig
{
    /// <summary>Feature width (Real-ESRGAN x4plus: 64).</summary>
    public int NumFeat { get; init; } = 64;

    /// <summary>Number of RRDB blocks in the trunk (Real-ESRGAN x4plus: 23).</summary>
    public int NumBlock { get; init; } = 23;

    /// <summary>Dense growth channels inside each residual-dense block (Real-ESRGAN x4plus: 32).</summary>
    public int NumGrowCh { get; init; } = 32;

    /// <summary>Output upscale factor. 4 → two nearest-neighbour upsample stages; 2 → one.</summary>
    public int Scale { get; init; } = 4;

    /// <summary>Real-ESRGAN x4plus (and the anime 6B variant uses NumBlock=6).</summary>
    public static RealEsrganConfig X4Plus => new() { NumFeat = 64, NumBlock = 23, NumGrowCh = 32, Scale = 4 };

    /// <summary>Real-ESRGAN x4plus anime 6B.</summary>
    public static RealEsrganConfig X4PlusAnime6B => new() { NumFeat = 64, NumBlock = 6, NumGrowCh = 32, Scale = 4 };

    /// <summary>Real-ESRGAN x2plus.</summary>
    public static RealEsrganConfig X2Plus => new() { NumFeat = 64, NumBlock = 23, NumGrowCh = 32, Scale = 2 };
}

/// <summary>A plain Conv2d layer (weight + bias) with "same" padding for odd kernels, stride 1.
/// All RRDBNet convs are 3×3 stride-1, so padding is derived from the kernel size at load time.</summary>
public sealed class Conv2dLayer
{
    private readonly int _outChannels;
    private Tensor? _weight; // [Cout, Cin, k, k]
    private Tensor? _bias;   // [Cout]

    /// <summary>Output channel count.</summary>
    public int OutChannels => _outChannels;

    /// <summary>Creates a conv layer that will produce <paramref name="outChannels"/> channels.</summary>
    public Conv2dLayer(int outChannels) => _outChannels = outChannels;

    /// <summary>Loads <c>{prefix}.weight</c> and <c>{prefix}.bias</c> (cast to F32).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _weight = EnsureF32(weights[$"{prefix}.weight"]);
        _bias = EnsureF32(weights[$"{prefix}.bias"]);
    }

    /// <summary>Yields weight + bias for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Conv2d with same-padding, stride 1. Owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        if (_weight is null || _bias is null)
            throw new InvalidOperationException("Conv2dLayer weights not loaded.");

        int n = (int)input.Shape[0];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];
        int k = (int)_weight.Shape[2];
        int pad = (k - 1) / 2;

        Tensor output = new Tensor(new TensorShape(n, _outChannels, h, w), DType.F32);
        backend.Conv2D(output, input, _weight, _bias, 1, 1, pad, pad);
        return output;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}

/// <summary>Residual Dense Block: five 3×3 convs with dense (concatenated) inputs and LeakyReLU(0.2),
/// the last conv producing <c>numFeat</c> channels, scaled by 0.2 and added to the input.</summary>
public sealed class ResidualDenseBlock
{
    private readonly Conv2dLayer _conv1;
    private readonly Conv2dLayer _conv2;
    private readonly Conv2dLayer _conv3;
    private readonly Conv2dLayer _conv4;
    private readonly Conv2dLayer _conv5;
    private const float ResidualScale = 0.2f;
    private const float LeakySlope = 0.2f;

    /// <summary>Creates an RDB for the given feature/growth widths.</summary>
    public ResidualDenseBlock(int numFeat, int numGrowCh)
    {
        _conv1 = new Conv2dLayer(numGrowCh);
        _conv2 = new Conv2dLayer(numGrowCh);
        _conv3 = new Conv2dLayer(numGrowCh);
        _conv4 = new Conv2dLayer(numGrowCh);
        _conv5 = new Conv2dLayer(numFeat);
    }

    /// <summary>Loads <c>{prefix}.conv1..conv5</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _conv1.LoadWeights(weights, $"{prefix}.conv1");
        _conv2.LoadWeights(weights, $"{prefix}.conv2");
        _conv3.LoadWeights(weights, $"{prefix}.conv3");
        _conv4.LoadWeights(weights, $"{prefix}.conv4");
        _conv5.LoadWeights(weights, $"{prefix}.conv5");
    }

    /// <summary>Yields all conv weights.</summary>
    public IEnumerable<Tensor> EnumerateWeights() =>
        _conv1.EnumerateWeights()
            .Concat(_conv2.EnumerateWeights())
            .Concat(_conv3.EnumerateWeights())
            .Concat(_conv4.EnumerateWeights())
            .Concat(_conv5.EnumerateWeights());

    /// <summary>Forward pass with dense concatenation. Owns the returned tensor; does not dispose <paramref name="input"/>.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        Tensor x1 = LeakyConv(backend, _conv1, input);

        Tensor c2 = Cat(backend, input, x1);
        Tensor x2 = LeakyConv(backend, _conv2, c2);
        c2.Dispose();

        Tensor c3 = Cat(backend, input, x1, x2);
        Tensor x3 = LeakyConv(backend, _conv3, c3);
        c3.Dispose();

        Tensor c4 = Cat(backend, input, x1, x2, x3);
        Tensor x4 = LeakyConv(backend, _conv4, c4);
        c4.Dispose();

        Tensor c5 = Cat(backend, input, x1, x2, x3, x4);
        Tensor x5 = _conv5.Forward(backend, c5); // no activation on conv5
        c5.Dispose();

        // out = x5 * 0.2 + input
        Tensor scaled = new Tensor(x5.Shape, DType.F32);
        backend.Scale(scaled, x5, ResidualScale);
        Tensor outT = new Tensor(x5.Shape, DType.F32);
        backend.Add(outT, scaled, input);

        x1.Dispose();
        x2.Dispose();
        x3.Dispose();
        x4.Dispose();
        x5.Dispose();
        scaled.Dispose();
        return outT;
    }

    private static Tensor LeakyConv(IBackend backend, Conv2dLayer conv, Tensor input)
    {
        Tensor conved = conv.Forward(backend, input);
        backend.LeakyRelu(conved, conved, LeakySlope);
        return conved;
    }

    /// <summary>Channel-concatenates tensors [N, C_i, H, W] → [N, ΣC_i, H, W]. Disposes every temporary
    /// concat input except the persistent <c>input</c>/<c>x*</c> tensors (which the caller owns).</summary>
    private static Tensor Cat(IBackend backend, params Tensor[] parts)
    {
        int n = (int)parts[0].Shape[0];
        int h = (int)parts[0].Shape[2];
        int w = (int)parts[0].Shape[3];
        int totalC = 0;
        foreach (Tensor p in parts) totalC += (int)p.Shape[1];

        Tensor outT = new Tensor(new TensorShape(n, totalC, h, w), DType.F32);
        backend.Concat(outT, parts, dim: 1);
        return outT;
    }
}

/// <summary>Residual-in-Residual Dense Block: three <see cref="ResidualDenseBlock"/>s with a 0.2 residual.</summary>
public sealed class Rrdb
{
    private readonly ResidualDenseBlock _rdb1;
    private readonly ResidualDenseBlock _rdb2;
    private readonly ResidualDenseBlock _rdb3;
    private const float ResidualScale = 0.2f;

    /// <summary>Creates an RRDB for the given feature/growth widths.</summary>
    public Rrdb(int numFeat, int numGrowCh)
    {
        _rdb1 = new ResidualDenseBlock(numFeat, numGrowCh);
        _rdb2 = new ResidualDenseBlock(numFeat, numGrowCh);
        _rdb3 = new ResidualDenseBlock(numFeat, numGrowCh);
    }

    /// <summary>Loads <c>{prefix}.rdb1..rdb3</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _rdb1.LoadWeights(weights, $"{prefix}.rdb1");
        _rdb2.LoadWeights(weights, $"{prefix}.rdb2");
        _rdb3.LoadWeights(weights, $"{prefix}.rdb3");
    }

    /// <summary>Yields all conv weights.</summary>
    public IEnumerable<Tensor> EnumerateWeights() =>
        _rdb1.EnumerateWeights().Concat(_rdb2.EnumerateWeights()).Concat(_rdb3.EnumerateWeights());

    /// <summary>Forward pass with global 0.2 residual. Owns the returned tensor; does not dispose <paramref name="input"/>.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        Tensor o1 = _rdb1.Forward(backend, input);
        Tensor o2 = _rdb2.Forward(backend, o1);
        o1.Dispose();
        Tensor o3 = _rdb3.Forward(backend, o2);
        o2.Dispose();

        Tensor scaled = new Tensor(o3.Shape, DType.F32);
        backend.Scale(scaled, o3, ResidualScale);
        Tensor outT = new Tensor(o3.Shape, DType.F32);
        backend.Add(outT, scaled, input);

        o3.Dispose();
        scaled.Dispose();
        return outT;
    }
}

/// <summary>RRDBNet generator (ESRGAN / Real-ESRGAN). Conv-first → RRDB trunk → conv-body global
/// residual → nearest-neighbour upsample stages → HR convs → conv-last. Input/output are RGB in [0,1].</summary>
public sealed class RrdbNet
{
    private readonly RealEsrganConfig _config;
    private readonly Conv2dLayer _convFirst;
    private readonly Rrdb[] _body;
    private readonly Conv2dLayer _convBody;
    private readonly Conv2dLayer _convUp1;
    private readonly Conv2dLayer? _convUp2;
    private readonly Conv2dLayer _convHr;
    private readonly Conv2dLayer _convLast;
    private const float LeakySlope = 0.2f;

    /// <summary>The configuration this network was built with.</summary>
    public RealEsrganConfig Config => _config;

    /// <summary>Creates an RRDBNet for the given config.</summary>
    public RrdbNet(RealEsrganConfig config)
    {
        if (config.Scale != 2 && config.Scale != 4)
            throw new ArgumentException($"RrdbNet supports scale 2 or 4; got {config.Scale}.", nameof(config));

        _config = config;
        _convFirst = new Conv2dLayer(config.NumFeat);
        _body = new Rrdb[config.NumBlock];
        for (int i = 0; i < config.NumBlock; i++) _body[i] = new Rrdb(config.NumFeat, config.NumGrowCh);
        _convBody = new Conv2dLayer(config.NumFeat);
        _convUp1 = new Conv2dLayer(config.NumFeat);
        _convUp2 = config.Scale == 4 ? new Conv2dLayer(config.NumFeat) : null;
        _convHr = new Conv2dLayer(config.NumFeat);
        _convLast = new Conv2dLayer(3);
    }

    /// <summary>Loads weights using Real-ESRGAN / BasicSR naming (<c>conv_first</c>, <c>body.{i}.rdb{n}.conv{n}</c>,
    /// <c>conv_body</c>, <c>conv_up1/2</c>, <c>conv_hr</c>, <c>conv_last</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convFirst.LoadWeights(weights, "conv_first");
        for (int i = 0; i < _body.Length; i++) _body[i].LoadWeights(weights, $"body.{i}");
        _convBody.LoadWeights(weights, "conv_body");
        _convUp1.LoadWeights(weights, "conv_up1");
        _convUp2?.LoadWeights(weights, "conv_up2");
        _convHr.LoadWeights(weights, "conv_hr");
        _convLast.LoadWeights(weights, "conv_last");
    }

    /// <summary>Enumerates all weights for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        IEnumerable<Tensor> all = _convFirst.EnumerateWeights();
        foreach (Rrdb b in _body) all = all.Concat(b.EnumerateWeights());
        all = all.Concat(_convBody.EnumerateWeights())
                 .Concat(_convUp1.EnumerateWeights())
                 .Concat(_convHr.EnumerateWeights())
                 .Concat(_convLast.EnumerateWeights());
        if (_convUp2 is not null) all = all.Concat(_convUp2.EnumerateWeights());
        return all;
    }

    /// <summary>Runs the network. Input [1, 3, H, W] in [0,1]; output [1, 3, H*scale, W*scale] in [0,1]
    /// (not clamped — the pipeline clamps on byte conversion). Owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        Tensor feat = _convFirst.Forward(backend, input);

        // RRDB trunk.
        Tensor trunk = feat;
        bool trunkIsFeat = true;
        foreach (Rrdb block in _body)
        {
            Tensor next = block.Forward(backend, trunk);
            if (!trunkIsFeat) trunk.Dispose();
            trunk = next;
            trunkIsFeat = false;
        }

        Tensor bodyOut = _convBody.Forward(backend, trunk);
        if (!trunkIsFeat) trunk.Dispose();

        // Global residual: feat = feat + conv_body(trunk).
        Tensor merged = new Tensor(feat.Shape, DType.F32);
        backend.Add(merged, feat, bodyOut);
        feat.Dispose();
        bodyOut.Dispose();
        feat = merged;

        // Upsample stage 1.
        feat = UpsampleConv(backend, _convUp1, feat);
        // Upsample stage 2 (scale 4 only).
        if (_convUp2 is not null) feat = UpsampleConv(backend, _convUp2, feat);

        // HR convs.
        Tensor hr = _convHr.Forward(backend, feat);
        feat.Dispose();
        backend.LeakyRelu(hr, hr, LeakySlope);

        Tensor outT = _convLast.Forward(backend, hr);
        hr.Dispose();
        return outT;
    }

    private static Tensor UpsampleConv(IBackend backend, Conv2dLayer conv, Tensor input)
    {
        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];

        Tensor up = new Tensor(new TensorShape(n, c, h * 2, w * 2), DType.F32);
        backend.UpsampleNearest2D(up, input, 2, 2);
        input.Dispose();

        Tensor conved = conv.Forward(backend, up);
        up.Dispose();
        backend.LeakyRelu(conved, conved, LeakySlope);
        return conved;
    }
}
