using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>The glm5_next vision tower (transformers <c>Glm5NextVisionModel</c> at 52bac5d0) on <see cref="IBackend"/> ops with host F32 rows: a 3D-conv patch
/// embedding, blocks of per-head q/k RMSNorm, axial 2D RoPE, non-causal attention per frame and a clamped SiLU-GLU MLP with biases, a final RMSNorm, a
/// strided-conv downsample over each merge block, and the patch merger. The merger's rows replace the image and video placeholders.</summary>
/// <remarks>Input rows are the processor's flattened patches in merge-block order (<c>[channel, time, y, x]</c> per row). The rope base is the transformers default,
/// 10000, because the published vision config sets none.</remarks>
public sealed class Glm5NextVisionTower : IDisposable
{
    private const float MergerNormEps = 1e-5f;

    private readonly IBackend _backend;
    private readonly Glm5NextVisionConfig _c;
    private readonly float _ropeTheta;
    private readonly List<Tensor> _owned = [];
    private readonly Tensor _patchW, _postNorm, _downW, _mergerNormW, _mergerNormB, _mergerProj, _mergerGate, _mergerUp, _mergerDown;
    private readonly float[] _patchB, _downB;
    private readonly Block[] _blocks;

    /// <summary>Loads the tower from <paramref name="source"/> (keys <c>model.visual.*</c>).</summary>
    /// <exception cref="InvalidDataException">The config is not one the tower runs (see <see cref="Glm5NextVisionConfig.Validate"/>), or a tensor is
    /// missing or does not have the shape the config implies.</exception>
    public Glm5NextVisionTower(IBackend backend, Glm5NextVisionConfig config, IErnie5WeightSource source, float ropeTheta = 10000f)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        config.Validate();
        (_backend, _c, _ropeTheta) = (backend, config, ropeTheta);
        string v = Glm5NextWeightNames.Visual;
        int h = config.Hidden, o = config.OutHidden, m = config.SpatialMergeSize, inter = config.Intermediate, mi = config.MergerIntermediate;
        _blocks = new Block[config.Depth];
        try
        {
            // The patch embedding is a Conv3d [h, 3, T, P, P] and the downsample a Conv2d [o, h, m, m]: their rows are the same values flattened.
            _patchW = Own(Flat(source, v + "patch_embed.proj.weight", h, PatchRowWidth));
            _patchB = Host(source, v + "patch_embed.proj.bias", h);
            _postNorm = Own(HostTensor(source, v + "post_layernorm.weight", h));
            _downW = Own(Flat(source, v + "downsample.weight", o, h * m * m));
            _downB = Host(source, v + "downsample.bias", o);
            _mergerProj = Own(Matrix(source, v + "merger.proj.weight", o, o));
            _mergerNormW = Own(HostTensor(source, v + "merger.post_projection_norm.weight", o));
            _mergerNormB = Own(HostTensor(source, v + "merger.post_projection_norm.bias", o));
            _mergerGate = Own(Matrix(source, v + "merger.gate_proj.weight", mi, o));
            _mergerUp = Own(Matrix(source, v + "merger.up_proj.weight", mi, o));
            _mergerDown = Own(Matrix(source, v + "merger.down_proj.weight", o, mi));
            for (int i = 0; i < config.Depth; i++) _blocks[i] = LoadBlock(source, $"{v}blocks.{i}.", h, inter, config.HeadDim);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Width of one flattened patch row: <c>3 · TemporalPatchSize · PatchSize²</c>.</summary>
    public int PatchRowWidth => 3 * _c.TemporalPatchSize * _c.PatchSize * _c.PatchSize;

    /// <summary>Runs the tower and returns the merger rows <c>[tokens / merge², OutHidden]</c>, grids in order.</summary>
    /// <param name="patches"><c>[tokens, PatchRowWidth]</c>: every grid's rows, each grid frame by frame in merge-block order.</param>
    /// <param name="grids">One <c>(t, h, w)</c> per image or clip, in patches; <c>h</c> and <c>w</c> are multiples of the merge size.</param>
    public float[] Forward(ReadOnlySpan<float> patches, IReadOnlyList<(int T, int H, int W)> grids)
    {
        float[] down = Downsample(patches, grids);
        return Merge(down, down.Length / _c.OutHidden);
    }

    /// <summary>The tower up to the downsample, <c>[tokens / merge², OutHidden]</c>: transformers' <c>last_hidden_state</c>, before the merger.</summary>
    internal float[] Downsample(ReadOnlySpan<float> patches, IReadOnlyList<(int T, int H, int W)> grids)
    {
        int h = _c.Hidden, hd = _c.HeadDim, m = _c.SpatialMergeSize;
        int tokens = 0;
        foreach ((int t, int gh, int gw) in grids)
        {
            if (gh % m != 0 || gw % m != 0 || t < 1) throw new ArgumentException($"Grid ({t}, {gh}, {gw}) must have t >= 1 and sides that are multiples of {m}.", nameof(grids));
            tokens += t * gh * gw;
        }
        if (patches.Length != (long)tokens * PatchRowWidth) throw new ArgumentException($"patches must hold {tokens} rows of {PatchRowWidth} values.", nameof(patches));

        float[] x = HostOps.Project(_backend, patches, tokens, PatchRowWidth, _patchW, h);
        AddBias(x, _patchB, tokens);
        (float[] cos, float[] sin, List<(int Start, int Length)> segments) = Positions(grids, tokens);
        using Tensor cosT = HostMath.Tensor(cos, tokens, hd), sinT = HostMath.Tensor(sin, tokens, hd);
        foreach (Block b in _blocks) x = RunBlock(b, x, tokens, cosT, sinT, segments);
        HostOps.RmsNorm(_backend, x, h, _postNorm, _c.RmsNormEps);

        // The strided conv over each m x m merge block: rows [block, (my, mx), channel] become [block, channel, my, mx] features.
        int mm = m * m, blocks = tokens / mm, o = _c.OutHidden;
        float[] folded = new float[blocks * h * mm];
        for (int b = 0; b < blocks; b++)
            for (int r = 0; r < mm; r++)
                for (int ch = 0; ch < h; ch++) folded[b * h * mm + ch * mm + r] = x[(b * mm + r) * h + ch];
        float[] down = HostOps.Project(_backend, folded, blocks, h * mm, _downW, o);
        AddBias(down, _downB, blocks);
        return down;
    }

    private float[] RunBlock(Block b, float[] x, int tokens, Tensor cosT, Tensor sinT, List<(int Start, int Length)> segments)
    {
        int h = _c.Hidden, hd = _c.HeadDim;
        float[] n1 = (float[])x.Clone();
        HostOps.RmsNorm(_backend, n1, h, b.Norm1, _c.RmsNormEps);
        float[] qkv = HostOps.Project(_backend, n1, tokens, h, b.Qkv, 3 * h);
        AddBias(qkv, b.QkvBias, tokens);
        float[] q = new float[tokens * h], k = new float[tokens * h], v = new float[tokens * h];
        for (int t = 0; t < tokens; t++)
        {
            qkv.AsSpan(t * 3 * h, h).CopyTo(q.AsSpan(t * h, h));
            qkv.AsSpan(t * 3 * h + h, h).CopyTo(k.AsSpan(t * h, h));
            qkv.AsSpan(t * 3 * h + 2 * h, h).CopyTo(v.AsSpan(t * h, h));
        }
        HostOps.RmsNorm(_backend, q, hd, b.QNorm, _c.RmsNormEps);
        HostOps.RmsNorm(_backend, k, hd, b.KNorm, _c.RmsNormEps);
        Rotate(q, tokens, cosT, sinT);
        Rotate(k, tokens, cosT, sinT);

        float[] attended = new float[tokens * h];
        foreach ((int start, int len) in segments) AttendSegment(q, k, v, start, len, attended);
        float[] proj = HostOps.Project(_backend, attended, tokens, h, b.Proj, h);
        AddBias(proj, b.ProjBias, tokens);
        for (int i = 0; i < x.Length; i++) x[i] += proj[i];

        float[] n2 = (float[])x.Clone();
        HostOps.RmsNorm(_backend, n2, h, b.Norm2, _c.RmsNormEps);
        int inter = _c.Intermediate;
        float[] gate = HostOps.Project(_backend, n2, tokens, h, b.Gate, inter), up = HostOps.Project(_backend, n2, tokens, h, b.Up, inter);
        AddBias(gate, b.GateBias, tokens);
        AddBias(up, b.UpBias, tokens);
        float[] down = Glu(gate, up, tokens, inter, _c.SwigluLimit, b.Down, h, b.DownBias);
        for (int i = 0; i < x.Length; i++) x[i] += down[i];
        return x;
    }

    // Rotate-half over each head's width with the shared per-token tables.
    private void Rotate(float[] data, int tokens, Tensor cosT, Tensor sinT)
    {
        using Tensor t = HostMath.Tensor(data, 1, tokens, _c.Heads, _c.HeadDim);
        _backend.Rotary(RotaryCall.TokenMajor(t, cosT, sinT, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));
        t.AsReadOnlySpan<float>().CopyTo(data);
    }

    // Full attention inside one frame: [len, heads, hd] rows to head-major tensors, SDPA without a mask, back to token-major rows.
    private void AttendSegment(float[] q, float[] k, float[] v, int start, int len, float[] output)
    {
        int heads = _c.Heads, hd = _c.HeadDim, h = _c.Hidden;
        using Tensor qM = HeadMajor(q, start, len), kM = HeadMajor(k, start, len), vM = HeadMajor(v, start, len);
        using Tensor o = new(new TensorShape(1, heads, len, hd), DType.F32);
        _backend.ScaledDotProductAttention(o, qM, kM, vM, null, 1f / MathF.Sqrt(hd));
        ReadOnlySpan<float> r = o.AsReadOnlySpan<float>();
        for (int t = 0; t < len; t++)
            for (int head = 0; head < heads; head++)
                r.Slice((head * len + t) * hd, hd).CopyTo(output.AsSpan((start + t) * h + head * hd, hd));
    }

    private Tensor HeadMajor(float[] rows, int start, int len)
    {
        int heads = _c.Heads, hd = _c.HeadDim, h = _c.Hidden;
        Tensor t = new(new TensorShape(1, heads, len, hd), DType.F32);
        Span<float> dst = t.AsSpan<float>();
        for (int i = 0; i < len; i++)
            for (int head = 0; head < heads; head++)
                rows.AsSpan((start + i) * h + head * hd, hd).CopyTo(dst.Slice((head * len + i) * hd, hd));
        return t;
    }

    // Patch merger: proj, LayerNorm, exact GELU, then the clamped SiLU-GLU without biases.
    private float[] Merge(float[] x, int rows)
    {
        int o = _c.OutHidden;
        float[] y = HostOps.Project(_backend, x, rows, o, _mergerProj, o);
        HostOps.LayerNorm(_backend, y, o, _mergerNormW, _mergerNormB, MergerNormEps);
        using (Tensor input = HostMath.Tensor(y, rows, o))
        using (Tensor act = new(new TensorShape(rows, o), DType.F32))
        {
            _backend.GeluErf(act, input);
            act.AsReadOnlySpan<float>().CopyTo(y);
        }
        int inter = _c.MergerIntermediate;
        float[] gate = HostOps.Project(_backend, y, rows, o, _mergerGate, inter), up = HostOps.Project(_backend, y, rows, o, _mergerUp, inter);
        return Glu(gate, up, rows, inter, _c.SwigluLimit, _mergerDown, o, null);
    }

    private float[] Glu(float[] gate, float[] up, int rows, int inter, float limit, Tensor downW, int outWidth, float[]? downBias)
    {
        Tensor g = HostMath.Tensor(gate, rows, inter), u = HostMath.Tensor(up, rows, inter);
        using Tensor act = GatedFfn.SwiGlu(_backend, ActivationKind.Silu, g, u, limit, 0f, 0f, 0f);
        float[] y = HostOps.Project(_backend, act.AsReadOnlySpan<float>(), rows, inter, downW, outWidth);
        if (downBias is not null) AddBias(y, downBias, rows);
        return y;
    }

    // Rope rows per token and one attention segment per frame. Each frame repeats the same block-major positions.
    private (float[] Cos, float[] Sin, List<(int Start, int Length)> Segments) Positions(IReadOnlyList<(int T, int H, int W)> grids, int tokens)
    {
        int hd = _c.HeadDim;
        float[] cos = new float[tokens * hd], sin = new float[tokens * hd];
        List<(int, int)> segments = [];
        int at = 0;
        foreach ((int t, int gh, int gw) in grids)
        {
            (float[] c, float[] s) = QwenVlOps.BuildRope(gh, gw, hd, _c.SpatialMergeSize, _ropeTheta);
            for (int f = 0; f < t; f++)
            {
                c.CopyTo(cos, at * hd);
                s.CopyTo(sin, at * hd);
                segments.Add((at, gh * gw));
                at += gh * gw;
            }
        }
        return (cos, sin, segments);
    }

    private Block LoadBlock(IErnie5WeightSource source, string p, int h, int inter, int hd) => new(
        Norm1: Own(HostTensor(source, p + "norm1.weight", h)), Norm2: Own(HostTensor(source, p + "norm2.weight", h)),
        Qkv: Own(Matrix(source, p + "attn.qkv.weight", 3 * h, h)), QkvBias: Host(source, p + "attn.qkv.bias", 3 * h),
        Proj: Own(Matrix(source, p + "attn.proj.weight", h, h)), ProjBias: Host(source, p + "attn.proj.bias", h),
        QNorm: Own(HostTensor(source, p + "attn.q_norm.weight", hd)), KNorm: Own(HostTensor(source, p + "attn.k_norm.weight", hd)),
        Gate: Own(Matrix(source, p + "mlp.gate_proj.weight", inter, h)), GateBias: Host(source, p + "mlp.gate_proj.bias", inter),
        Up: Own(Matrix(source, p + "mlp.up_proj.weight", inter, h)), UpBias: Host(source, p + "mlp.up_proj.bias", inter),
        Down: Own(Matrix(source, p + "mlp.down_proj.weight", h, inter)), DownBias: Host(source, p + "mlp.down_proj.bias", h));

    // A 2-D weight with exactly the shape the config implies.
    private Tensor Matrix(IErnie5WeightSource source, string name, int rows, int cols)
    {
        Tensor t = Read(source, name);
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
        {
            t.Dispose();
            throw new InvalidDataException($"{name} must be [{rows}, {cols}]; got {t.Shape}.");
        }
        return t;
    }

    // A conv weight stored flat, with rows x cols values: the same values in the row-major [rows, cols] layout.
    private Tensor Flat(IErnie5WeightSource source, string name, int rows, int cols)
    {
        Tensor t = Read(source, name);
        if (t.ElementCount != (long)rows * cols)
        {
            t.Dispose();
            throw new InvalidDataException($"{name} holds {t.ElementCount} values; the config implies {rows} x {cols}.");
        }
        return t.Reshape(new TensorShape(rows, cols));
    }

    // A checkpoint read that reports a missing tensor as the constructor documents.
    private static Tensor Read(IErnie5WeightSource source, string name)
    {
        try
        {
            return source.Read(name);
        }
        catch (KeyNotFoundException ex)
        {
            throw new InvalidDataException($"The vision tower has no tensor {name}.", ex);
        }
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private static void AddBias(float[] y, float[] bias, int rows)
    {
        int n = bias.Length;
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++) y[r * n + j] += bias[j];
    }

    private static Tensor HostTensor(IErnie5WeightSource source, string name, long elements)
    {
        float[] v = Host(source, name, elements);
        return HostMath.Tensor(v, v.Length);
    }

    private static float[] Host(IErnie5WeightSource source, string name, long elements)
    {
        Tensor raw = Read(source, name);
        Tensor f32 = ProjectionOps.EnsureF32(raw);
        try
        {
            if (f32.ElementCount != elements) throw new InvalidDataException($"'{name}' holds {f32.ElementCount} values; the config implies {elements}.");
            return f32.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            if (!ReferenceEquals(f32, raw)) f32.Dispose();
            raw.Dispose();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    private sealed record Block(Tensor Norm1, Tensor Norm2, Tensor Qkv, float[] QkvBias, Tensor Proj, float[] ProjBias, Tensor QNorm, Tensor KNorm,
        Tensor Gate, float[] GateBias, Tensor Up, float[] UpBias, Tensor Down, float[] DownBias);
}
