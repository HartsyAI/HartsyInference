using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>Carried state of one Mamba layer across calls: the last <c>conv - 1</c> pre-convolution frames and the scan state, per row.</summary>
internal sealed class VoiceIsolatorMambaCarry(Tensor history, Tensor state) : IDisposable
{
    public Tensor History { get; } = history;

    public Tensor State { get; } = state;

    public void Dispose()
    {
        History.Dispose();
        State.Dispose();
    }
}

/// <summary>One selective state-space layer of the isolator (in_proj, depthwise causal conv, x_proj, dt_proj, scan, out_proj).</summary>
internal sealed class VoiceIsolatorMamba : IDisposable
{
    private readonly VoiceIsolatorConfig _config;
    private readonly Tensor _inProj;
    private readonly Tensor _conv;
    private readonly Tensor _convBias;
    private readonly Tensor _xProj;
    private readonly Tensor _dtProj;
    private readonly Tensor _dtBias;
    private readonly Tensor _decay;
    private readonly Tensor _skip;
    private readonly Tensor _outProj;

    public unsafe VoiceIsolatorMamba(VoiceIsolatorConfig config, IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _config = config;
        _inProj = weights[$"{prefix}.in_proj.weight"];
        _conv = weights[$"{prefix}.conv1d.weight"];
        _convBias = weights[$"{prefix}.conv1d.bias"];
        _xProj = weights[$"{prefix}.x_proj.weight"];
        _dtProj = weights[$"{prefix}.dt_proj.weight"];
        _dtBias = weights[$"{prefix}.dt_proj.bias"];
        _skip = weights[$"{prefix}.D"];
        _outProj = weights[$"{prefix}.out_proj.weight"];
        Tensor aLog = weights[$"{prefix}.A_log"];
        VoiceIsolatorOps.RequireF32(aLog, $"{prefix}.A_log", (long)config.MambaInner * config.MambaState);
        VoiceIsolatorOps.RequireF32(_conv, $"{prefix}.conv1d.weight", (long)config.MambaInner * config.MambaConv);
        VoiceIsolatorOps.RequireF32(_convBias, $"{prefix}.conv1d.bias", config.MambaInner);
        VoiceIsolatorOps.RequireF32(_dtBias, $"{prefix}.dt_proj.bias", config.MambaInner);
        VoiceIsolatorOps.RequireF32(_skip, $"{prefix}.D", config.MambaInner);
        _decay = new Tensor(aLog.Shape, DType.F32);
        float* src = (float*)aLog.DataPointer;
        float* dst = (float*)_decay.DataPointer;
        for (long i = 0; i < aLog.ElementCount; i++)
        {
            dst[i] = -MathF.Exp(src[i]);
        }
    }

    /// <summary>Runs the layer over <paramref name="x"/> <c>[rows * frames, width]</c> (row-major sequences) and returns <c>[1, rows * frames, width]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int rows, int frames, VoiceIsolatorMambaCarry? carry)
    {
        int width = _config.Width;
        int inner = _config.MambaInner;
        int states = _config.MambaState;
        int dtRank = _config.MambaDtRank;
        long n = (long)rows * frames;
        Tensor xz = WhisperOps.ProjectLinear(backend, x, _inProj, null, 1, (int)n, width, 2 * inner);
        Tensor u = new(new TensorShape(1, n, inner), DType.F32);
        Tensor z = new(new TensorShape(1, n, inner), DType.F32);
        VoiceIsolatorOps.SplitHalves(backend, xz, u, z);
        xz.Dispose();
        VoiceIsolatorOps.CausalDepthwiseSilu(u, _conv, _convBias, carry?.History, rows, frames, inner, _config.MambaConv);
        Tensor projected = WhisperOps.ProjectLinear(backend, u, _xProj, null, 1, (int)n, inner, dtRank + 2 * states);
        Tensor dt = new(new TensorShape(1, n, dtRank), DType.F32);
        Tensor b = new(new TensorShape(1, n, states), DType.F32);
        Tensor c = new(new TensorShape(1, n, states), DType.F32);
        VoiceIsolatorOps.SplitDtBc(backend, projected, dt, b, c);
        projected.Dispose();
        Tensor deltaPreActivation = WhisperOps.ProjectLinear(backend, dt, _dtProj, _dtBias, 1, (int)n, dtRank, inner);
        dt.Dispose();
        Tensor delta = new(deltaPreActivation.Shape, DType.F32);
        backend.Softplus(delta, deltaPreActivation);
        deltaPreActivation.Dispose();
        Tensor y = new(new TensorShape(n, inner), DType.F32);
        backend.SelectiveScan(y, u, delta, _decay, b, c, _skip, z, carry?.State, rows, frames, inner, states);
        u.Dispose();
        delta.Dispose();
        b.Dispose();
        c.Dispose();
        z.Dispose();
        Tensor output = WhisperOps.ProjectLinear(backend, y, _outProj, null, 1, (int)n, inner, width);
        y.Dispose();
        return output;
    }

    public void Dispose() => _decay.Dispose();
}

/// <summary>Forward (and, unless causal, backward) Mamba whose outputs are projected together and normalised.</summary>
internal sealed class VoiceIsolatorBidirectionalMamba : IDisposable
{
    private readonly VoiceIsolatorConfig _config;
    private readonly VoiceIsolatorMamba _forward;
    private readonly VoiceIsolatorMamba? _backward;
    private readonly Tensor _outputProj;
    private readonly Tensor _outputBias;
    private readonly Tensor _normWeight;
    private readonly Tensor _normBias;

    public VoiceIsolatorBidirectionalMamba(VoiceIsolatorConfig config, IReadOnlyDictionary<string, Tensor> weights, string prefix, bool causal)
    {
        _config = config;
        Causal = causal;
        _forward = new VoiceIsolatorMamba(config, weights, $"{prefix}.forward_blocks");
        _backward = causal ? null : new VoiceIsolatorMamba(config, weights, $"{prefix}.backward_blocks");
        _outputProj = weights[$"{prefix}.output_proj.weight"];
        _outputBias = weights[$"{prefix}.output_proj.bias"];
        _normWeight = weights[$"{prefix}.norm.weight"];
        _normBias = weights[$"{prefix}.norm.bias"];
    }

    /// <summary>True when only the forward scan exists.</summary>
    public bool Causal { get; }

    /// <summary>Runs over <paramref name="x"/> <c>[rows * frames, width]</c>; <paramref name="carry"/> continues the forward scan of a stream.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int rows, int frames, VoiceIsolatorMambaCarry? carry)
    {
        int width = _config.Width;
        long n = (long)rows * frames;
        Tensor forward = _forward.Forward(backend, x, rows, frames, carry);
        Tensor mixed;
        if (Causal)
        {
            mixed = new Tensor(new TensorShape(1, n, width), DType.F32);
            backend.Add(mixed, forward, x);
            forward.Dispose();
            return Normalise(backend, mixed, n, width, width);
        }
        if (carry is not null)
        {
            throw new NotSupportedException("A bidirectional Mamba cannot carry a stream state.");
        }
        Tensor flipped = new(x.Shape, DType.F32);
        VoiceIsolatorOps.FlipFrames(flipped, x, rows, frames, width);
        Tensor backward = _backward!.Forward(backend, flipped, rows, frames, null);
        flipped.Dispose();
        Tensor backwardFlipped = new(backward.Shape, DType.F32);
        VoiceIsolatorOps.FlipFrames(backwardFlipped, backward, rows, frames, width);
        backward.Dispose();
        mixed = new Tensor(new TensorShape(1, n, 2 * width), DType.F32);
        VoiceIsolatorOps.ConcatPlusResidual(backend, mixed, forward, backwardFlipped, x);
        forward.Dispose();
        backwardFlipped.Dispose();
        return Normalise(backend, mixed, n, 2 * width, width);
    }

    private Tensor Normalise(IBackend backend, Tensor mixed, long n, int inWidth, int width)
    {
        Tensor projected = WhisperOps.ProjectLinear(backend, mixed, _outputProj, _outputBias, 1, (int)n, inWidth, width);
        mixed.Dispose();
        Tensor normalised = new(projected.Shape, DType.F32);
        backend.LayerNorm(normalised, projected, _normWeight, _normBias, 1e-5f);
        projected.Dispose();
        return normalised;
    }

    public void Dispose()
    {
        _forward.Dispose();
        _backward?.Dispose();
    }
}
