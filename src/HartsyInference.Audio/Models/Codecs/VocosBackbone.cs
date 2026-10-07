using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Generic Vocos ConvNeXt backbone (real upstream <c>vocos.py</c>'s <c>VocosBackbone</c>, non-AdaLN
/// variant only — <c>adanorm_num_embeddings=None</c>): <c>embed(Conv1d k7) → norm → N× ConvNeXtBlock → final
/// norm</c>, each block <c>x += γ·pwconv2(GELU(pwconv1(LN(dwconv(x)))))</c>. Channels-first in, channels-last
/// out (matching the real <c>forward</c>'s documented <c>(B, L, H)</c> return).</summary>
/// <remarks>This shape is already duplicated five times in this codebase under one-off names (<c>VocosDecoder</c>,
/// <c>Vocos</c>, <c>SparkVocosBackbone</c>, <c>VibeVoiceConvNeXtBlock</c>, <c>Mert2ConvNextBlock</c>) — this is the
/// first config-driven, shared copy; new callers should reuse this rather than add a sixth.</remarks>
public sealed unsafe class VocosBackbone : IDisposable
{
    private const float LayerNormEps = 1e-6f;

    private readonly int _inputChannels, _dim, _intermediateDim, _numLayers;
    private Tensor? _embedW, _embedB;
    private Tensor? _normW, _normB;
    private Tensor?[] _dwW = [], _dwB = [], _blockNormW = [], _blockNormB = [];
    private Tensor?[] _pw1W = [], _pw1B = [], _pw2W = [], _pw2B = [], _gamma = [];
    private Tensor? _finalNormW, _finalNormB;
    private int _disposed;

    public VocosBackbone(int inputChannels, int dim, int intermediateDim, int numLayers)
    {
        _inputChannels = inputChannels;
        _dim = dim;
        _intermediateDim = intermediateDim;
        _numLayers = numLayers;
        _dwW = new Tensor?[numLayers]; _dwB = new Tensor?[numLayers];
        _blockNormW = new Tensor?[numLayers]; _blockNormB = new Tensor?[numLayers];
        _pw1W = new Tensor?[numLayers]; _pw1B = new Tensor?[numLayers];
        _pw2W = new Tensor?[numLayers]; _pw2B = new Tensor?[numLayers];
        _gamma = new Tensor?[numLayers];
    }

    /// <summary><paramref name="prefix"/> is this backbone's own module path (e.g. <c>"encoder"</c> — the real
    /// upstream wraps a <c>VocosBackbone</c> inside <c>nn.Sequential(VocosBackbone(...), nn.Linear(...))</c>, so a
    /// typical caller's prefix is <c>"encoder.0"</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _embedW = WhisperOps.EnsureF32(w[$"{prefix}.embed.weight"]);
        _embedB = WhisperOps.EnsureF32(w[$"{prefix}.embed.bias"]);
        _normW = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _normB = WhisperOps.EnsureF32(w[$"{prefix}.norm.bias"]);
        for (int i = 0; i < _numLayers; i++)
        {
            string p = $"{prefix}.convnext.{i}";
            _dwW[i] = WhisperOps.EnsureF32(w[$"{p}.dwconv.weight"]); _dwB[i] = WhisperOps.EnsureF32(w[$"{p}.dwconv.bias"]);
            _blockNormW[i] = WhisperOps.EnsureF32(w[$"{p}.norm.weight"]); _blockNormB[i] = WhisperOps.EnsureF32(w[$"{p}.norm.bias"]);
            _pw1W[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv1.weight"]); _pw1B[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv1.bias"]);
            _pw2W[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv2.weight"]); _pw2B[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv2.bias"]);
            _gamma[i] = WhisperOps.EnsureF32(w[$"{p}.gamma"]);
        }
        _finalNormW = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{prefix}.final_layer_norm.bias"]);
    }

    /// <summary>Runs <paramref name="input"/> <c>[1, inputChannels, T]</c> (channels-first) through the backbone,
    /// returning <c>[1, T, dim]</c> (channels-last, matching the real <c>forward</c>'s documented output layout).</summary>
    public Tensor Forward(IBackend backend, Tensor input, int t)
    {
        ThrowIfDisposed();
        if (_embedW is null) throw new InvalidOperationException("VocosBackbone weights not loaded.");

        Tensor embedded = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Conv1d(embedded, input, _embedW!, _embedB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);

        Tensor chLast = new(new TensorShape(1, t, _dim), DType.F32);
        backend.Transpose2D(chLast, embedded, _dim, t);
        embedded.Dispose();
        Tensor x = new(chLast.Shape, DType.F32);
        backend.LayerNorm(x, chLast, _normW!, _normB!, LayerNormEps);
        chLast.Dispose();

        for (int i = 0; i < _numLayers; i++)
        {
            Tensor next = ConvNeXtBlock(backend, x, t, i);
            x.Dispose();
            x = next;
        }

        Tensor outT = new(x.Shape, DType.F32);
        backend.LayerNorm(outT, x, _finalNormW!, _finalNormB!, LayerNormEps);
        x.Dispose();
        return outT;
    }

    /// <summary>One <c>ConvNeXtBlock</c>: <c>residual + γ·pwconv2(GELU(pwconv1(LN(dwconv(x)))))</c>. Input/output
    /// both channels-last <c>[1, T, dim]</c>; channels-first internally for the depthwise conv.</summary>
    private Tensor ConvNeXtBlock(IBackend backend, Tensor xChLast, int t, int i)
    {
        Tensor chFirst = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Transpose2D(chFirst, xChLast, t, _dim);

        Tensor dw = new(new TensorShape(1, _dim, t), DType.F32);
        backend.Conv1d(dw, chFirst, _dwW[i]!, _dwB[i], stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: _dim);
        chFirst.Dispose();

        Tensor dwChLast = new(new TensorShape(1, t, _dim), DType.F32);
        backend.Transpose2D(dwChLast, dw, _dim, t);
        dw.Dispose();

        Tensor normed = new(dwChLast.Shape, DType.F32);
        backend.LayerNorm(normed, dwChLast, _blockNormW[i]!, _blockNormB[i]!, LayerNormEps);
        dwChLast.Dispose();

        Tensor h = WhisperOps.ProjectLinear(backend, normed, _pw1W[i]!, _pw1B[i], 1, t, _dim, _intermediateDim);
        normed.Dispose();
        backend.GeluErf(h, h);
        Tensor pw2 = WhisperOps.ProjectLinear(backend, h, _pw2W[i]!, _pw2B[i], 1, t, _intermediateDim, _dim);
        h.Dispose();

        ScaleByGammaInPlace(pw2, _gamma[i]!, t);

        Tensor outT = new(xChLast.Shape, DType.F32);
        AddInto(outT, xChLast, pw2);
        pw2.Dispose();
        return outT;
    }

    private void ScaleByGammaInPlace(Tensor x, Tensor gamma, int t)
    {
        float* xp = (float*)x.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        for (int i = 0; i < t; i++)
        {
            float* row = xp + (long)i * _dim;
            for (int c = 0; c < _dim; c++) row[c] *= gp[c];
        }
    }

    private static void AddInto(Tensor dst, Tensor a, Tensor b)
    {
        float* dp = (float*)dst.DataPointer;
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] = ap[i] + bp[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_embedW, _embedB, _normW, _normB, _finalNormW, _finalNormB];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        for (int i = 0; i < _numLayers; i++)
        {
            Tensor?[] layer = [_dwW[i], _dwB[i], _blockNormW[i], _blockNormB[i], _pw1W[i], _pw1B[i], _pw2W[i], _pw2B[i], _gamma[i]];
            foreach (Tensor? t in layer) if (t is not null) yield return t;
        }
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(VocosBackbone));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
