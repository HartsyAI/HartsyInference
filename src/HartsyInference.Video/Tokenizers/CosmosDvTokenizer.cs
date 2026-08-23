using HartsyInference.Core.Backends;
using HartsyInference.Core.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Video.Tokenizers;

/// <summary>Pure-C# Cosmos-Tokenize1-DV8x16x16-720p discrete video tokenizer. Composes the three well-specified,
/// deterministic pieces of the DV codec — a 2-level 3D <see cref="HaarWavelet3D"/> front-end, a causal-conv-3D
/// autoencoder (<see cref="CausalConv3d"/> / <see cref="AvgDown3D"/> / <see cref="DupUp3D"/>), and
/// <see cref="Fsq"/> finite-scalar quantization with levels <c>[8,8,8,5,5,5]</c> (64,000-entry codebook) — into
/// RGB↔token conversion. Reusable by any Cosmos-lineage AR world model that shares the DV token space.
///
/// <para><b>What is fully implemented and tested</b>: the token-space math — FSQ index packing
/// (<see cref="QuantizeToCodes"/> / <see cref="CodesToContinuous"/>, bit-exact vs NVIDIA's FSQuantizer since FSQ
/// is deterministic), the latent-grid geometry (<see cref="LatentGrid"/>), and the invertible Haar front-end.</para>
///
/// <para><b>What is integration-pending</b>: the learned convolutional autoencoder's exact stage schedule
/// (channels, resnet/attention blocks) is not recoverable without NVIDIA's <c>encoder.jit</c> weight dump
/// (research-doc Open Q §10). <see cref="Encode"/> runs the <see cref="CosmosDvEncoder"/> stack once
/// <see cref="LoadWeights"/> supplies real weights, and throws a clear <see cref="NotSupportedException"/> until
/// then; <see cref="Decode"/> has no stack at all and always throws. The AR pipeline's parity gate (encode 9
/// frames → token IDs match <c>encoder.jit</c>) validates the encode path end-to-end at integration.</para></summary>
public sealed unsafe class CosmosDvTokenizer : IDiscreteVideoTokenizer, IDisposable
{
    private readonly CosmosDvTokenizerConfig _cfg;
    private readonly int[] _levels;
    private readonly int _vocabSize;
    private CosmosDvEncoder? _encoder;
    private int _disposed;

    /// <summary>Builds the tokenizer for the given config (default DV8x16x16). Conv weights are loaded separately
    /// via <see cref="LoadWeights"/>; the FSQ / Haar / grid helpers work without weights.</summary>
    public CosmosDvTokenizer(CosmosDvTokenizerConfig? config = null)
    {
        _cfg = config ?? CosmosDvTokenizerConfig.Dv8x16x16;
        _levels = _cfg.Levels;
        if (_cfg.LatentChannels != _levels.Length)
            throw new ArgumentException($"LatentChannels ({_cfg.LatentChannels}) must equal Levels.Length ({_levels.Length}).");
        _vocabSize = Fsq.VocabSize(_levels);
    }

    /// <inheritdoc/>
    public int VocabSize => _vocabSize;

    /// <summary>Per-axis FSQ levels (<c>[8,8,8,5,5,5]</c>).</summary>
    public ReadOnlySpan<int> Levels => _levels;

    /// <inheritdoc/>
    public (int T, int H, int W) LatentGrid(int frames, int height, int width)
    {
        int rt = _cfg.CompressionRatio[0], rh = _cfg.CompressionRatio[1], rw = _cfg.CompressionRatio[2];
        // Temporal compression is causal: a leading frame stays a frame, the rest compress by rt
        // ((F-1)/rt + 1), matching Cosmos latent_chunk_duration = (pixel_chunk-1)/8 + 1.
        if ((frames - 1) % rt != 0)
            throw new ArgumentException($"frames-1 ({frames - 1}) must be divisible by temporal ratio {rt} (valid: 1, 9, 17, ..., 33).");
        if (height % rh != 0 || width % rw != 0)
            throw new ArgumentException($"height/width ({height}×{width}) must be divisible by spatial ratio {rh}×{rw}.");
        return ((frames - 1) / rt + 1, height / rh, width / rw);
    }

    /// <summary>Quantizes a continuous latent <paramref name="continuous"/> <c>[B, C=6, T, H, W]</c> (the conv
    /// encoder output) into flattened codes <c>[B, T·H·W]</c> Int32, row-major in <c>(t, h, w)</c>. This is the
    /// deterministic FSQ index-packing path — the AR vocab.</summary>
    public Tensor QuantizeToCodes(Tensor continuous)
    {
        if (continuous.Shape.Rank != 5) throw new ArgumentException($"continuous must be [B,C,T,H,W], got {continuous.Shape}.");
        int b = (int)continuous.Shape[0], c = (int)continuous.Shape[1];
        int t = (int)continuous.Shape[2], h = (int)continuous.Shape[3], w = (int)continuous.Shape[4];
        if (c != _levels.Length) throw new ArgumentException($"channel count {c} must equal FSQ dims {_levels.Length}.");
        int n = t * h * w;

        // Cosmos FSQuantizer: parity-aware tanh bound with atanh shift + eps 1e-3 (lucidrains/Cosmos), NOT the
        // shared Core.Fsq (which uses atan + eps 1e-6 — a different, non-Cosmos bound). Channels-first
        // [B,C,T,H,W] → the 6 channels are the FSQ axes, packed by the mixed-radix basis. Runs host-side.
        Tensor codes = new Tensor(new TensorShape(b, n), DType.I32);
        int* cp = (int*)codes.DataPointer;
        float* src = (float*)continuous.DataPointer;

        Span<float> halfL = stackalloc float[c];
        Span<float> shift = stackalloc float[c];
        Span<int> halfW = stackalloc int[c];
        Span<int> basis = stackalloc int[c];
        const float eps = 1e-3f;
        int place = 1;
        for (int d = 0; d < c; d++)
        {
            int L = _levels[d];
            halfL[d] = (L - 1) * (1f + eps) / 2f;
            float offset = (L % 2 == 0) ? 0.5f : 0f;
            shift[d] = MathF.Atanh(offset / halfL[d]);
            halfW[d] = L / 2;
            basis[d] = place; place *= L;
        }
        for (int bi = 0; bi < b; bi++)
            for (int i = 0; i < n; i++)
            {
                int code = 0;
                for (int d = 0; d < c; d++)
                {
                    int L = _levels[d];
                    float offset = (L % 2 == 0) ? 0.5f : 0f;
                    float z = src[((long)bi * c + d) * n + i];
                    float bounded = MathF.Tanh(z + shift[d]) * halfL[d] - offset;
                    int zhat = (int)MathF.Round(bounded, MidpointRounding.ToEven);
                    int digit = zhat + halfW[d];
                    if (digit < 0) digit = 0; else if (digit >= L) digit = L - 1;
                    code += digit * basis[d];
                }
                cp[(long)bi * n + i] = code;
            }
        return codes;
    }

    /// <summary>Inverse of <see cref="QuantizeToCodes"/>: flattened codes <c>[B, T·H·W]</c> → continuous latent
    /// <c>[B, C=6, T, H, W]</c> (fed to the conv decoder), using the lucidrains/Cosmos FSQ dequant convention.</summary>
    public Tensor CodesToContinuous(Tensor codes, int t, int h, int w)
    {
        if (codes.Shape.Rank != 2) throw new ArgumentException($"codes must be [B, N], got {codes.Shape}.");
        int b = (int)codes.Shape[0], n = (int)codes.Shape[1], c = _levels.Length;
        if (n != t * h * w) throw new ArgumentException($"codes length {n} != T·H·W ({t}·{h}·{w}).");

        using Tensor zLast = new Tensor(new TensorShape(b, n, c), DType.F32);
        Fsq.Dequantize(zLast, codes, _levels);

        Tensor zFirst = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* src = (float*)zLast.DataPointer;
        float* dst = (float*)zFirst.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < n; i++)
                    dst[((long)bi * c + ci) * n + i] = src[((long)bi * n + i) * c + ci];
        return zFirst;
    }

    /// <inheritdoc/>
    public Tensor Encode(IBackend backend, Tensor video)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new NotSupportedException("CosmosDvTokenizer.Encode requires the DV tokenizer weights. Load the shipped " +
                "model.pt (network.* set) via CosmosDvTokenizerConverter and pass them to LoadWeights first.");
        Tensor continuous = _encoder.Encode(backend, video);   // [1,6,Tl,Hl,Wl] pre-FSQ
        Tensor codes = QuantizeToCodes(continuous);
        continuous.Dispose();
        return codes;
    }

    /// <summary>Runs the encoder up to the pre-FSQ continuous latent <c>[1,6,Tl,Hl,Wl]</c> (before quantization).
    /// Exposed for parity verification against <c>encoder.jit</c>'s continuous output.</summary>
    public Tensor EncodeContinuous(IBackend backend, Tensor video)
    {
        ThrowIfDisposed();
        if (_encoder is null)
            throw new NotSupportedException("CosmosDvTokenizer.EncodeContinuous requires loaded DV tokenizer weights.");
        return _encoder.Encode(backend, video);
    }

    /// <inheritdoc/>
    /// <remarks>Unimplemented, and refused rather than approximated: the decoder's conv stage schedule is not
    /// recoverable without NVIDIA's <c>decoder.jit</c> weight dump (research-doc Open Q §10), so there is no stack to
    /// run. <see cref="CodesToContinuous"/> and <see cref="HaarWavelet3D.Inverse"/> — the two deterministic halves of
    /// this path — are implemented and tested; only the learned middle is missing.</remarks>
    public Tensor Decode(IBackend backend, Tensor codes, int latentT, int latentH, int latentW)
    {
        ThrowIfDisposed();
        throw new NotSupportedException("CosmosDvTokenizer.Decode is not implemented: the DV decoder's conv stage " +
            "schedule requires the converted decoder.jit weights (research-doc Open Q §10), which are not wired. " +
            "Encode / EncodeContinuous / the FSQ token-space helpers are available.");
    }

    /// <summary>Wires the DV encoder from the loaded <c>network.*</c> tokenizer weights (F32; produced by
    /// <see cref="HartsyInference.ModelAssets.CheckpointConverters.CosmosDvTokenizerConverter"/> with
    /// <c>castToF32: true</c>). Keys are the prefix-stripped module names (e.g. <c>encoder.conv_in.0.conv3d.weight</c>,
    /// <c>quant_conv.conv3d.weight</c>).</summary>
    public void LoadWeights(Dictionary<string, Tensor> weightsF32)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(weightsF32);
        _encoder?.Dispose();
        _encoder = new CosmosDvEncoder(weightsF32);
    }

    /// <summary>Conv weights for GPU preload — always empty; <see cref="CosmosDvEncoder"/> owns the loaded weights and
    /// faults them to device itself, and there is no decoder stack to preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => [];

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CosmosDvTokenizer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _encoder?.Dispose();
        _encoder = null;
    }
}
