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
/// (research-doc Open Q §10). <see cref="Encode"/> / <see cref="Decode"/> run the generic sequential conv stack
/// loaded via <see cref="LoadWeights"/>; until real weights + the resolved schedule are supplied they throw a
/// clear <see cref="NotSupportedException"/>. The AR pipeline's parity gate (encode 9 frames → token IDs match
/// <c>encoder.jit</c>) validates this end-to-end at integration.</para></summary>
public sealed unsafe class CosmosDvTokenizer : IDiscreteVideoTokenizer, IDisposable
{
    private readonly CosmosDvTokenizerConfig _cfg;
    private readonly int[] _levels;
    private readonly int _vocabSize;
    private CausalConv3d[]? _encoderConvs;
    private CausalConv3d[]? _decoderConvs;
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

        // FSQ consumes channels-last [B, N, C]; the conv output is channels-first [B, C, T, H, W]. Rearrange host-side
        // (the tokenizer runs once per clip — not a GPU hot path).
        using Tensor zLast = new Tensor(new TensorShape(b, n, c), DType.F32);
        float* src = (float*)continuous.DataPointer;
        float* dst = (float*)zLast.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < n; i++)
                    dst[((long)bi * n + i) * c + ci] = src[((long)bi * c + ci) * n + i];

        Tensor codes = new Tensor(new TensorShape(b, n), DType.I32);
        Fsq.Quantize(codes, zLast, _levels);
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
        if (_encoderConvs is null)
            throw new NotSupportedException("CosmosDvTokenizer.Encode requires the converted encoder.jit weights and the " +
                "resolved conv stage schedule (research-doc Open Q §10). Load them via LoadWeights first.");
        Tensor wavelet = HaarWavelet3D.Forward(video, _cfg.HaarLevels);
        Tensor features = RunConvStack(backend, _encoderConvs, wavelet);
        wavelet.Dispose();
        Tensor codes = QuantizeToCodes(features);
        features.Dispose();
        return codes;
    }

    /// <inheritdoc/>
    public Tensor Decode(IBackend backend, Tensor codes, int latentT, int latentH, int latentW)
    {
        ThrowIfDisposed();
        if (_decoderConvs is null)
            throw new NotSupportedException("CosmosDvTokenizer.Decode requires the converted decoder.jit weights and the " +
                "resolved conv stage schedule (research-doc Open Q §10). Load them via LoadWeights first.");
        Tensor continuous = CodesToContinuous(codes, latentT, latentH, latentW);
        Tensor features = RunConvStack(backend, _decoderConvs, continuous);
        continuous.Dispose();
        Tensor rgb = HaarWavelet3D.Inverse(features, _cfg.HaarLevels);
        features.Dispose();
        return rgb;
    }

    private static Tensor RunConvStack(IBackend backend, CausalConv3d[] stack, Tensor input)
    {
        Tensor current = input;
        bool ownsCurrent = false;
        foreach (CausalConv3d conv in stack)
        {
            Tensor next = conv.Forward(backend, current);
            if (ownsCurrent) current.Dispose();
            current = next;
            ownsCurrent = true;
        }
        if (!ownsCurrent)   // empty stack: return an owned copy so the caller can dispose uniformly
        {
            Tensor copy = new Tensor(input.Shape, DType.F32);
            backend.CopyTo(copy, input);
            return copy;
        }
        return current;
    }

    /// <summary>Wires the loaded encoder/decoder causal-conv stacks (built by <c>CosmosDvTokenizerConverter</c>
    /// from the converted JIT weights). Called at integration once the exact stage schedule is known.</summary>
    public void LoadWeights(CausalConv3d[] encoderConvs, CausalConv3d[] decoderConvs)
    {
        ThrowIfDisposed();
        _encoderConvs = encoderConvs ?? throw new ArgumentNullException(nameof(encoderConvs));
        _decoderConvs = decoderConvs ?? throw new ArgumentNullException(nameof(decoderConvs));
    }

    /// <summary>All conv weights for GPU preload (empty until <see cref="LoadWeights"/> runs).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_encoderConvs is not null)
            foreach (CausalConv3d conv in _encoderConvs)
                foreach (Tensor t in conv.EnumerateWeights()) yield return t;
        if (_decoderConvs is not null)
            foreach (CausalConv3d conv in _decoderConvs)
                foreach (Tensor t in conv.EnumerateWeights()) yield return t;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CosmosDvTokenizer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _encoderConvs = null;
        _decoderConvs = null;
    }
}
