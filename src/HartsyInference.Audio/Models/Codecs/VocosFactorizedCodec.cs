using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Amphion/MaskGCT's Vocos-encoder + factorized single-codebook VQ codec (real <c>RepCodec</c> /
/// <c>EnhancedCodec</c> — see <see cref="VocosFactorizedCodecConfig"/>). <c>Quantize</c> runs the encoder half
/// (optional 2x downsample → <see cref="VocosBackbone"/> → project up → project down to the codebook's own
/// dimension → nearest-neighbor lookup, reusing <see cref="VqOps"/>'s cosine-similarity search); <c>Decode</c>
/// runs the decoder half from codes alone (codebook embed → project up → <see cref="VocosBackbone"/> → project
/// down → optional 2x upsample), skipping the encoder entirely — exactly the real <c>quantize()</c>/<c>decode()</c>
/// split, not a reimplementation from the training-time <c>forward()</c>.</summary>
public sealed unsafe class VocosFactorizedCodec : IDisposable
{
    private readonly VocosFactorizedCodecConfig _cfg;
    private readonly VocosBackbone _encoderBackbone, _decoderBackbone;

    private Tensor? _downW, _downB, _upW, _upB;
    private Tensor? _encoderProjW, _encoderProjB, _decoderProjW, _decoderProjB;
    private Tensor? _inProjW, _inProjB, _outProjW, _outProjB;
    private Tensor? _codebook, _codebookNormalized;
    private bool _decoderLoaded;
    private int _disposed;

    public VocosFactorizedCodec(VocosFactorizedCodecConfig cfg)
    {
        _cfg = cfg;
        _encoderBackbone = new VocosBackbone(cfg.HiddenSize, cfg.VocosDim, cfg.VocosIntermediateDim, cfg.VocosNumLayers);
        _decoderBackbone = new VocosBackbone(cfg.HiddenSize, cfg.VocosDim, cfg.VocosIntermediateDim, cfg.VocosNumLayers);
    }

    /// <summary><paramref name="prefix"/> is this codec's own top-level module path (empty for a standalone
    /// checkpoint). Pass <paramref name="loadDecoder"/> false to skip the decoder half entirely when it is
    /// confirmed dead for the caller's inference path (IndexTTS-2.0's dependency on this codec never decodes).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "", bool loadDecoder = true)
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _encoderBackbone.LoadWeights(w, $"{p}encoder.0");
        _encoderProjW = WhisperOps.EnsureF32(w[$"{p}encoder.1.weight"]);
        _encoderProjB = WhisperOps.EnsureF32(w[$"{p}encoder.1.bias"]);

        _inProjW = WeightNormFusion.Compose(w, $"{p}quantizer.quantizers.0.in_project");
        _inProjB = WhisperOps.EnsureF32(w[$"{p}quantizer.quantizers.0.in_project.bias"]);
        _outProjW = WeightNormFusion.Compose(w, $"{p}quantizer.quantizers.0.out_project");
        _outProjB = WhisperOps.EnsureF32(w[$"{p}quantizer.quantizers.0.out_project.bias"]);
        _codebook = WhisperOps.EnsureF32(w[$"{p}quantizer.quantizers.0.codebook.weight"]);
        _codebookNormalized = VqOps.L2NormalizeRows(_codebook, _cfg.CodebookSize, _cfg.CodebookDim);

        if (_cfg.DownsampleScale > 1)
        {
            _downW = WhisperOps.EnsureF32(w[$"{p}down.weight"]);
            _downB = WhisperOps.EnsureF32(w[$"{p}down.bias"]);
        }

        _decoderLoaded = loadDecoder;
        if (!loadDecoder) return;

        _decoderBackbone.LoadWeights(w, $"{p}decoder.0");
        _decoderProjW = WhisperOps.EnsureF32(w[$"{p}decoder.1.weight"]);
        _decoderProjB = WhisperOps.EnsureF32(w[$"{p}decoder.1.bias"]);
        if (_cfg.DownsampleScale > 1)
        {
            _upW = WhisperOps.EnsureF32(w[$"{p}up.weight"]);
            _upB = WhisperOps.EnsureF32(w[$"{p}up.bias"]);
        }
    }

    /// <summary>Encodes a reference feature <c>[1, T, hiddenSize]</c> (channels-last; IndexTTS-2 feeds its
    /// normalized w2v-bert feature here) to <c>(codes[T'], continuousEmbedding[1, T', hiddenSize])</c> — the real
    /// <c>quantize()</c>'s two return values, the second already through <c>out_project</c> (not the raw
    /// pre-quantization latent). <see cref="Decode"/> is the separate codes-only reconstruction path
    /// (<c>vq2emb</c> + decoder), used when only discrete codes are available (e.g. the GPT's AR output).</summary>
    public (int[] Codes, Tensor Continuous) Quantize(IBackend backend, Tensor input, int t)
    {
        ThrowIfDisposed();
        int h = _cfg.HiddenSize;

        Tensor beforeEncoder = input;
        bool ownsBeforeEncoder = false;
        int tIn = t;
        if (_cfg.DownsampleScale > 1)
        {
            Tensor chFirst = new(new TensorShape(1, h, t), DType.F32);
            backend.Transpose2D(chFirst, input, t, h);
            int tDown = (t + 1) / 2; // kernel=3, stride=2, pad=1 -> floor((t+2*1-3)/2)+1 = floor((t-1)/2)+1 = ceil(t/2)
            Tensor down = new(new TensorShape(1, h, tDown), DType.F32);
            backend.Conv1d(down, chFirst, _downW!, _downB, stride: 2, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
            chFirst.Dispose();
            backend.GeluErf(down, down);
            Tensor chLast = new(new TensorShape(1, tDown, h), DType.F32);
            backend.Transpose2D(chLast, down, h, tDown);
            down.Dispose();
            beforeEncoder = chLast;
            ownsBeforeEncoder = true;
            tIn = tDown;
        }

        Tensor encoderChFirst = new(new TensorShape(1, h, tIn), DType.F32);
        backend.Transpose2D(encoderChFirst, beforeEncoder, tIn, h);
        if (ownsBeforeEncoder) beforeEncoder.Dispose();

        Tensor backboneOut = _encoderBackbone.Forward(backend, encoderChFirst, tIn);  // [1, tIn, vocosDim] channels-last
        encoderChFirst.Dispose();
        Tensor projected = WhisperOps.ProjectLinear(backend, backboneOut, _encoderProjW!, _encoderProjB, 1, tIn, _cfg.VocosDim, h);
        backboneOut.Dispose();

        Tensor projectedChFirst = new(new TensorShape(1, h, tIn), DType.F32);
        backend.Transpose2D(projectedChFirst, projected, tIn, h);
        projected.Dispose();

        Tensor latent = new(new TensorShape(1, _cfg.CodebookDim, tIn), DType.F32);
        backend.Conv1d(latent, projectedChFirst, _inProjW!, _inProjB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        projectedChFirst.Dispose();

        int[] codes = new int[tIn];
        fixed (int* codesPtr = codes)
            VqOps.NearestCodebookIndices((float*)latent.DataPointer, (float*)_codebookNormalized!.DataPointer, codesPtr,
                batch: 1, t: tIn, codebookDim: _cfg.CodebookDim, codebookSize: _cfg.CodebookSize);
        latent.Dispose();

        fixed (int* codesPtr = codes)
        {
            Tensor gathered = VqOps.GatherCodebookVectors(_codebook!, codesPtr, batch: 1, t: tIn, codebookDim: _cfg.CodebookDim);
            Tensor outProjected = new(new TensorShape(1, h, tIn), DType.F32);
            backend.Conv1d(outProjected, gathered, _outProjW!, _outProjB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            gathered.Dispose();

            Tensor continuous = new(new TensorShape(1, tIn, h), DType.F32);
            backend.Transpose2D(continuous, outProjected, h, tIn);
            outProjected.Dispose();
            return (codes, continuous);
        }
    }

    /// <summary>The real <c>quantizer.vq2emb(codes)</c>: codebook gather → <c>out_project</c>, nothing else (no
    /// backbone decoder, no resample, no normalization). Needs only the quantizer weights, so it works with
    /// <c>loadDecoder: false</c> — IndexTTS-2.0's codes→S2Mel handoff (<c>infer_v2.py</c>:
    /// <c>S_infer = vq2emb(codes) + gpt_layer(latent)</c>) uses exactly this. Returns channels-last
    /// <c>[1, T, hiddenSize]</c>, the layout <c>vq2emb(...).transpose(1, 2)</c> yields upstream.</summary>
    public Tensor VqToEmbedding(IBackend backend, ReadOnlySpan<int> codes)
    {
        ThrowIfDisposed();
        int h = _cfg.HiddenSize, t = codes.Length;
        Tensor chFirst = QuantizedChannelsFirst(backend, codes);
        Tensor chLast = new(new TensorShape(1, t, h), DType.F32);
        backend.Transpose2D(chLast, chFirst, h, t);
        chFirst.Dispose();
        return chLast;
    }

    /// <summary>Codebook gather + <c>out_project</c> into channels-first <c>[1, hiddenSize, T]</c> — the shared
    /// first half of <see cref="Decode"/> and <see cref="VqToEmbedding"/>.</summary>
    private Tensor QuantizedChannelsFirst(IBackend backend, ReadOnlySpan<int> codes)
    {
        int h = _cfg.HiddenSize, t = codes.Length;
        Tensor gathered;
        fixed (int* codesPtr = codes)
            gathered = VqOps.GatherCodebookVectors(_codebook!, codesPtr, batch: 1, t: t, codebookDim: _cfg.CodebookDim);
        Tensor quantizedChFirst = new(new TensorShape(1, h, t), DType.F32);
        backend.Conv1d(quantizedChFirst, gathered, _outProjW!, _outProjB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        gathered.Dispose();
        return quantizedChFirst;
    }

    /// <summary>Decodes discrete codes straight to the continuous embedding — the real <c>decode()</c>: codebook
    /// embed lookup → <c>out_project</c> (no encoder, no nearest-neighbor search) → <see cref="VocosBackbone"/>
    /// decoder → project down → optional 2x upsample. Requires <c>loadDecoder: true</c> at load time.</summary>
    public Tensor Decode(IBackend backend, ReadOnlySpan<int> codes)
    {
        ThrowIfDisposed();
        if (!_decoderLoaded) throw new InvalidOperationException("VocosFactorizedCodec decoder weights not loaded (loadDecoder: false).");
        int h = _cfg.HiddenSize, t = codes.Length;

        Tensor quantizedChFirst = QuantizedChannelsFirst(backend, codes);

        Tensor backboneOut = _decoderBackbone.Forward(backend, quantizedChFirst, t);  // [1, t, vocosDim] channels-last
        quantizedChFirst.Dispose();
        Tensor projected = WhisperOps.ProjectLinear(backend, backboneOut, _decoderProjW!, _decoderProjB, 1, t, _cfg.VocosDim, h);
        backboneOut.Dispose();

        if (_cfg.DownsampleScale <= 1) return projected;

        Tensor projectedChFirst = new(new TensorShape(1, h, t), DType.F32);
        backend.Transpose2D(projectedChFirst, projected, t, h);
        projected.Dispose();

        int tUp = t * 2;
        Tensor upsampled = new(new TensorShape(1, h, tUp), DType.F32);
        NearestUpsampleBy2(upsampled, projectedChFirst, h, t);
        projectedChFirst.Dispose();

        Tensor upProjected = new(new TensorShape(1, h, tUp), DType.F32);
        backend.Conv1d(upProjected, upsampled, _upW!, _upB, stride: 1, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        upsampled.Dispose();

        Tensor outChLast = new(new TensorShape(1, tUp, h), DType.F32);
        backend.Transpose2D(outChLast, upProjected, h, tUp);
        upProjected.Dispose();
        return outChLast;
    }

    /// <summary><c>F.interpolate(x, scale_factor=2, mode='nearest')</c> on a channels-first <c>[1, c, t]</c> tensor.</summary>
    private static void NearestUpsampleBy2(Tensor dst, Tensor src, int c, int t)
    {
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            float* srcRow = sp + (long)ch * t;
            float* dstRow = dp + (long)ch * t * 2;
            for (int i = 0; i < t; i++) { dstRow[2 * i] = srcRow[i]; dstRow[2 * i + 1] = srcRow[i]; }
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoderBackbone.EnumerateWeights()) yield return t;
        Tensor?[] rest =
        [
            _encoderProjW, _encoderProjB, _inProjW, _inProjB, _outProjW, _outProjB, _codebook, _codebookNormalized,
            _downW, _downB,
        ];
        foreach (Tensor? t in rest) if (t is not null) yield return t;
        if (!_decoderLoaded) yield break;
        foreach (Tensor t in _decoderBackbone.EnumerateWeights()) yield return t;
        Tensor?[] decoderRest = [_decoderProjW, _decoderProjB, _upW, _upB];
        foreach (Tensor? t in decoderRest) if (t is not null) yield return t;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(VocosFactorizedCodec));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _encoderBackbone.Dispose();
        _decoderBackbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
