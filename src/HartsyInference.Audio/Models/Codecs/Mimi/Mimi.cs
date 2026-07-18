using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi neural audio codec (Kyutai), HF <c>transformers</c> MimiModel layout. DECODE path
/// (codes -> 24 kHz PCM), used by Sesame CSM and the Kyutai delayed-streams models:
/// <list type="number">
///   <item><see cref="MimiSplitRvq"/> EMA split semantic+acoustic RVQ: codes -> latent <c>[B,512,T]</c></item>
///   <item><c>upsample</c>: depthwise ConvTranspose1d (k4, stride 2, groups 512), causal trim -> 12.5 to 25 Hz</item>
///   <item><see cref="MimiTransformer"/> decoder transformer (8 layers, LayerScale, sliding-window RoPE)</item>
///   <item><see cref="MimiSeanetDecoder"/> SEANet (ratios [8,6,5,4]) -> 25 Hz to 24 kHz</item>
/// </list>
/// Decode order matches <c>MimiModel._decode_frame</c>: quantizer.decode -> upsample -> decoder_transformer ->
/// decoder. Verified against the real kyutai/mimi weights.</summary>
public sealed unsafe class Mimi
{
    public MimiConfig Config { get; }
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;
    public int NCodebooks => Config.TotalCodebooks;

    private const int UpStride = 2, UpKernel = 4;
    private const int DownStride = 2, DownKernel = 4;

    private readonly MimiSplitRvq _rvq;
    private readonly MimiTransformer _decoderTransformer;
    private readonly MimiSeanetDecoder _decoder;
    private Tensor? _upsampleW;     // [latentDim, 1, 4] depthwise convtr

    // Encode path (audio -> codes), lazily loaded on first Encode: SEANet encoder + encoder transformer +
    // strided downsample conv (the mirror of decode's upsample). Kept optional so decode-only consumers
    // (CSM) don't pay for it.
    private readonly MimiTransformer _encoderTransformer;
    private readonly MimiSeanetEncoder _encoder;
    private Tensor? _downsampleW;   // [latentDim, latentDim, 4] strided conv
    private Tensor? _downsampleB;
    private bool _encodeLoaded;

    public Mimi(MimiConfig cfg)
    {
        Config = cfg;
        _rvq = new MimiSplitRvq("quantizer", cfg.NumSemanticCodebooks, cfg.TotalCodebooks, cfg.CodebookDim, cfg.LatentDim);
        _decoderTransformer = new MimiTransformer(cfg, "decoder_transformer");
        _decoder = new MimiSeanetDecoder("decoder", cfg.LatentDim);
        _encoderTransformer = new MimiTransformer(cfg, "encoder_transformer");
        _encoder = new MimiSeanetEncoder("encoder", cfg.LatentDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        w = NormalizeKeys(w);
        if (MimiDsmWeights.IsDsm(w))
            w = MimiDsmWeights.Adapt(w);   // moshi-native DSM checkpoint → HF layout for the decode-path loaders
        _rvq.LoadWeights(w);
        _decoderTransformer.LoadWeights(w);
        _decoder.LoadWeights(w);
        _upsampleW = Whisper.WhisperOps.EnsureF32(w["upsample.conv.weight"]);

        // Load the encode path too when the checkpoint carries the encoder (STT needs audio->codes; the
        // -trfs / full Mimi checkpoint ships encoder.* / encoder_transformer.* / downsample.conv.*).
        if (w.ContainsKey("encoder.layers.0.conv.weight") && w.ContainsKey("downsample.conv.weight"))
        {
            _encoder.LoadWeights(w);
            _encoderTransformer.LoadWeights(w);
            _downsampleW = Whisper.WhisperOps.EnsureF32(w["downsample.conv.weight"]);
            _downsampleB = w.TryGetValue("downsample.conv.bias", out Tensor? db) ? Whisper.WhisperOps.EnsureF32(db) : null;
            _encodeLoaded = true;
        }
    }

    /// <summary>Decodes <c>[B, K, T]</c> Int32 codes (K = total codebooks) to PCM <c>[B, 1, T*frame_size]</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor codes, int batch, int tFrames)
    {
        if (_upsampleW is null) throw new InvalidOperationException("Mimi weights not loaded.");

        Tensor emb = _rvq.Decode(backend, codes, batch, tFrames);                 // [B, 512, T]
        int latent = Config.LatentDim;

        int tUp = tFrames * UpStride;
        Tensor up = new(new TensorShape(batch, latent, tUp), DType.F32);
        backend.ConvTranspose1d(up, emb, _upsampleW!, null, stride: UpStride, padLeft: 0, padRight: UpKernel - UpStride, dilation: 1, groups: latent);
        emb.Dispose();

        Tensor cl = new(new TensorShape(batch, tUp, latent), DType.F32);
        backend.Transpose2D(cl, up, latent, tUp);
        up.Dispose();
        Tensor ctx = _decoderTransformer.Forward(backend, cl, batch, tUp);
        cl.Dispose();
        Tensor ctxCf = new(new TensorShape(batch, latent, tUp), DType.F32);
        backend.Transpose2D(ctxCf, ctx, tUp, latent);
        ctx.Dispose();

        Tensor pcm = _decoder.Forward(backend, ctxCf, batch, tUp);
        ctxCf.Dispose();
        return pcm;
    }

    /// <summary>Encodes PCM <c>[B, 1, tPcm]</c> to codes <c>[B, K, T]</c> (Int32, K = total codebooks). Mirror of
    /// <see cref="Decode"/> / HF <c>MimiModel._encode_frame</c>: SEANet encoder → encoder transformer → strided
    /// downsample conv → split-RVQ nearest-neighbour encode. <paramref name="tPcm"/> should be a multiple of the
    /// SEANet downsample product (960); the caller pads to a frame boundary.</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        if (!_encodeLoaded) throw new InvalidOperationException("Mimi encode path not loaded (checkpoint lacks encoder.*/downsample.*).");
        int latent = Config.LatentDim;

        Tensor emb = _encoder.Forward(backend, pcm, batch, tPcm);   // [B, latent, tEnc] (25 Hz)
        int tEnc = (int)emb.Shape[2];

        // encoder transformer runs channels-last over the SEANet output.
        Tensor cl = new(new TensorShape(batch, tEnc, latent), DType.F32);
        backend.Transpose2D(cl, emb, latent, tEnc);
        emb.Dispose();
        Tensor ctx = _encoderTransformer.Forward(backend, cl, batch, tEnc);
        cl.Dispose();
        Tensor ctxCf = new(new TensorShape(batch, latent, tEnc), DType.F32);
        backend.Transpose2D(ctxCf, ctx, tEnc, latent);
        ctx.Dispose();

        // Strided causal downsample (25 Hz → 12.5 Hz): left-pad (kernel-stride).
        int tDown = tEnc / DownStride;
        Tensor down = new(new TensorShape(batch, latent, tDown), DType.F32);
        backend.Conv1d(down, ctxCf, _downsampleW!, _downsampleB, stride: DownStride, padLeft: DownKernel - DownStride, padRight: 0, dilation: 1, groups: 1);
        ctxCf.Dispose();

        Tensor codes = _rvq.Encode(backend, down, batch, tDown);
        down.Dispose();
        return codes;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _rvq.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoderTransformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        if (_upsampleW is not null) yield return _upsampleW;
        if (_encodeLoaded)
        {
            foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
            foreach (Tensor t in _encoderTransformer.EnumerateWeights()) yield return t;
            if (_downsampleW is not null) yield return _downsampleW;
            if (_downsampleB is not null) yield return _downsampleB;
        }
    }

    private static IReadOnlyDictionary<string, Tensor> NormalizeKeys(IReadOnlyDictionary<string, Tensor> w)
    {
        bool needs = false;
        foreach (string k in w.Keys)
            if (k.Contains(".parametrizations.weight.", StringComparison.Ordinal)) { needs = true; break; }
        if (!needs) return w;
        Dictionary<string, Tensor> d = new(w.Count);
        foreach (KeyValuePair<string, Tensor> kv in w) d[CodecKeyUtils.NormalizeWeightNormKey(kv.Key)] = kv.Value;
        return d;
    }
}
