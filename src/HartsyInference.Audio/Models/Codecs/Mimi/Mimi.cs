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

    private readonly MimiSplitRvq _rvq;
    private readonly MimiTransformer _decoderTransformer;
    private readonly MimiSeanetDecoder _decoder;
    private Tensor? _upsampleW;     // [latentDim, 1, 4] depthwise convtr

    public Mimi(MimiConfig cfg)
    {
        Config = cfg;
        _rvq = new MimiSplitRvq("quantizer", cfg.NumSemanticCodebooks, cfg.TotalCodebooks, cfg.CodebookDim, cfg.LatentDim);
        _decoderTransformer = new MimiTransformer(cfg, "decoder_transformer");
        _decoder = new MimiSeanetDecoder("decoder", cfg.LatentDim);
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

    /// <summary>Encode (audio -> codes) is not yet ported to the HF layout (decode is the path CSM/Kyutai use).</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm)
        => throw new NotSupportedException("Mimi encode (audio->codes) is not yet wired for the HF layout; decode is verified.");

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _rvq.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoderTransformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        if (_upsampleW is not null) yield return _upsampleW;
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
