using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>CosyVoice 2 flow-matching stage (speech tokens → mel). Mirrors
/// <c>cosyvoice/flow/flow.py:CausalMaskedDiffWithXvec</c>: embeds the LM speech tokens, runs them
/// through the (chunk-aware causal) encoder, time-upsamples 25 Hz → 50 Hz, projects to the 80-bin mel
/// conditioning <c>μ</c>, projects the CAM++ speaker vector to mel dim, then solves the OT-CFM ODE with
/// classifier-free guidance to produce the target mel.
///
/// <para>The token-conditioning path is now the real <see cref="UpsampleConformerEncoder"/>: embedded
/// tokens → conformer stack → 2× ConvTranspose1d upsample (25 Hz → 50 Hz) → conformer stack →
/// <c>encoder_proj</c> → mel conditioning <c>μ</c>. Everything downstream of <c>μ</c> (the CFM solve, CFG,
/// speaker conditioning) is exact.</para></summary>
public sealed unsafe class CosyVoiceFlow : IDisposable
{
    private readonly CosyVoiceConfig _cfg;
    private readonly CausalConditionalDecoder _estimator;
    private readonly ConditionalCfm _cfm;
    private readonly UpsampleConformerEncoder _encoder;
    private int _disposed;

    private Tensor? _inputEmbedding;     // [speechVocab, inputSize]
    private Tensor? _encoderProjW, _encoderProjB;   // encoderOutputSize → melBins
    private Tensor? _spkAffineW, _spkAffineB;        // 192 → melBins

    public CosyVoiceFlow(CosyVoiceConfig cfg)
    {
        _cfg = cfg;
        _estimator = new CausalConditionalDecoder(cfg.Flow);
        _cfm = new ConditionalCfm(_estimator, cfg.Flow.MelBins);
        _encoder = new UpsampleConformerEncoder(cfg.Flow.EncoderOutputSize, cfg.Flow.EncoderNumHeads,
            cfg.Flow.EncoderNumPreBlocks, cfg.Flow.EncoderNumPostBlocks);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inputEmbedding = WhisperOps.EnsureF32(w[$"{p}input_embedding.weight"]);
        _encoder.LoadWeights(w, $"{p}encoder");
        _encoderProjW = WhisperOps.EnsureF32(w[$"{p}encoder_proj.weight"]);
        _encoderProjB = WhisperOps.EnsureF32(w[$"{p}encoder_proj.bias"]);
        _spkAffineW = WhisperOps.EnsureF32(w[$"{p}spk_embed_affine_layer.weight"]);
        _spkAffineB = WhisperOps.EnsureF32(w[$"{p}spk_embed_affine_layer.bias"]);
        _estimator.LoadWeights(w, $"{p}decoder.estimator");
    }

    /// <summary>Generates the target mel <c>[1, melBins, T_mel]</c> for a speech-token stream.
    /// <paramref name="promptSpeechTokens"/> + <paramref name="promptMel"/> are the reference clip's
    /// tokens + mel (empty for preset-voice modes); <paramref name="speakerEmbed"/> is the CAM++
    /// 192-d vector.</summary>
    public Tensor Inference(IBackend backend,
        ReadOnlySpan<int> speechTokens,
        ReadOnlySpan<int> promptSpeechTokens,
        Tensor? promptMel,
        Tensor speakerEmbed,
        int seed = 0)
    {
        if (_inputEmbedding is null) throw new InvalidOperationException("CosyVoiceFlow weights not loaded.");
        int inputSize = _cfg.Flow.InputSize;
        int mel = _cfg.Flow.MelBins;

        // Concatenate prompt + target speech tokens and embed.
        int nTok = promptSpeechTokens.Length + speechTokens.Length;
        Tensor tokEmb = new(new TensorShape(1, nTok, inputSize), DType.F32);
        int row = 0;
        for (int i = 0; i < promptSpeechTokens.Length; i++) WriteEmbRow(tokEmb, row++, promptSpeechTokens[i], inputSize);
        for (int i = 0; i < speechTokens.Length; i++) WriteEmbRow(tokEmb, row++, speechTokens[i], inputSize);

        // UpsampleConformerEncoder: conformer stack → 2× time upsample (25 Hz token → 50 Hz) → conformer stack.
        Tensor up = _encoder.Forward(backend, tokEmb, inputSize);
        tokEmb.Dispose();
        int tMel = (int)up.Shape[1];
        int encOut = (int)up.Shape[2];

        // encoder_proj → μ [1, T_mel, mel] then transpose to channels-first [1, mel, T_mel].
        Tensor muSeq = WhisperOps.ProjectLinear(backend, up, _encoderProjW!, _encoderProjB, 1, tMel, encOut, mel);
        up.Dispose();
        Tensor mu = new(new TensorShape(1, mel, tMel), DType.F32);
        backend.Transpose2D(mu, muSeq, tMel, mel);
        muSeq.Dispose();

        // Speaker embedding → mel dim [1, mel] (kept as [1, mel, 1] for broadcast).
        Tensor spk = WhisperOps.ProjectLinear(backend, ReshapeRow(speakerEmbed, _cfg.Flow.SpeakerEmbedDim), _spkAffineW!, _spkAffineB, 1, 1, _cfg.Flow.SpeakerEmbedDim, mel);
        Tensor spkChan = spk.Reshape(new TensorShape(1, mel, 1));

        // Reference-mel conditioning: place the prompt mel in the prefix, zeros elsewhere.
        Tensor cond = new(new TensorShape(1, mel, tMel), DType.F32);
        if (promptMel is not null) WritePromptCond(cond, promptMel, mel, tMel);

        Tensor outMel = _cfm.Solve(backend, mu, spkChan, cond, _cfg.Flow.NumEulerSteps, _cfg.Flow.CfgRate, seed);
        mu.Dispose();
        spk.Dispose();
        cond.Dispose();
        return outMel;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_inputEmbedding, _encoderProjW, _encoderProjB, _spkAffineW, _spkAffineB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _estimator.EnumerateWeights()) yield return t;
    }

    private void WriteEmbRow(Tensor dst, int row, int token, int dim)
    {
        int vocab = (int)_inputEmbedding!.Shape[0];
        if ((uint)token >= (uint)vocab) throw new ArgumentException($"speech token {token} out of range [0, {vocab}).");
        float* sp = (float*)_inputEmbedding.DataPointer + (long)token * dim;
        float* dp = (float*)dst.DataPointer + (long)row * dim;
        Buffer.MemoryCopy(sp, dp, dim * 4, dim * 4);
    }

    private static Tensor ReshapeRow(Tensor v, int dim)
    {
        if (v.ElementCount != dim) throw new ArgumentException($"speaker embed must have {dim} elements, got {v.ElementCount}.");
        return v.Reshape(new TensorShape(1, 1, dim));
    }

    private static void WritePromptCond(Tensor cond, Tensor promptMel, int mel, int tMel)
    {
        int tp = Math.Min((int)promptMel.Shape[2], tMel);
        float* pp = (float*)promptMel.DataPointer;
        float* cp = (float*)cond.DataPointer;
        for (int c = 0; c < mel; c++)
            for (int j = 0; j < tp; j++)
                cp[(long)c * tMel + j] = pp[(long)c * (int)promptMel.Shape[2] + j];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _encoder.Dispose();
        GC.SuppressFinalize(this);
    }
}
