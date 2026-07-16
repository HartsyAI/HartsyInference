using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Moshi/Kyutai <c>ConditionProvider</c> + <c>ConditionFuser</c> for the TTS model. Two LUT conditioners
/// (<c>control</c> "ok" and <c>cfg</c> the guidance scale) are summed into a per-step offset added to the stream
/// embedding; the <c>speaker_wavs</c> tensor conditioner projects a precomputed voice embedding and (with a
/// sinusoidal positional embedding) forms the cross-attention source. Each conditioner is an embedding/identity
/// followed by an <c>output_proj</c> Linear; masked positions fall back to a learnt padding (only relevant when a
/// condition is absent, which this path does not exercise). Validated in <c>KyutaiConditionerParityTests</c>.</summary>
public sealed unsafe class MoshiConditioner : IDisposable
{
    public const int Dim = 2048, CfgInner = 16, SpeakerDim = 512, MaxSpeakers = 5;
    private static readonly string[] CfgValues = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0"];

    private Tensor? _cfgEmbed, _cfgProj, _controlEmbed, _controlProj, _speakerProj, _speakerPadding;
    private int _disposed;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "condition_provider.conditioners")
    {
        _cfgEmbed = WhisperOps.EnsureF32(w[$"{prefix}.cfg.embed.weight"]);          // [8,16]
        _cfgProj = WhisperOps.EnsureF32(w[$"{prefix}.cfg.output_proj.weight"]);     // [2048,16]
        _controlEmbed = WhisperOps.EnsureF32(w[$"{prefix}.control.embed.weight"]);  // [2,2048]
        _controlProj = WhisperOps.EnsureF32(w[$"{prefix}.control.output_proj.weight"]); // [2048,2048]
        _speakerProj = WhisperOps.EnsureF32(w[$"{prefix}.speaker_wavs.output_proj.weight"]); // [2048,512]
        _speakerPadding = WhisperOps.EnsureF32(w[$"{prefix}.speaker_wavs.learnt_padding"]);  // [1,1,2048]
    }

    /// <summary>The cfg LUT bin for a guidance coefficient (e.g. 2.0 → 2); -1 if not a supported value.</summary>
    public static int CfgBin(float coef)
    {
        string key = coef.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture);
        return Array.IndexOf(CfgValues, key);
    }

    /// <summary>Per-step sum offset: <c>control_proj(control_embed[0]) + cfg_proj(cfg_embed[cfgBin])</c> → [1,1,2048].</summary>
    public Tensor ComputeSum(IBackend backend, int cfgBin)
    {
        Tensor ctrlRow = EmbedRow(_controlEmbed!, 0, Dim);
        Tensor ctrl = WhisperOps.ProjectLinear(backend, ctrlRow, _controlProj!, null, 1, 1, Dim, Dim);
        ctrlRow.Dispose();
        Tensor cfgRow = EmbedRow(_cfgEmbed!, cfgBin, CfgInner);
        Tensor cfg = WhisperOps.ProjectLinear(backend, cfgRow, _cfgProj!, null, 1, 1, CfgInner, Dim);
        cfgRow.Dispose();
        backend.Add(ctrl, ctrl, cfg);
        cfg.Dispose();
        return ctrl;   // [1,1,Dim]
    }

    /// <summary>Cross-attention source from a voice embedding <paramref name="voice"/> <c>[1,T,512]</c>. Mirrors
    /// moshi <c>make_condition_attributes</c> + <c>ConditionFuser.get_cross</c>: the voice fills the FIRST of
    /// <see cref="MaxSpeakers"/> (=5) speaker slots, the other four are masked and so contribute the learnt
    /// padding vector; the whole <c>maxSpeakers·T</c>-row tensor then gets a continuous sinusoidal position
    /// embedding (<c>cross_attention_pos_emb=True</c>). Result <c>[1, maxSpeakers·T, 2048]</c>. Those padding rows
    /// are NOT inert — cross-attention attends over all of them, so omitting them (previously the source was only
    /// the T real rows) shifts every cross-attention output and corrupts the generated audio codes.</summary>
    public Tensor ComputeCross(IBackend backend, Tensor voice, int maxSpeakers = MaxSpeakers)
    {
        int t = (int)voice.Shape[1], total = maxSpeakers * t;
        Tensor proj = WhisperOps.ProjectLinear(backend, voice, _speakerProj!, null, 1, t, SpeakerDim, Dim);
        Tensor cross = new(new TensorShape(1, total, Dim), DType.F32);
        float* cp = (float*)cross.DataPointer, pp = (float*)proj.DataPointer, pad = (float*)_speakerPadding!.DataPointer;
        Buffer.MemoryCopy(pp, cp, (long)t * Dim * 4, (long)t * Dim * 4);          // speaker 0 = real voice projection
        for (int r = t; r < total; r++)                                            // speakers 1..4 = learnt padding
            Buffer.MemoryCopy(pad, cp + (long)r * Dim, (long)Dim * 4, (long)Dim * 4);
        proj.Dispose();
        AddSinEmbedding(cross, total);                                             // continuous positions 0..total-1
        return cross;   // [1, total, Dim]
    }

    private static Tensor EmbedRow(Tensor table, int row, int width)
    {
        Tensor outT = new(new TensorShape(1, 1, width), DType.F32);
        Buffer.MemoryCopy((float*)table.DataPointer + (long)row * width, (void*)outT.DataPointer, width * 4, width * 4);
        return outT;
    }

    // Adds create_sin_embedding(positions, Dim): phase = pos / 10000^(i/(half-1)); cat([cos, sin]).
    private static void AddSinEmbedding(Tensor x, int t)
    {
        int half = Dim / 2;
        float* p = (float*)x.DataPointer;
        for (int s = 0; s < t; s++)
            for (int i = 0; i < half; i++)
            {
                double phase = s / Math.Pow(10000.0, (double)i / (half - 1));
                long b = (long)s * Dim;
                p[b + i] += (float)Math.Cos(phase);
                p[b + half + i] += (float)Math.Sin(phase);
            }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
