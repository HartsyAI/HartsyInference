namespace HartsyInference.Engine.Requests;

/// <summary>Native text-to-music request. Flat fields cover the common and ACE-Step/HeartMuLa knobs; the LM-planner
/// block drives ACE-Step's 5 Hz planner; and the continuation/repaint/cover inputs select the editing mode over a
/// supplied source clip (null on all three = plain text-to-music).</summary>
public sealed record MusicRequest
{
    /// <summary>The music description prompt.</summary>
    public required string Prompt { get; init; }

    /// <summary>Genre/style tag (ACE-Step style prompt, YuE genre); empty for MusicGen/AudioGen.</summary>
    public string Genre { get; init; } = "";

    /// <summary>Target duration in seconds.</summary>
    public double Duration { get; init; } = 10;

    /// <summary>Sampling seed.</summary>
    public int Seed { get; init; }

    /// <summary>Noise-schedule shift (ACE-Step); null uses the variant default.</summary>
    public double? Shift { get; init; }

    /// <summary>Diffusion step count (ACE-Step); null uses the variant default.</summary>
    public int? InferSteps { get; init; }

    /// <summary>Classifier-free guidance scale; null uses the pipeline default (1.0 disables the 2× CFG pass).</summary>
    public double? CfgScale { get; init; }

    /// <summary>Sampling temperature (HeartMuLa); null uses the pipeline default.</summary>
    public double? Temperature { get; init; }

    /// <summary>Top-K sampling cutoff (HeartMuLa); null uses the pipeline default.</summary>
    public int? TopK { get; init; }

    /// <summary>Nucleus sampling threshold (YuE); null uses the pipeline default.</summary>
    public double? TopP { get; init; }

    /// <summary>Repetition penalty (YuE); null uses the pipeline default. Below 1 it would reward repetition —
    /// YuE loops into instrumental phrases without it, so the loader floors it at 1.</summary>
    public double? RepetitionPenalty { get; init; }

    /// <summary>BPM meta (ACE-Step prompt template); null = "N/A".</summary>
    public int? Bpm { get; init; }

    /// <summary>Key/scale meta (ACE-Step prompt template); empty = "N/A".</summary>
    public string KeyScale { get; init; } = "";

    /// <summary>Time-signature meta (ACE-Step prompt template); empty = "N/A".</summary>
    public string TimeSignature { get; init; } = "";

    /// <summary>Lyric vocal language (ACE-Step lyric template); empty = "en".</summary>
    public string VocalLanguage { get; init; } = "";

    /// <summary>ODE (default) or SDE diffusion solver (ACE-Step base/sft).</summary>
    public string InferMethod { get; init; } = "";

    /// <summary>Use ADG guidance instead of the default APG (ACE-Step base/sft CFG).</summary>
    public bool UseAdg { get; init; }

    /// <summary>CFG active-interval start over t (ACE-Step base/sft).</summary>
    public double CfgIntervalStart { get; init; }

    /// <summary>CFG active-interval end over t (ACE-Step base/sft).</summary>
    public double CfgIntervalEnd { get; init; } = 1.0;

    /// <summary>ACE-Step 5 Hz LM planner selection: "", "none", "0.6b", or "4b".</summary>
    public string LmModel { get; init; } = "";

    /// <summary>LM planner CoT thinking phase on/off.</summary>
    public bool Thinking { get; init; } = true;

    /// <summary>LM planner sampling temperature.</summary>
    public double LmTemperature { get; init; } = 0.85;

    /// <summary>LM planner CFG scale.</summary>
    public double LmCfgScale { get; init; } = 2.0;

    /// <summary>LM planner top-K cutoff.</summary>
    public int LmTopK { get; init; }

    /// <summary>LM planner top-P cutoff.</summary>
    public double LmTopP { get; init; } = 0.9;

    /// <summary>LM planner negative prompt.</summary>
    public string LmNegativePrompt { get; init; } = "";

    /// <summary>LoRAs to merge into the model's weights before loading, the same stack the image and video requests
    /// take. Changing the set reloads the model, so it is part of the runner cache key. Only MiniMax Music 3
    /// consumes these today.</summary>
    public LoraStack? Loras { get; init; }

    /// <summary>Continuation seed audio: extend this clip forward. Null for none.</summary>
    public AudioClip? Continuation { get; init; }

    /// <summary>Repaint (inpaint) source audio: regenerate the [<see cref="RepaintStart"/>, <see cref="RepaintEnd"/>]
    /// span while preserving the rest. Null for none.</summary>
    public AudioClip? Repaint { get; init; }

    /// <summary>Repaint span start in seconds (used when <see cref="Repaint"/> is set).</summary>
    public double RepaintStart { get; init; }

    /// <summary>Repaint span end in seconds (used when <see cref="Repaint"/> is set).</summary>
    public double RepaintEnd { get; init; }

    /// <summary>Cover source audio: re-render this clip under the new prompt/style. Null for none.</summary>
    public AudioClip? Cover { get; init; }

    /// <summary>Cover conditioning strength (0 keeps the source, 1 fully re-renders).</summary>
    public double CoverStrength { get; init; } = 0.5;
}
