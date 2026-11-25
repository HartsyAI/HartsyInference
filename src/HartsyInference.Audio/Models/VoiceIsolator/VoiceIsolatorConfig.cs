using System.Globalization;
using System.Text.Json;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>Architecture of an Inworld Voice Isolator checkpoint, read from the <c>config</c> entry of its safetensors metadata.</summary>
public sealed record VoiceIsolatorConfig
{
    /// <summary>Token width of every block.</summary>
    public int Width { get; init; } = 512;

    /// <summary>One entry per block: <c>transformer</c> (dual-path attention) or <c>mamba</c> (dual-path state space).</summary>
    public required IReadOnlyList<string> BlockTypes { get; init; }

    /// <summary>Attention heads of the time and band transformers.</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>STFT window and transform size.</summary>
    public int NFft { get; init; } = 2048;

    /// <summary>STFT hop in samples (one frame).</summary>
    public int HopLength { get; init; } = 480;

    /// <summary>Sample rate the network runs at.</summary>
    public int SampleRate { get; init; } = 48_000;

    /// <summary>Band plan: (band width in Hz, upper limit in Hz) pairs.</summary>
    public required IReadOnlyList<(int WidthHz, int UpToHz)> BandPlan { get; init; }

    /// <summary>Bound of the residual correction relative to each bin's magnitude.</summary>
    public float CorrectionBound { get; init; } = 0.5f;

    /// <summary>Offline render window in seconds.</summary>
    public double ChunkSeconds { get; init; } = 12.0;

    /// <summary>Offline crossfade between windows in seconds.</summary>
    public double OverlapSeconds { get; init; } = 2.0;

    /// <summary>Whether the time attention removes each output's component along its own value.</summary>
    public bool ExclusiveSelfAttention { get; init; } = true;

    /// <summary>State size of every Mamba layer.</summary>
    public int MambaState { get; init; } = 16;

    /// <summary>Depthwise convolution taps of every Mamba layer.</summary>
    public int MambaConv { get; init; } = 4;

    /// <summary>Inner width factor of every Mamba layer.</summary>
    public int MambaExpand { get; init; } = 2;

    /// <summary>True for the streaming checkpoint: causal attention and time scan.</summary>
    public bool Causal { get; init; }

    /// <summary>Look-ahead in frames the checkpoint runs at by default.</summary>
    public int LookaheadFrames { get; init; }

    /// <summary>Look-ahead positions the checkpoint was trained at; empty for the offline checkpoint.</summary>
    public IReadOnlyList<int> LookaheadChoices { get; init; } = [];

    /// <summary>Frames each causal query may look back over (itself included); 0 means unbounded.</summary>
    public int AttentionWindowFrames { get; init; }

    /// <summary>Per-head width of the attention layers.</summary>
    public int HeadDim => Width / NumHeads;

    /// <summary>Inner width of every Mamba layer.</summary>
    public int MambaInner => MambaExpand * Width;

    /// <summary>Rank of the Mamba step-size projection.</summary>
    public int MambaDtRank => (Width + 15) / 16;

    /// <summary>Parses the checkpoint's <c>config</c> JSON. Fields this port does not implement are refused rather than guessed.</summary>
    public static VoiceIsolatorConfig FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrEmpty(json);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("width", out _))
        {
            throw new InvalidDataException("The voice isolator config has no 'width'; refusing to guess the architecture.");
        }
        RejectUnsupported(root);
        List<(int, int)> plan = [];
        foreach (JsonElement pair in root.GetProperty("band_plan").EnumerateArray())
        {
            plan.Add((pair[0].GetInt32(), pair[1].GetInt32()));
        }
        List<string> blocks = [];
        foreach (JsonElement kind in root.GetProperty("block_types").EnumerateArray())
        {
            string name = kind.GetString() ?? string.Empty;
            if (name is not ("transformer" or "mamba"))
            {
                throw new InvalidDataException($"Unknown voice isolator block type '{name}'.");
            }
            blocks.Add(name);
        }
        List<int> choices = [];
        if (root.TryGetProperty("lookahead_frames_choices", out JsonElement choiceArray))
        {
            foreach (JsonElement choice in choiceArray.EnumerateArray())
            {
                choices.Add(choice.GetInt32());
            }
        }
        VoiceIsolatorConfig config = new()
        {
            Width = root.GetProperty("width").GetInt32(),
            BlockTypes = blocks,
            NumHeads = Int(root, "num_heads", 8),
            NFft = Int(root, "n_fft", 2048),
            HopLength = Int(root, "hop_length", 480),
            SampleRate = Int(root, "sample_rate", 48_000),
            BandPlan = plan,
            CorrectionBound = (float)Double(root, "correction_bound", 0.5),
            ChunkSeconds = Double(root, "chunk_seconds", 12.0),
            OverlapSeconds = Double(root, "overlap_seconds", 2.0),
            ExclusiveSelfAttention = Bool(root, "exclusive_self_attention", true),
            MambaState = Int(root, "mamba_state", 16),
            MambaConv = Int(root, "mamba_conv", 4),
            MambaExpand = Int(root, "mamba_expand", 2),
            Causal = Bool(root, "causal", false),
            LookaheadFrames = Int(root, "lookahead_frames", 0),
            LookaheadChoices = choices,
            AttentionWindowFrames = Int(root, "attention_window_frames", 0),
        };
        config.Validate();
        return config;
    }

    private void Validate()
    {
        if (Width < 1 || NumHeads < 1 || Width % (2 * NumHeads) != 0)
        {
            throw new InvalidDataException("width must be a positive multiple of twice num_heads.");
        }
        if (NFft < 4 || NFft % 2 != 0 || HopLength < 1 || HopLength > NFft / 2)
        {
            throw new InvalidDataException("n_fft must be even and hop_length within (0, n_fft / 2].");
        }
        if (BlockTypes.Count == 0 || BandPlan.Count == 0 || MambaState < 1 || MambaConv < 1 || MambaExpand < 1)
        {
            throw new InvalidDataException("The voice isolator config needs blocks, a band plan and positive Mamba dimensions.");
        }
        if (CorrectionBound <= 0f || CorrectionBound >= 1f)
        {
            throw new InvalidDataException("correction_bound must lie in (0, 1).");
        }
        if (AttentionWindowFrames < 0 || (AttentionWindowFrames > 0 && !Causal))
        {
            throw new InvalidDataException("attention_window_frames needs a causal model and cannot be negative.");
        }
        if (LookaheadFrames < 0 || (LookaheadChoices.Count > 0 && !LookaheadChoices.Contains(LookaheadFrames)))
        {
            throw new InvalidDataException("lookahead_frames must be one of lookahead_frames_choices.");
        }
        if (LookaheadFrames > 0 && !Causal)
        {
            throw new InvalidDataException("A look-ahead needs a causal model.");
        }
    }

    private static void RejectUnsupported(JsonElement root)
    {
        if (Bool(root, "dual_mode", false) || Bool(root, "mode_film", false) || Int(root, "offline_passes", 1) != 1)
        {
            throw new InvalidDataException("Dual-mode, mode-FiLM and multi-pass voice isolator checkpoints are not supported.");
        }
        if (root.TryGetProperty("exclusive_axes", out JsonElement axes) && axes.GetString() != "time")
        {
            throw new InvalidDataException($"exclusive_axes '{axes.GetString()}' is not supported; only 'time' is.");
        }
        foreach ((string key, string expected) in new[] { ("representation", "bands"), ("decoder_type", "complex_mask_residual") })
        {
            if (root.TryGetProperty(key, out JsonElement value) && value.GetString() != expected)
            {
                throw new InvalidDataException($"{key} '{value.GetString()}' is not supported; only '{expected}' is.");
            }
        }
    }

    private static int Int(JsonElement root, string name, int fallback) =>
        root.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.Number ? e.GetInt32() : fallback;

    private static double Double(JsonElement root, string name, double fallback) =>
        root.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.Number
            ? double.Parse(e.GetRawText(), CultureInfo.InvariantCulture) : fallback;

    private static bool Bool(JsonElement root, string name, bool fallback) =>
        root.TryGetProperty(name, out JsonElement e) && e.ValueKind is JsonValueKind.True or JsonValueKind.False ? e.GetBoolean() : fallback;
}
