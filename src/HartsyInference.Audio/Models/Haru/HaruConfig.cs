using System.Text.Json;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.LLM.Ssm;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Haru's <c>config.json</c>: the frame geometry and the special tokens, the Qwen3.5 text trunk (<c>lm</c>), the audio codec
/// and the MeanFlow head, checked against each other so a mismatched checkpoint fails at load rather than mid-conversation.</summary>
internal sealed record HaruConfig
{
    public required int SampleRate { get; init; }

    /// <summary>Samples in one frame of either stream: 3200 at 24 kHz, 133 ms.</summary>
    public required int FrameSamples { get; init; }

    public required int LatentDim { get; init; }

    /// <summary>The text token of a frame in which the model says nothing.</summary>
    public required int PadTokenId { get; init; }

    /// <summary>The text token of the frame before the model starts to speak (end of padding).</summary>
    public required int EopTokenId { get; init; }

    public int? EosTokenId { get; init; }

    public required Qwen35HfConfig Lm { get; init; }

    public required VibeVoiceTokenizerConfig Codec { get; init; }

    public required HaruHeadConfig Head { get; init; }

    /// <summary>Frames per second: 7.5.</summary>
    public double FrameRate => (double)SampleRate / FrameSamples;

    internal static HaruConfig FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        string? modelType = root.TryGetProperty("model_type", out JsonElement type) ? type.GetString() : null;
        if (modelType != "haru")
        {
            throw new InvalidDataException($"Not a Haru checkpoint (model_type '{modelType}').");
        }
        JsonElement lm = root.GetProperty("lm");
        if (lm.TryGetProperty("attn_output_gate", out JsonElement gate) && gate.ValueKind == JsonValueKind.False)
        {
            throw new InvalidDataException("The engine's Qwen3.5 trunk always gates the attention output; attn_output_gate=false is not supported.");
        }
        HaruConfig config = new()
        {
            SampleRate = root.GetProperty("sample_rate").GetInt32(),
            FrameSamples = root.GetProperty("frame_samples").GetInt32(),
            LatentDim = root.GetProperty("latent_dim").GetInt32(),
            PadTokenId = root.GetProperty("pad_token_id").GetInt32(),
            EopTokenId = root.GetProperty("eop_token_id").GetInt32(),
            EosTokenId = lm.TryGetProperty("eos_token_id", out JsonElement eos) && eos.ValueKind == JsonValueKind.Number ? eos.GetInt32() : null,
            Lm = Qwen35HfConfig.FromTextConfig(lm),
            Codec = HaruCodec.ConfigFrom(root.GetProperty("codec")),
            Head = HaruHeadConfig.FromJson(root.GetProperty("head")),
        };
        config.Validate();
        return config;
    }

    private void Validate()
    {
        if (FrameSamples != HaruCodec.FrameSamples)
        {
            throw new InvalidDataException($"frame_samples is {FrameSamples}; the codec's frame is {HaruCodec.FrameSamples}.");
        }
        if (Codec.VaeDim != LatentDim || Head.LatentDim != LatentDim)
        {
            throw new InvalidDataException($"latent_dim {LatentDim} disagrees with the codec ({Codec.VaeDim}) or the head ({Head.LatentDim}).");
        }
        if (Head.HiddenSize != Lm.HiddenSize)
        {
            throw new InvalidDataException($"The head width {Head.HiddenSize} is not the language model's hidden size {Lm.HiddenSize}.");
        }
        foreach (int id in new[] { PadTokenId, EopTokenId })
        {
            if (id < 0 || id >= Lm.VocabSize)
            {
                throw new InvalidDataException($"Special token id {id} is outside the {Lm.VocabSize}-token vocabulary.");
            }
        }
        if (PadTokenId == EopTokenId)
        {
            throw new InvalidDataException("The PAD and EOP tokens must differ.");
        }
        if (EosTokenId is int eos && (eos < 0 || eos >= Lm.VocabSize || eos == PadTokenId || eos == EopTokenId))
        {
            throw new InvalidDataException($"eos_token_id {eos} must be inside the {Lm.VocabSize}-token vocabulary and differ from PAD and EOP.");
        }
        if (!Lm.TieWordEmbeddings)
        {
            throw new InvalidDataException("Haru reads its text logits from the tied embedding table; tie_word_embeddings must be true.");
        }
    }
}
