using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>AudioGen ships PyTorch <c>.bin</c> pickles (not safetensors) and a separately-fetched
/// <c>google/t5-base</c>. These cover (a) the source-format-agnostic key mapping the AudioGen path relies on,
/// and (b) the <c>.bin</c> reader end-to-end when a generated checkpoint is provided.</summary>
public sealed class AudioGenLoaderTests
{
    private static Tensor T(params int[] shape)
    {
        long[] dims = Array.ConvertAll(shape, x => (long)x);
        return new Tensor(new TensorShape(dims), DType.F32);
    }

    [Fact]
    public void MapWeights_SplitsCombinedCheckpointAcrossDecoderTextEncoderAndCodec()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["decoder.model.decoder.layers.0.self_attn.q_proj.weight"] = T(4, 4),
            ["decoder.model.decoder.embed_positions.weights"] = T(8, 4),   // sinusoidal buffer → dropped
            ["decoder.lm_heads.0.weight"] = T(4, 4),
            ["enc_to_dec_proj.weight"] = T(4, 4),
            ["text_encoder.shared.weight"] = T(6, 4),
            ["audio_encoder.encoder.layers.0.conv.weight_g"] = T(4, 1, 1),
            ["audio_encoder.encoder.layers.0.conv.weight_v"] = T(4, 2, 3),
        };
        try
        {
            Dictionary<string, Tensor> dec = MusicGenCheckpointConverter.MapDecoderWeights(raw);
            Assert.True(dec.ContainsKey("model.decoder.layers.0.self_attn.q_proj.weight")); // decoder. stripped
            Assert.True(dec.ContainsKey("lm_heads.0.weight"));
            Assert.True(dec.ContainsKey("enc_to_dec_proj.weight"));
            Assert.DoesNotContain(dec.Keys, k => k.Contains("embed_positions"));
            Assert.DoesNotContain(dec.Keys, k => k.StartsWith("text_encoder") || k.StartsWith("audio_encoder"));

            Dictionary<string, Tensor> t5 = MusicGenCheckpointConverter.MapTextEncoderWeights(raw);
            Assert.True(t5.ContainsKey("shared.weight")); // text_encoder. stripped
            Assert.Single(t5);

            Dictionary<string, Tensor> codec = MusicGenCheckpointConverter.MapEnCodecWeights(raw);
            // audio_encoder.* subset, remapped HF → Meta naming (encoder.layers.N → encoder.model.N.conv.conv).
            Assert.Contains(codec.Keys, k => k.Contains("encoder.model.0.conv.conv"));
        }
        finally { foreach (Tensor t in raw.Values) t.Dispose(); }
    }

    [Fact]
    public void LoadDecoderAny_ReadsPytorchBin()
    {
        // Set HARTSYINFERENCE_AUDIOGEN_BIN to a torch.save'd checkpoint (see tests/python-reference) to run.
        string? binPath = Environment.GetEnvironmentVariable("HARTSYINFERENCE_AUDIOGEN_BIN");
        if (binPath is null || !File.Exists(binPath)) { Assert.True(true, "set HARTSYINFERENCE_AUDIOGEN_BIN to run"); return; }

        (Dictionary<string, Tensor> weights, IDisposable loader) = MusicGenCheckpointConverter.LoadDecoderAny(binPath, castToF32: true);
        using (loader)
        {
            Assert.True(weights.ContainsKey("model.decoder.layers.0.self_attn.q_proj.weight"));
            Assert.True(weights.ContainsKey("lm_heads.0.weight"));
            Assert.DoesNotContain(weights.Keys, k => k.Contains("embed_positions"));
            Assert.DoesNotContain(weights.Keys, k => k.StartsWith("text_encoder") || k.StartsWith("audio_encoder"));
        }

        (Dictionary<string, Tensor> t5, IDisposable t5Loader) = MusicGenCheckpointConverter.LoadTextEncoderAny(binPath, castToF32: true);
        using (t5Loader)
            Assert.True(t5.ContainsKey("shared.weight"));
    }
}
