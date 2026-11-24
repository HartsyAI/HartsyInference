using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Tests.LongCat;

/// <summary>A small Whisper encoder with deterministic pseudo-random weights, for tests that check wiring rather than
/// values a real checkpoint would give.</summary>
internal static class TinyWhisperEncoder
{
    public static WhisperConfig Config(int layers, int melBins = 4, int hidden = 8) => new()
    {
        NumMelBins = melBins,
        EncoderLayers = layers,
        DecoderLayers = 1,
        HiddenSize = hidden,
        NumHeads = 2,
        IntermediateSize = hidden * 2,
    };

    public static (WhisperEncoder Encoder, List<Tensor> Weights) Build(WhisperConfig cfg, int seed = 7)
    {
        Random rng = new(seed);
        List<Tensor> owned = [];
        Dictionary<string, Tensor> weights = new();

        Tensor Make(string name, params int[] dims)
        {
            Tensor t = new(new TensorShape(Array.ConvertAll(dims, d => (long)d)), DType.F32);
            Span<float> span = t.AsSpan<float>();
            for (int i = 0; i < span.Length; i++) span[i] = (float)(rng.NextDouble() - 0.5) * 0.4f;
            weights[name] = t;
            owned.Add(t);
            return t;
        }

        void Norm(string name, int d)
        {
            Tensor w = Make(name + ".weight", d);
            w.AsSpan<float>().Fill(1f);
            Make(name + ".bias", d).AsSpan<float>().Clear();
        }

        int dModel = cfg.HiddenSize;
        const string P = "model.encoder";
        Make($"{P}.conv1.weight", dModel, cfg.NumMelBins, 3);
        Make($"{P}.conv1.bias", dModel);
        Make($"{P}.conv2.weight", dModel, dModel, 3);
        Make($"{P}.conv2.bias", dModel);
        Make($"{P}.embed_positions.weight", cfg.MaxAudioPositions, dModel);
        for (int i = 0; i < cfg.EncoderLayers; i++)
        {
            string l = $"{P}.layers.{i}";
            Norm($"{l}.self_attn_layer_norm", dModel);
            foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "out_proj" })
            {
                Make($"{l}.self_attn.{proj}.weight", dModel, dModel);
                if (proj != "k_proj") Make($"{l}.self_attn.{proj}.bias", dModel);
            }
            Make($"{l}.fc1.weight", cfg.IntermediateSize, dModel);
            Make($"{l}.fc1.bias", cfg.IntermediateSize);
            Make($"{l}.fc2.weight", dModel, cfg.IntermediateSize);
            Make($"{l}.fc2.bias", dModel);
            Norm($"{l}.final_layer_norm", dModel);
        }
        Norm($"{P}.layer_norm", dModel);

        WhisperEncoder encoder = new(cfg);
        encoder.LoadWeights(weights);
        return (encoder, owned);
    }
}
