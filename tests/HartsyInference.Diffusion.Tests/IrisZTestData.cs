using System.Collections;
using System.Diagnostics.CodeAnalysis;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A tiny random Iris-Z-shaped checkpoint and FLUX.2-shaped VAE for CPU tests: 8 token channels (a 2-channel
/// latent under 2×2 patches), hidden 32.</summary>
internal static class IrisZTestData
{
    public const int Hidden = 32;
    public const int Mlp = 96;
    public const int Channels = 8;
    public const int Context = 12;
    public const int LatentChannels = 2;
    public const int VaeWidth = 8;

    public static Flux2Config Config(int modalities) => new()
    {
        HiddenSize = Hidden,
        NumHeads = 2,
        HeadDim = 16,
        Depth = 1,
        DepthSingleBlocks = 2,
        InChannels = Channels,
        OutChannels = Channels,
        ContextInDim = Context,
        AxesDim = [4, 4, 4, 4],
        VaeLatentChannels = LatentChannels,
        ModalityCount = modalities,
        TextEncoderType = Flux2TextEncoderType.Qwen,
    };

    /// <summary>Four 8-wide stages (8× downscale) with one layer each, the FLUX.2 VAE's shape at toy widths.</summary>
    public static VaeConfig Vae => VaeConfig.Flux2 with
    {
        LatentChannels = LatentChannels,
        BlockOutChannels = [VaeWidth, VaeWidth, VaeWidth, VaeWidth],
        LayersPerBlock = 1,
        NormNumGroups = 4,
    };

    /// <summary>A random tiny checkpoint in BFL key names, run through the converter the recipe uses.</summary>
    public static Dictionary<string, Tensor> Converted(bool zeroModality)
    {
        Random rng = new(7);
        Dictionary<string, Tensor> bfl = new(StringComparer.Ordinal)
        {
            ["img_in.weight"] = Random(rng, Hidden, Channels),
            ["txt_in.weight"] = Random(rng, Hidden, Context),
            ["time_in.in_layer.weight"] = Random(rng, Hidden, 256),
            ["time_in.out_layer.weight"] = Random(rng, Hidden, Hidden),
            ["double_stream_modulation_img.lin.weight"] = Random(rng, 6 * Hidden, Hidden),
            ["double_stream_modulation_txt.lin.weight"] = Random(rng, 6 * Hidden, Hidden),
            ["single_stream_modulation.lin.weight"] = Random(rng, 3 * Hidden, Hidden),
            ["final_layer.adaLN_modulation.1.weight"] = Random(rng, 2 * Hidden, Hidden),
            ["final_layer.linear.weight"] = Random(rng, Channels, Hidden),
            ["modality_in.weight"] = zeroModality ? Zeros(4, Hidden) : Random(rng, 4, Hidden),
        };
        foreach (string stream in new[] { "img", "txt" })
        {
            bfl[$"double_blocks.0.{stream}_attn.qkv.weight"] = Random(rng, 3 * Hidden, Hidden);
            bfl[$"double_blocks.0.{stream}_attn.proj.weight"] = Random(rng, Hidden, Hidden);
            bfl[$"double_blocks.0.{stream}_attn.norm.query_norm.scale"] = Scale(rng, 16);
            bfl[$"double_blocks.0.{stream}_attn.norm.key_norm.scale"] = Scale(rng, 16);
            bfl[$"double_blocks.0.{stream}_mlp.0.weight"] = Random(rng, 2 * Mlp, Hidden);
            bfl[$"double_blocks.0.{stream}_mlp.2.weight"] = Random(rng, Hidden, Mlp);
        }
        for (int block = 0; block < 2; block++)
        {
            bfl[$"single_blocks.{block}.linear1.weight"] = Random(rng, 3 * Hidden + 2 * Mlp, Hidden);
            bfl[$"single_blocks.{block}.linear2.weight"] = Random(rng, Hidden, Hidden + Mlp);
            bfl[$"single_blocks.{block}.norm.query_norm.scale"] = Scale(rng, 16);
            bfl[$"single_blocks.{block}.norm.key_norm.scale"] = Scale(rng, 16);
        }
        return new Flux2CheckpointConverter(Hidden, Mlp).ConvertTransformer(bfl);
    }

    public static Tensor Random(Random rng, params long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        return tensor;
    }

    public static Tensor Filled(float value, params long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }

    private static Tensor Zeros(params long[] dims) => Filled(0f, dims);

    private static Tensor Scale(Random rng, int length)
    {
        Tensor tensor = new(new TensorShape(length), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (float)(0.5 + rng.NextDouble());
        return tensor;
    }

    /// <summary>Weights for <see cref="Vae"/>, made on first read of each key: every stage is <see cref="VaeWidth"/> wide,
    /// so a key's name fixes its shape. Norms are identity; convolutions and projections are small random values.</summary>
    public sealed class VaeWeights : IReadOnlyDictionary<string, Tensor>
    {
        private readonly Dictionary<string, Tensor> _made = new(StringComparer.Ordinal);
        private readonly Random _rng = new(11);

        public Tensor this[string key]
        {
            get
            {
                if (!_made.TryGetValue(key, out Tensor? tensor))
                {
                    tensor = Make(key);
                    _made[key] = tensor;
                }
                return tensor;
            }
        }

        public IEnumerable<string> Keys => _made.Keys;

        public IEnumerable<Tensor> Values => _made.Values;

        public int Count => _made.Count;

        // The loaders probe only an optional shortcut this way, which equal-width stages never have.
        public bool ContainsKey(string key) => _made.ContainsKey(key);

        public bool TryGetValue(string key, [MaybeNullWhen(false)] out Tensor value) => _made.TryGetValue(key, out value);

        public IEnumerator<KeyValuePair<string, Tensor>> GetEnumerator() => _made.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        private Tensor Make(string key)
        {
            bool bias = key.EndsWith(".bias", StringComparison.Ordinal);
            string module = key[..key.LastIndexOf('.')];
            (long outChannels, long[] weightShape) = module switch
            {
                "encoder.conv_in" => (VaeWidth, new long[] { VaeWidth, 3, 3, 3 }),
                "encoder.conv_out" => (2 * LatentChannels, new long[] { 2 * LatentChannels, VaeWidth, 3, 3 }),
                "quant_conv" => (2 * LatentChannels, new long[] { 2 * LatentChannels, 2 * LatentChannels, 1, 1 }),
                "post_quant_conv" => (LatentChannels, new long[] { LatentChannels, LatentChannels, 1, 1 }),
                "decoder.conv_in" => (VaeWidth, new long[] { VaeWidth, LatentChannels, 3, 3 }),
                "decoder.conv_out" => (3, new long[] { 3, VaeWidth, 3, 3 }),
                _ when module.Contains(".to_", StringComparison.Ordinal) => (VaeWidth, new long[] { VaeWidth, VaeWidth }),
                _ when module.Contains("norm", StringComparison.Ordinal) => (VaeWidth, new long[] { VaeWidth }),
                _ => (VaeWidth, new long[] { VaeWidth, VaeWidth, 3, 3 }),
            };
            if (bias)
                return Filled(0f, outChannels);
            return module.Contains("norm", StringComparison.Ordinal) ? Filled(1f, weightShape) : Random(_rng, weightShape);
        }
    }
}
