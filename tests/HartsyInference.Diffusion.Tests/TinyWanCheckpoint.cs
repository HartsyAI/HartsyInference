using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A diffusers-named Wan checkpoint with random weights, built in memory: 2 layers, 2 heads of 128, inner 256. The
/// default is the plain latent model (4 channels, 1x2x2 patches, a <c>proj_out</c> head); the options give the patch embedding
/// another input width and patch, and leave the output projection out for a model that brings its own head.</summary>
internal sealed class TinyWanCheckpoint : IDisposable
{
    public const int Heads = 2;
    public const int HeadDim = 128;
    public const int Inner = Heads * HeadDim;
    public const int Layers = 2;
    public const int Ffn = 64;
    public const int Channels = 4;
    public const int TextDim = 32;
    public const int FreqDim = 32;

    private readonly Random _rng = new(7);

    public TinyWanCheckpoint(int inChannels = Channels, int patchHeight = 2, int patchWidth = 2, bool withProjection = true)
    {
        Weights["patch_embedding.weight"] = Random(Inner, inChannels, 1, patchHeight, patchWidth);
        Weights["patch_embedding.bias"] = Random(Inner);
        if (withProjection)
        {
            Weights["proj_out.weight"] = Random(Channels * 4, Inner);
            Weights["proj_out.bias"] = Random(Channels * 4);
            Weights["scale_shift_table"] = Random(1, 2, Inner);
        }
        Weights["condition_embedder.time_embedder.linear_1.weight"] = Random(Inner, FreqDim);
        Weights["condition_embedder.time_embedder.linear_1.bias"] = Random(Inner);
        Weights["condition_embedder.time_embedder.linear_2.weight"] = Random(Inner, Inner);
        Weights["condition_embedder.time_embedder.linear_2.bias"] = Random(Inner);
        Weights["condition_embedder.time_proj.weight"] = Random(6 * Inner, Inner);
        Weights["condition_embedder.time_proj.bias"] = Random(6 * Inner);
        Weights["condition_embedder.text_embedder.linear_1.weight"] = Random(Inner, TextDim);
        Weights["condition_embedder.text_embedder.linear_1.bias"] = Random(Inner);
        Weights["condition_embedder.text_embedder.linear_2.weight"] = Random(Inner, Inner);
        Weights["condition_embedder.text_embedder.linear_2.bias"] = Random(Inner);
        for (int i = 0; i < Layers; i++)
        {
            string block = $"blocks.{i}";
            Weights[$"{block}.scale_shift_table"] = Random(1, 6, Inner);
            foreach (string attention in new[] { "attn1", "attn2" })
            {
                foreach (string projection in new[] { "to_q", "to_k", "to_v", "to_out.0" })
                {
                    Weights[$"{block}.{attention}.{projection}.weight"] = Random(Inner, Inner);
                    Weights[$"{block}.{attention}.{projection}.bias"] = Random(Inner);
                }
                Weights[$"{block}.{attention}.norm_q.weight"] = Ones(Inner);
                Weights[$"{block}.{attention}.norm_k.weight"] = Ones(Inner);
            }
            Weights[$"{block}.norm2.weight"] = Ones(Inner);
            Weights[$"{block}.norm2.bias"] = Random(Inner);
            Weights[$"{block}.ffn.net.0.proj.weight"] = Random(Ffn, Inner);
            Weights[$"{block}.ffn.net.0.proj.bias"] = Random(Ffn);
            Weights[$"{block}.ffn.net.2.weight"] = Random(Inner, Ffn);
            Weights[$"{block}.ffn.net.2.bias"] = Random(Inner);
        }
    }

    public Dictionary<string, Tensor> Weights { get; } = [];

    public void Dispose()
    {
        foreach (Tensor tensor in Weights.Values) tensor.Dispose();
    }

    private Tensor Random(params long[] dims) => IrisZTestData.Random(_rng, dims);

    private static Tensor Ones(params long[] dims) => IrisZTestData.Filled(1f, dims);
}
