using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Tests.Dinov2;

public sealed class Dinov2HubKeysTests
{
    // A tiny DINOv2: width 8, two layers, a 2 x 2 patch grid of 14-pixel patches, one register.
    private static readonly Dinov2Preset Tiny = new()
    {
        Name = "tiny", HiddenSize = 8, NumLayers = 2, NumHeads = 2, IntermediateSize = 16, ImageSize = 28, PatchSize = 14, NumRegisterTokens = 1,
    };

    private static Tensor Numbered(float start, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        Span<float> data = t.AsSpan<float>();
        for (int i = 0; i < data.Length; i++)
            data[i] = start + i;
        return t;
    }

    private static Dictionary<string, Tensor> Hub(string prefix)
    {
        int d = Tiny.HiddenSize, ff = Tiny.IntermediateSize;
        Dictionary<string, Tensor> w = new()
        {
            [prefix + "patch_embed.proj.weight"] = Numbered(0, d, 3, 14, 14),
            [prefix + "patch_embed.proj.bias"] = Numbered(0, d),
            [prefix + "cls_token"] = Numbered(0, 1, 1, d),
            [prefix + "pos_embed"] = Numbered(0, 1, 1 + Tiny.NumPatches, d),
            [prefix + "register_tokens"] = Numbered(0, 1, 1, d),
            [prefix + "norm.weight"] = Numbered(0, d),
            [prefix + "norm.bias"] = Numbered(0, d),
        };
        for (int i = 0; i < Tiny.NumLayers; i++)
        {
            string b = $"{prefix}blocks.{i}.";
            foreach (string name in new[] { "norm1", "norm2" })
            {
                w[b + name + ".weight"] = Numbered(0, d);
                w[b + name + ".bias"] = Numbered(0, d);
            }

            w[b + "attn.qkv.weight"] = Numbered(1000 * (i + 1), 3 * d, d);
            w[b + "attn.qkv.bias"] = Numbered(500 * (i + 1), 3 * d);
            w[b + "attn.proj.weight"] = Numbered(0, d, d);
            w[b + "attn.proj.bias"] = Numbered(0, d);
            w[b + "ls1.gamma"] = Numbered(0, d);
            w[b + "ls2.gamma"] = Numbered(0, d);
            w[b + "mlp.fc1.weight"] = Numbered(0, ff, d);
            w[b + "mlp.fc1.bias"] = Numbered(0, ff);
            w[b + "mlp.fc2.weight"] = Numbered(0, d, ff);
            w[b + "mlp.fc2.bias"] = Numbered(0, d);
        }

        return w;
    }

    [Fact]
    public void Remap_RenamesToTheEncoderNamesAndSplitsTheFusedQkvIntoViews()
    {
        Dictionary<string, Tensor> hub = Hub("pretrained.");
        Dictionary<string, Tensor> mapped = Dinov2HubKeys.Remap(hub, "pretrained.", Tiny);

        Assert.Same(hub["pretrained.norm.weight"], mapped["layernorm.weight"]);
        Assert.Same(hub["pretrained.blocks.1.ls1.gamma"], mapped["encoder.layer.1.layer_scale1.lambda1"]);
        Assert.Same(hub["pretrained.register_tokens"], mapped["embeddings.register_tokens"]);

        // The fused [3d, d] projection splits in row order into query, key and value without a copy.
        int d = Tiny.HiddenSize;
        ReadOnlySpan<float> fused = hub["pretrained.blocks.1.attn.qkv.weight"].AsReadOnlySpan<float>();
        string[] names = ["query", "key", "value"];
        for (int part = 0; part < 3; part++)
        {
            Tensor view = mapped[$"encoder.layer.1.attention.attention.{names[part]}.weight"];
            Assert.Equal(new TensorShape(d, d), view.Shape);
            Assert.True(view.AsReadOnlySpan<float>().SequenceEqual(fused.Slice(part * d * d, d * d)));
            Tensor bias = mapped[$"encoder.layer.1.attention.attention.{names[part]}.bias"];
            Assert.True(bias.AsReadOnlySpan<float>().SequenceEqual(hub["pretrained.blocks.1.attn.qkv.bias"].AsReadOnlySpan<float>().Slice(part * d, d)));
        }
    }

    [Fact]
    public void Remap_RejectsAMissingMisshapenOrNonF32Weight()
    {
        Dictionary<string, Tensor> hub = Hub("");
        hub.Remove("blocks.1.ls2.gamma");
        Assert.Throws<InvalidOperationException>(() => Dinov2HubKeys.Remap(hub, "", Tiny));

        hub = Hub("");
        hub["pos_embed"] = Numbered(0, 1, 3, Tiny.HiddenSize);
        Assert.Throws<InvalidOperationException>(() => Dinov2HubKeys.Remap(hub, "", Tiny));

        hub = Hub("");
        hub["norm.bias"] = hub["norm.bias"].CastTo(DType.F16);
        Assert.NotNull(Dinov2HubKeys.Remap(hub, "", Tiny));
        Assert.Throws<InvalidOperationException>(() => Dinov2HubKeys.Remap(hub, "", Tiny, requireF32: true));
    }
}
