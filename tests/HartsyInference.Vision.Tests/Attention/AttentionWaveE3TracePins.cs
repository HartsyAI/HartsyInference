using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Vision.Tests.Attention;

public sealed partial class AttentionWaveE3TraceTests
{
    /// <summary>The ops must equal the pre-migration pin, and no attention call may hold more live tensors than it did before.</summary>
    private static void Pin(string site, AttentionTrace actual)
    {
        Assert.Equal(Pins[site], actual.Ops);
        int[] before = Live[site];
        Assert.Equal(before.Length, actual.LiveAtAttention.Length);
        for (int i = 0; i < before.Length; i++)
            Assert.True(actual.LiveAtAttention[i] <= before[i], $"{site}: attention call {i} holds {actual.LiveAtAttention[i]} live tensors, it held {before[i]} before");
    }

    /// <summary>Live tensors the stub has seen, at each attention call of the pre-migration code.</summary>
    private static readonly Dictionary<string, int[]> Live = new()
    {
        ["bert_masked"] = [5],
        ["bert_null"] = [5],
        ["clipseg_decoder"] = [5],
        ["dinov2_b1"] = [6],
        ["dinov2_b2"] = [6],
        ["dinovit"] = [6],
        ["groundingdino_mha"] = [9, 10],
        ["hiera_global"] = [15],
        ["hiera_pooled"] = [25],
        ["hiera_windowed"] = [15],
        ["rtdetr"] = [7],
        ["rtdetr_masked"] = [7],
        ["sam2_cross"] = [13],
        ["sam2_self"] = [13],
        ["siglip"] = [6],
        ["siglip_pool_b1"] = [5],
        ["siglip_pool_b2"] = [5],
        ["swin_shift0"] = [7],
        ["swin_shift2"] = [7],
    };

    private static readonly Dictionary<string, string[]> Pins = new()
    {
        // BERT: the host head reshape loops (three in, one out) are device Permute0213 ops now; the old ops had none.
        ["bert_masked"] =
        [
            "LayerNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 7, 8]:F32 q=[1, 2, 7, 8] k=[1, 2, 7, 8] mask=[1, 2, 7, 7]",
            "Permute0213 2,7,8",
            "Linear w16",
            "LayerNorm w16",
            "Linear w24",
            "Linear w16",
            "LayerNorm w16",
        ],
        ["bert_null"] =
        [
            "LayerNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 7, 8]:F32 q=[1, 2, 7, 8] k=[1, 2, 7, 8] mask=-",
            "Permute0213 2,7,8",
            "Linear w16",
            "LayerNorm w16",
            "Linear w24",
            "Linear w16",
            "LayerNorm w16",
        ],
        ["clipseg_decoder"] =
        [
            "Linear w64",
            "Linear w64",
            "Linear w64",
            "Permute0213 5,4,16",
            "Permute0213 5,4,16",
            "Permute0213 5,4,16",
            "Attn Dense scale=0.25 f16=False out=[1, 4, 5, 16]:F32 q=[1, 4, 5, 16] k=[1, 4, 5, 16] mask=-",
            "Permute0213 4,5,16",
            "Linear w64",
            "Add",
            "LayerNorm w64",
            "Linear w2048",
        ],
        ["dinov2_b1"] =
        [
            "LayerNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 7, 8]:F32 q=[1, 2, 7, 8] k=[1, 2, 7, 8] mask=-",
            "Permute0213 2,7,8",
            "Linear w16",
            "Add",
            "LayerNorm w16",
            "Linear w24",
            "Linear w16",
            "Add",
        ],
        ["dinov2_b2"] =
        [
            "LayerNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Attn Dense scale=0.353553 f16=False out=[2, 2, 7, 8]:F32 q=[2, 2, 7, 8] k=[2, 2, 7, 8] mask=-",
            "Permute0213 2,7,8",
            "Linear w16",
            "Add",
            "LayerNorm w16",
            "Linear w24",
            "Linear w16",
            "Add",
        ],
        ["dinovit"] =
        [
            "LayerNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Attn Dense scale=0.353553 f16=False out=[2, 2, 7, 8]:F32 q=[2, 2, 7, 8] k=[2, 2, 7, 8] mask=-",
            "Permute0213 2,7,8",
            "Linear w16",
            "Add",
            "LayerNorm w16",
            "Linear w24",
            "Linear w16",
            "Add",
        ],
        ["groundingdino_mha"] =
        [
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 7,2,8",
            "Permute0213 5,2,8",
            "Permute0213 5,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 7, 8]:F32 q=[1, 2, 7, 8] k=[1, 2, 5, 8] mask=[1, 2, 7, 5]",
            "Permute0213 2,7,8",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 5,2,8",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 5, 8]:F32 q=[1, 2, 5, 8] k=[1, 2, 7, 8] mask=-",
            "Permute0213 2,5,8",
            "Linear w16",
        ],
        ["hiera_global"] =
        [
            "LayerNorm w8",
            "Linear w24",
            "Split 3 dim=2",
            "Permute0213 16,2,4",
            "Permute0213 16,2,4",
            "Permute0213 16,2,4",
            "Attn Dense scale=0.5 f16=False out=[1, 2, 16, 4]:F32 q=[1, 2, 16, 4] k=[1, 2, 16, 4] mask=-",
            "Permute0213 2,16,4",
            "Linear w8",
            "Add",
            "LayerNorm w8",
            "Linear w32",
            "Gelu",
            "Linear w8",
            "Add",
        ],
        ["hiera_pooled"] =
        [
            "LayerNorm w8",
            "Linear w16",
            "Transpose2D 36,16",
            "Transpose2D 16,9",
            "Linear w48",
            "Split 3 dim=2",
            "Transpose2D 16,16",
            "Transpose2D 16,4",
            "Permute0213 4,2,8",
            "Permute0213 16,2,8",
            "Permute0213 16,2,8",
            "Attn Dense scale=0.353553 f16=False out=[4, 2, 4, 8]:F32 q=[4, 2, 4, 8] k=[4, 2, 16, 8] mask=-",
            "Permute0213 2,4,8",
            "Linear w16",
            "Add",
            "LayerNorm w16",
            "Linear w64",
            "Gelu",
            "Linear w16",
            "Add",
        ],
        ["hiera_windowed"] =
        [
            "LayerNorm w8",
            "Linear w24",
            "Split 3 dim=2",
            "Permute0213 4,2,4",
            "Permute0213 4,2,4",
            "Permute0213 4,2,4",
            "Attn Dense scale=0.5 f16=False out=[9, 2, 4, 4]:F32 q=[9, 2, 4, 4] k=[9, 2, 4, 4] mask=-",
            "Permute0213 2,4,4",
            "Linear w8",
            "Add",
            "LayerNorm w8",
            "Linear w32",
            "Gelu",
            "Linear w8",
            "Add",
        ],
        ["rtdetr"] =
        [
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 6, 8]:F32 q=[1, 2, 6, 8] k=[1, 2, 6, 8] mask=-",
            "Permute0213 2,6,8",
        ],
        ["rtdetr_masked"] =
        [
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "Permute0213 6,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 6, 8]:F32 q=[1, 2, 6, 8] k=[1, 2, 6, 8] mask=[1, 1, 6, 6]",
            "Permute0213 2,6,8",
        ],
        ["sam2_cross"] =
        [
            "Linear w8",
            "Linear w8",
            "Linear w8",
            "Permute0213 5,2,4",
            "Permute0213 9,2,4",
            "Permute0213 9,2,4",
            "Attn Dense scale=0.5 f16=False out=[2, 2, 5, 4]:F32 q=[2, 2, 5, 4] k=[2, 2, 9, 4] mask=-",
            "Permute0213 2,5,4",
            "Linear w16",
        ],
        ["sam2_self"] =
        [
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 5,2,8",
            "Permute0213 5,2,8",
            "Permute0213 5,2,8",
            "Attn Dense scale=0.353553 f16=False out=[2, 2, 5, 8]:F32 q=[2, 2, 5, 8] k=[2, 2, 5, 8] mask=-",
            "Permute0213 2,5,8",
            "Linear w16",
        ],
        ["siglip"] =
        [
            "LayerNorm w16",
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Permute0213 7,2,8",
            "Attn Dense scale=0.353553 f16=False out=[2, 2, 7, 8]:F32 q=[2, 2, 7, 8] k=[2, 2, 7, 8] mask=-",
            "Permute0213 2,7,8",
            "Linear w16",
            "Add",
            "LayerNorm w16",
            "Linear w24",
            "Gelu",
            "Linear w16",
            "Add",
        ],
        ["siglip_pool_b1"] =
        [
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 1,2,8",
            "Permute0213 4,2,8",
            "Permute0213 4,2,8",
            "Attn Dense scale=0.353553 f16=False out=[1, 2, 1, 8]:F32 q=[1, 2, 1, 8] k=[1, 2, 4, 8] mask=-",
            "Permute0213 2,1,8",
            "Linear w16",
        ],
        ["siglip_pool_b2"] =
        [
            "Linear w16",
            "Linear w16",
            "Linear w16",
            "Permute0213 1,2,8",
            "Permute0213 4,2,8",
            "Permute0213 4,2,8",
            "Attn Dense scale=0.353553 f16=False out=[2, 2, 1, 8]:F32 q=[2, 2, 1, 8] k=[2, 2, 4, 8] mask=-",
            "Permute0213 2,1,8",
            "Linear w16",
        ],
        ["swin_shift0"] =
        [
            "Permute0213 16,2,4",
            "Permute0213 16,2,4",
            "Permute0213 16,2,4",
            "Attn Dense scale=0.5 f16=False out=[4, 2, 16, 4]:F32 q=[4, 2, 16, 4] k=[4, 2, 16, 4] mask=[4, 2, 16, 16]",
            "Permute0213 2,16,4",
        ],
        ["swin_shift2"] =
        [
            "Permute0213 16,2,4",
            "Permute0213 16,2,4",
            "Permute0213 16,2,4",
            "Attn Dense scale=0.5 f16=False out=[4, 2, 16, 4]:F32 q=[4, 2, 16, 4] k=[4, 2, 16, 4] mask=[4, 2, 16, 16]",
            "Permute0213 2,16,4",
        ],
    };
}
