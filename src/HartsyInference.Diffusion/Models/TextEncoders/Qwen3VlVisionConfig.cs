using System.Text.Json;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for the Qwen3-VL vision tower (<c>Qwen3VLVisionModel</c>). Verified against
/// <c>Boogu-Image-0.1-Base/mllm/config.json</c> <c>vision_config</c>. A ViT that turns pixel patches into image tokens
/// merged into the Qwen3-VL language stream, with the Qwen3-VL <b>deepstack</b> feature taps and a learned position
/// embedding interpolated to the grid.</summary>
public sealed record Qwen3VlVisionConfig
{
    /// <summary>Number of vision transformer blocks (27).</summary>
    public int Depth { get; init; } = 27;

    /// <summary>Vision hidden size (1152).</summary>
    public int HiddenSize { get; init; } = 1152;

    /// <summary>Attention heads (16). head_dim = HiddenSize / NumHeads = 72.</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Per-head dim.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>MLP intermediate size (4304).</summary>
    public int IntermediateSize { get; init; } = 4304;

    /// <summary>Input channels (3, RGB).</summary>
    public int InChannels { get; init; } = 3;

    /// <summary>Spatial patch size (16).</summary>
    public int PatchSize { get; init; } = 16;

    /// <summary>Spatial merge size (2): a 2×2 block of patches merges into one image token.</summary>
    public int SpatialMergeSize { get; init; } = 2;

    /// <summary>Temporal patch size (2): a still image is duplicated to fill one temporal patch.</summary>
    public int TemporalPatchSize { get; init; } = 2;

    /// <summary>Output hidden size after the merger (4096 = LM hidden).</summary>
    public int OutHiddenSize { get; init; } = 4096;

    /// <summary>Number of learned position embeddings (2304); the per-side grid is sqrt of this.</summary>
    public int NumPositionEmbeddings { get; init; } = 2304;

    /// <summary>Vision blocks whose outputs feed the LM deepstack mergers ([8, 16, 24]).</summary>
    public int[] DeepstackVisualIndexes { get; init; } = [8, 16, 24];

    /// <summary>LayerNorm epsilon for the vision norms (1e-6).</summary>
    public float NormEps { get; init; } = 1e-6f;

    /// <summary>Whether the patch merger's GELU is the exact (erf) one, as transformers' Qwen3.5 merger (<c>nn.GELU()</c>)
    /// runs it; false keeps the tanh approximation the blocks use.</summary>
    public bool MergerExactGelu { get; init; }

    /// <summary>Per-side grid count for the learned position embedding = round(sqrt(NumPositionEmbeddings)) = 48.</summary>
    public int NumGridPerSide => (int)Math.Round(Math.Sqrt(NumPositionEmbeddings));

    /// <summary>Flattened patch-embed input dim = InChannels · TemporalPatchSize · PatchSize² = 1536.</summary>
    public int PatchEmbedInDim => InChannels * TemporalPatchSize * PatchSize * PatchSize;

    /// <summary>Boogu-Image / Qwen3-VL-8B vision preset.</summary>
    public static Qwen3VlVisionConfig Qwen3Vl8B => new();

    /// <summary>Reads a transformers <c>vision_config</c> object (Qwen3-VL or Qwen3.5); absent keys keep the 8B preset's.</summary>
    public static Qwen3VlVisionConfig FromJson(JsonElement vision)
    {
        Qwen3VlVisionConfig preset = Qwen3Vl8B;
        int[] deepstack = preset.DeepstackVisualIndexes;
        if (vision.TryGetProperty("deepstack_visual_indexes", out JsonElement taps) && taps.ValueKind == JsonValueKind.Array)
        {
            deepstack = new int[taps.GetArrayLength()];
            int at = 0;
            foreach (JsonElement tap in taps.EnumerateArray()) deepstack[at++] = tap.GetInt32();
        }
        return new Qwen3VlVisionConfig
        {
            Depth = Int(vision, "depth", preset.Depth),
            HiddenSize = Int(vision, "hidden_size", preset.HiddenSize),
            NumHeads = Int(vision, "num_heads", preset.NumHeads),
            IntermediateSize = Int(vision, "intermediate_size", preset.IntermediateSize),
            InChannels = Int(vision, "in_channels", preset.InChannels),
            PatchSize = Int(vision, "patch_size", preset.PatchSize),
            SpatialMergeSize = Int(vision, "spatial_merge_size", preset.SpatialMergeSize),
            TemporalPatchSize = Int(vision, "temporal_patch_size", preset.TemporalPatchSize),
            OutHiddenSize = Int(vision, "out_hidden_size", preset.OutHiddenSize),
            NumPositionEmbeddings = Int(vision, "num_position_embeddings", preset.NumPositionEmbeddings),
            DeepstackVisualIndexes = deepstack,
        };
    }

    /// <summary>Derives the tower's shape from its bare weights (<c>patch_embed.*</c>, <c>blocks.{i}.*</c>,
    /// <c>merger.*</c>), for a checkpoint shipped without its config. The head count is not in the shapes: pass it, or
    /// it is taken from the family's head widths (72, else 64). Deepstack taps are allowed only as the Qwen3-VL triple.</summary>
    public static Qwen3VlVisionConfig FromWeights(IReadOnlyDictionary<string, Tensor> bare, int? numHeads = null)
    {
        ArgumentNullException.ThrowIfNull(bare);
        Tensor patch = Require(bare, "patch_embed.proj.weight");
        int hidden = (int)patch.Shape[0];
        int depth = 0;
        while (bare.ContainsKey($"blocks.{depth}.attn.qkv.weight")) depth++;
        if (depth == 0)
            throw new InvalidDataException("The vision tower has no blocks.{i}.attn.qkv.weight.");
        int mergedIn = (int)Require(bare, "merger.linear_fc1.weight").Shape[1];
        int merge = (int)Math.Round(Math.Sqrt(mergedIn / (double)hidden));
        if (merge < 1 || merge * merge * hidden != mergedIn)
            throw new InvalidDataException($"The merger reads {mergedIn} features; that is no square merge of a {hidden}-wide token.");
        int patchSize = patch.Shape.Rank == 5 ? (int)patch.Shape[3] : 16;
        int temporal = patch.Shape.Rank == 5 ? (int)patch.Shape[2] : 2;
        int heads = numHeads ?? (hidden % 72 == 0 ? hidden / 72 : hidden % 64 == 0 ? hidden / 64
            : throw new InvalidDataException($"A {hidden}-wide vision tower needs its head count from a config."));
        int deepstackCount = 0;
        while (bare.ContainsKey($"deepstack_merger_list.{deepstackCount}.linear_fc1.weight")) deepstackCount++;
        int[] deepstack = deepstackCount switch
        {
            0 => [],
            3 => Qwen3Vl8B.DeepstackVisualIndexes,
            _ => throw new InvalidDataException($"The vision tower carries {deepstackCount} deepstack mergers; their taps need a config."),
        };
        return new Qwen3VlVisionConfig
        {
            Depth = depth,
            HiddenSize = hidden,
            NumHeads = heads,
            IntermediateSize = (int)Require(bare, "blocks.0.mlp.linear_fc1.weight").Shape[0],
            PatchSize = patchSize,
            TemporalPatchSize = temporal,
            SpatialMergeSize = merge,
            OutHiddenSize = (int)Require(bare, "merger.linear_fc2.weight").Shape[0],
            NumPositionEmbeddings = (int)Require(bare, "pos_embed.weight").Shape[0],
            DeepstackVisualIndexes = deepstack,
        };
    }

    private static int Int(JsonElement e, string key, int fallback) =>
        e.TryGetProperty(key, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> bare, string key) =>
        bare.TryGetValue(key, out Tensor? t) ? t : throw new InvalidDataException($"The vision tower is missing '{key}'.");
}
