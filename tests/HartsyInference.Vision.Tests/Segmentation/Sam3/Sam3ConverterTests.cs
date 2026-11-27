using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation.Sam3;
using Xunit;

namespace HartsyInference.Vision.Tests.Segmentation.Sam3;

public sealed class Sam3ConverterTests : IDisposable
{
    private const string VisionTrunk = "VisionTrunk";
    private const string VisionNeck = "VisionNeck";
    private const string TextEncoder = "TextEncoder";
    private const string FusionEncoder = "FusionEncoder";
    private const string Decoder = "Decoder";
    private const string GeometryEncoder = "GeometryEncoder";
    private const string Scoring = "Scoring";
    private const string SegmentationHead = "SegmentationHead";

    private const int VitDepth = 32;
    private const int TextLayers = 24;
    private const int EncoderLayers = 6;
    private const int DecoderLayers = 6;
    private const int GeometryLayers = 3;

    private static readonly string[] AllGroupNames =
    [
        VisionTrunk, VisionNeck, TextEncoder, FusionEncoder, Decoder, GeometryEncoder, Scoring, SegmentationHead,
    ];

    private readonly List<Tensor> _tensors = [];

    public void Dispose()
    {
        foreach (Tensor tensor in _tensors)
        {
            tensor.Dispose();
        }

        _tensors.Clear();
    }

    [Fact]
    public void Convert_RoutesEverySpecKeyToItsGroupWithRelativeName()
    {
        List<SpecKey> specKeys = BuildSpecKeys();
        Dictionary<string, Tensor> checkpoint = BuildCheckpoint(FullNames(specKeys));

        Sam3WeightGroups groups = Sam3Converter.Convert(checkpoint);

        foreach (SpecKey spec in specKeys)
        {
            string relative = spec.FullName[PrefixOf(spec.Group).Length..];
            IReadOnlyDictionary<string, Tensor> group = GroupOf(groups, spec.Group);
            Assert.True(group.ContainsKey(relative), "Expected '" + relative + "' in group " + spec.Group + ".");
            Assert.Same(checkpoint[spec.FullName], group[relative]);
        }

        Assert.Equal(specKeys.Count, CountRouted(groups));
        Assert.Empty(groups.UnusedKeys);
    }

    [Fact]
    public void Convert_ReportsTrackerScalpAndBufferKeysAsUnused()
    {
        List<SpecKey> specKeys = BuildSpecKeys();
        List<string> unusedKeys = UnusedCheckpointKeys();
        List<string> names = FullNames(specKeys);
        names.AddRange(unusedKeys);
        Dictionary<string, Tensor> checkpoint = BuildCheckpoint(names);

        Sam3WeightGroups groups = Sam3Converter.Convert(checkpoint);

        List<string> expectedUnused = new(unusedKeys);
        expectedUnused.Sort(StringComparer.Ordinal);
        Assert.Equal(expectedUnused, groups.UnusedKeys);
        Assert.Equal(specKeys.Count, CountRouted(groups));
    }

    [Theory]
    [InlineData("detector.unknown_head.weight")]
    [InlineData("detector.dot_prod_scoring.prompt_mlp.layers.2.weight")]
    public void Convert_ThrowsInvalidDataExceptionNamingUnexpectedDetectorKey(string unexpectedKey)
    {
        List<string> names = FullNames(BuildSpecKeys());
        names.Add(unexpectedKey);
        Dictionary<string, Tensor> checkpoint = BuildCheckpoint(names);

        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => { Sam3Converter.Convert(checkpoint); });

        Assert.Contains(unexpectedKey, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Convert_ThrowsInvalidDataExceptionNamingRequiredKeyThatIsMissing()
    {
        const string missingKey = "detector.transformer.decoder.norm.weight";
        List<string> names = FullNames(BuildSpecKeys());
        Assert.True(names.Remove(missingKey), "Test setup: the key must exist in the spec list.");
        Dictionary<string, Tensor> checkpoint = BuildCheckpoint(names);

        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => { Sam3Converter.Convert(checkpoint); });

        Assert.Contains(missingKey, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Convert_ThrowsInvalidDataExceptionForEmptyStateDict()
    {
        Dictionary<string, Tensor> empty = new(StringComparer.Ordinal);

        Assert.Throws<InvalidDataException>(() => { Sam3Converter.Convert(empty); });
    }

    [Fact]
    public void Convert_ThrowsArgumentNullExceptionForNullStateDict()
    {
        Assert.Throws<ArgumentNullException>(() => { Sam3Converter.Convert(null!); });
    }

    [Fact]
    public void Convert_UnwrapsModelPrefixedKeysToTheSameGroups()
    {
        List<SpecKey> specKeys = BuildSpecKeys();
        List<string> names = FullNames(specKeys);
        names.Add("tracker.maskmem_backbone.conv.weight");
        Dictionary<string, Tensor> flat = BuildCheckpoint(names);
        Dictionary<string, Tensor> wrapped = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in flat)
        {
            wrapped.Add("model." + entry.Key, entry.Value);
        }

        Sam3WeightGroups groups = Sam3Converter.Convert(wrapped);

        foreach (SpecKey spec in specKeys)
        {
            string relative = spec.FullName[PrefixOf(spec.Group).Length..];
            IReadOnlyDictionary<string, Tensor> group = GroupOf(groups, spec.Group);
            Assert.True(group.ContainsKey(relative), "Expected '" + relative + "' in group " + spec.Group + ".");
            Assert.Same(flat[spec.FullName], group[relative]);
        }

        Assert.Equal(specKeys.Count, CountRouted(groups));
        Assert.Single(groups.UnusedKeys);
        Assert.Equal("model.tracker.maskmem_backbone.conv.weight", groups.UnusedKeys[0]);
    }

    private static List<string> UnusedCheckpointKeys() =>
    [
        "tracker.maskmem_backbone.conv.weight",
        "tracker.image_encoder.trunk.blocks.0.attn.qkv.weight",
        "multiplex_memory.layers.0.weight",
        "detector.backbone.vision_backbone.convs.3.conv_1x1.weight",
        "detector.backbone.vision_backbone.convs.3.conv_1x1.bias",
        "detector.backbone.vision_backbone.convs.3.conv_3x3.weight",
        "detector.backbone.vision_backbone.convs.3.conv_3x3.bias",
        "detector.backbone.vision_backbone.trunk.blocks.0.attn.freqs_cis",
        "detector.backbone.vision_backbone.trunk.blocks.7.attn.freqs_cis",
        "detector.backbone.language_backbone.encoder.attn_mask",
        "detector.backbone.vision_backbone.sam2_convs.0.conv.weight",
        "detector.transformer.encoder.text_pooling_proj.weight",
        "detector.transformer.encoder.level_embed",
        "detector.transformer.decoder.instance_query_embed.weight",
        "detector.geometry_encoder.mask_label_embed.weight",
        "detector.segmentation_head.presence_head.layers.0.weight",
        "detector.segmentation_head.pixel_decoder.conv_layers.2.weight",
        "detector.segmentation_head.pixel_decoder.norms.2.bias",
        "detector.class_embed.weight",
        "detector.instance_dot_prod_scoring.prompt_proj.weight",
    ];

    /// <summary>Image-path keys written out from SPEC section 4, with each key's owning group.</summary>
    private static List<SpecKey> BuildSpecKeys()
    {
        List<SpecKey> keys = [];

        // 4.1 ViT trunk
        Add(keys, VisionTrunk, "patch_embed.proj.weight");
        Add(keys, VisionTrunk, "pos_embed");
        AddPair(keys, VisionTrunk, "ln_pre");
        for (int block = 0; block < VitDepth; block++)
        {
            string prefix = "blocks." + Index(block) + ".";
            AddPair(keys, VisionTrunk, prefix + "norm1");
            AddPair(keys, VisionTrunk, prefix + "attn.qkv");
            AddPair(keys, VisionTrunk, prefix + "attn.proj");
            AddPair(keys, VisionTrunk, prefix + "norm2");
            AddPair(keys, VisionTrunk, prefix + "mlp.fc1");
            AddPair(keys, VisionTrunk, prefix + "mlp.fc2");
        }

        // 4.2 FPN neck (scale levels 0-2 only)
        AddPair(keys, VisionNeck, "convs.0.dconv_2x2_0");
        AddPair(keys, VisionNeck, "convs.0.dconv_2x2_1");
        AddPair(keys, VisionNeck, "convs.0.conv_1x1");
        AddPair(keys, VisionNeck, "convs.0.conv_3x3");
        AddPair(keys, VisionNeck, "convs.1.dconv_2x2");
        AddPair(keys, VisionNeck, "convs.1.conv_1x1");
        AddPair(keys, VisionNeck, "convs.1.conv_3x3");
        AddPair(keys, VisionNeck, "convs.2.conv_1x1");
        AddPair(keys, VisionNeck, "convs.2.conv_3x3");

        // 4.3 Text tower and resizer
        Add(keys, TextEncoder, "encoder.token_embedding.weight");
        Add(keys, TextEncoder, "encoder.positional_embedding");
        for (int layer = 0; layer < TextLayers; layer++)
        {
            string prefix = "encoder.transformer.resblocks." + Index(layer) + ".";
            AddAttention(keys, TextEncoder, prefix + "attn");
            AddPair(keys, TextEncoder, prefix + "ln_1");
            AddPair(keys, TextEncoder, prefix + "ln_2");
            AddPair(keys, TextEncoder, prefix + "mlp.c_fc");
            AddPair(keys, TextEncoder, prefix + "mlp.c_proj");
        }

        AddPair(keys, TextEncoder, "encoder.ln_final");
        Add(keys, TextEncoder, "encoder.text_projection");
        AddPair(keys, TextEncoder, "resizer");

        // 4.4 Fusion encoder
        for (int layer = 0; layer < EncoderLayers; layer++)
        {
            string prefix = "layers." + Index(layer) + ".";
            AddAttention(keys, FusionEncoder, prefix + "self_attn");
            AddAttention(keys, FusionEncoder, prefix + "cross_attn_image");
            AddPair(keys, FusionEncoder, prefix + "linear1");
            AddPair(keys, FusionEncoder, prefix + "linear2");
            AddPair(keys, FusionEncoder, prefix + "norm1");
            AddPair(keys, FusionEncoder, prefix + "norm2");
            AddPair(keys, FusionEncoder, prefix + "norm3");
        }

        // 4.5 Decoder
        Add(keys, Decoder, "query_embed.weight");
        Add(keys, Decoder, "reference_points.weight");
        Add(keys, Decoder, "presence_token.weight");
        AddPair(keys, Decoder, "norm");
        AddPair(keys, Decoder, "bbox_embed.layers.0");
        AddPair(keys, Decoder, "bbox_embed.layers.1");
        AddPair(keys, Decoder, "bbox_embed.layers.2");
        AddPair(keys, Decoder, "ref_point_head.layers.0");
        AddPair(keys, Decoder, "ref_point_head.layers.1");
        AddPair(keys, Decoder, "boxRPB_embed_x.layers.0");
        AddPair(keys, Decoder, "boxRPB_embed_x.layers.1");
        AddPair(keys, Decoder, "boxRPB_embed_y.layers.0");
        AddPair(keys, Decoder, "boxRPB_embed_y.layers.1");
        AddPair(keys, Decoder, "presence_token_head.layers.0");
        AddPair(keys, Decoder, "presence_token_head.layers.1");
        AddPair(keys, Decoder, "presence_token_head.layers.2");
        AddPair(keys, Decoder, "presence_token_out_norm");
        for (int layer = 0; layer < DecoderLayers; layer++)
        {
            string prefix = "layers." + Index(layer) + ".";
            AddAttention(keys, Decoder, prefix + "cross_attn");
            AddAttention(keys, Decoder, prefix + "ca_text");
            AddAttention(keys, Decoder, prefix + "self_attn");
            AddPair(keys, Decoder, prefix + "linear1");
            AddPair(keys, Decoder, prefix + "linear2");
            AddPair(keys, Decoder, prefix + "norm1");
            AddPair(keys, Decoder, prefix + "norm2");
            AddPair(keys, Decoder, prefix + "norm3");
            AddPair(keys, Decoder, prefix + "catext_norm");
        }

        // 4.6 Geometry encoder
        Add(keys, GeometryEncoder, "label_embed.weight");
        Add(keys, GeometryEncoder, "cls_embed.weight");
        AddPair(keys, GeometryEncoder, "points_direct_project");
        AddPair(keys, GeometryEncoder, "points_pool_project");
        AddPair(keys, GeometryEncoder, "points_pos_enc_project");
        AddPair(keys, GeometryEncoder, "boxes_direct_project");
        AddPair(keys, GeometryEncoder, "boxes_pool_project");
        AddPair(keys, GeometryEncoder, "boxes_pos_enc_project");
        AddPair(keys, GeometryEncoder, "img_pre_norm");
        AddPair(keys, GeometryEncoder, "final_proj");
        AddPair(keys, GeometryEncoder, "norm");
        for (int layer = 0; layer < GeometryLayers; layer++)
        {
            string prefix = "encode." + Index(layer) + ".";
            AddAttention(keys, GeometryEncoder, prefix + "self_attn");
            AddAttention(keys, GeometryEncoder, prefix + "cross_attn_image");
            AddPair(keys, GeometryEncoder, prefix + "linear1");
            AddPair(keys, GeometryEncoder, prefix + "linear2");
            AddPair(keys, GeometryEncoder, prefix + "norm1");
            AddPair(keys, GeometryEncoder, prefix + "norm2");
            AddPair(keys, GeometryEncoder, prefix + "norm3");
        }

        AddPair(keys, GeometryEncoder, "encode_norm");

        // 4.7 Dot-product scorer
        AddPair(keys, Scoring, "prompt_mlp.layers.0");
        AddPair(keys, Scoring, "prompt_mlp.layers.1");
        AddPair(keys, Scoring, "prompt_mlp.out_norm");
        AddPair(keys, Scoring, "prompt_proj");
        AddPair(keys, Scoring, "hs_proj");

        // 4.8 Segmentation head
        AddPair(keys, SegmentationHead, "pixel_decoder.conv_layers.0");
        AddPair(keys, SegmentationHead, "pixel_decoder.conv_layers.1");
        AddPair(keys, SegmentationHead, "pixel_decoder.norms.0");
        AddPair(keys, SegmentationHead, "pixel_decoder.norms.1");
        AddPair(keys, SegmentationHead, "mask_predictor.mask_embed.layers.0");
        AddPair(keys, SegmentationHead, "mask_predictor.mask_embed.layers.1");
        AddPair(keys, SegmentationHead, "mask_predictor.mask_embed.layers.2");
        AddAttention(keys, SegmentationHead, "cross_attend_prompt");
        AddPair(keys, SegmentationHead, "cross_attn_norm");
        AddPair(keys, SegmentationHead, "instance_seg_head");
        AddPair(keys, SegmentationHead, "semantic_seg_head");

        return keys;
    }

    private static void Add(List<SpecKey> keys, string group, string relative)
    {
        keys.Add(new SpecKey(group, PrefixOf(group) + relative));
    }

    private static void AddPair(List<SpecKey> keys, string group, string name)
    {
        Add(keys, group, name + ".weight");
        Add(keys, group, name + ".bias");
    }

    private static void AddAttention(List<SpecKey> keys, string group, string name)
    {
        Add(keys, group, name + ".in_proj_weight");
        Add(keys, group, name + ".in_proj_bias");
        AddPair(keys, group, name + ".out_proj");
    }

    private static string Index(int value) => value.ToString(CultureInfo.InvariantCulture);

    private static string PrefixOf(string group) => group switch
    {
        VisionTrunk => "detector.backbone.vision_backbone.trunk.",
        VisionNeck => "detector.backbone.vision_backbone.",
        TextEncoder => "detector.backbone.language_backbone.",
        FusionEncoder => "detector.transformer.encoder.",
        Decoder => "detector.transformer.decoder.",
        GeometryEncoder => "detector.geometry_encoder.",
        Scoring => "detector.dot_prod_scoring.",
        SegmentationHead => "detector.segmentation_head.",
        _ => throw new ArgumentOutOfRangeException(nameof(group), group, "Unknown group."),
    };

    private static IReadOnlyDictionary<string, Tensor> GroupOf(Sam3WeightGroups groups, string group) => group switch
    {
        VisionTrunk => groups.VisionTrunk,
        VisionNeck => groups.VisionNeck,
        TextEncoder => groups.TextEncoder,
        FusionEncoder => groups.FusionEncoder,
        Decoder => groups.Decoder,
        GeometryEncoder => groups.GeometryEncoder,
        Scoring => groups.Scoring,
        SegmentationHead => groups.SegmentationHead,
        _ => throw new ArgumentOutOfRangeException(nameof(group), group, "Unknown group."),
    };

    private static int CountRouted(Sam3WeightGroups groups)
    {
        int total = 0;
        foreach (string group in AllGroupNames)
        {
            total += GroupOf(groups, group).Count;
        }

        return total;
    }

    private static List<string> FullNames(List<SpecKey> specKeys)
    {
        List<string> names = new(specKeys.Count);
        foreach (SpecKey spec in specKeys)
        {
            names.Add(spec.FullName);
        }

        return names;
    }

    private Dictionary<string, Tensor> BuildCheckpoint(List<string> keys)
    {
        Dictionary<string, Tensor> checkpoint = new(StringComparer.Ordinal);
        foreach (string key in keys)
        {
            Tensor tensor = new(new TensorShape(1), DType.F32);
            _tensors.Add(tensor);
            checkpoint.Add(key, tensor);
        }

        return checkpoint;
    }

    private sealed record SpecKey(string Group, string FullName);
}
