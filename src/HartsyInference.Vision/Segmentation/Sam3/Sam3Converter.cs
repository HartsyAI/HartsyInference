using System.Globalization;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Routes a SAM 3 checkpoint state dict into image-path weight groups and checks their key coverage.</summary>
/// <remarks>
/// Input is a flat map of full checkpoint names. <c>PytorchPickleLoader.Load</c> with default flags descends one
/// <c>"model"</c> envelope, so flat keys are the normal input. Keys carrying a leading <c>model.</c> (a recursive
/// flatten) are accepted when the remainder starts with <c>detector.</c> or <c>tracker.</c>.
/// Keys outside <c>detector.</c> (tracker, video and multiplex modules) are never read by the image path and are
/// reported in <see cref="Sam3WeightGroups.UnusedKeys"/>. The tensors are not owned, disposed or copied.
/// </remarks>
public static class Sam3Converter
{
    private const string DetectorPrefix = "detector.";
    private const string TrackerPrefix = "tracker.";
    private const string ModelWrapperPrefix = "model.";

    private static readonly Sam3Config Defaults = new();

    /// <summary>Component routing order. The trunk must precede the neck, because the neck prefix contains the trunk prefix.</summary>
    private static readonly Sam3Part[] RouteOrder =
    [
        Sam3Part.VisionTrunk,
        Sam3Part.VisionNeck,
        Sam3Part.TextEncoder,
        Sam3Part.FusionEncoder,
        Sam3Part.Decoder,
        Sam3Part.GeometryEncoder,
        Sam3Part.Scoring,
        Sam3Part.SegmentationHead,
    ];

    /// <summary>Full-name prefixes of checkpoint keys the image path deliberately does not load (scalp level, dead stages, never-built modules).</summary>
    private static readonly string[] UnusedPrefixes =
    [
        "detector.backbone.vision_backbone.convs.3.",
        "detector.backbone.vision_backbone.sam2_convs.",
        "detector.transformer.encoder.text_pooling_proj.",
        "detector.transformer.encoder.level_embed",
        "detector.transformer.decoder.instance_",
        "detector.transformer.decoder.roi_pooler.",
        "detector.transformer.decoder.fine_layers",
        "detector.geometry_encoder.mask_encoder.",
        "detector.geometry_encoder.mask_label_embed",
        "detector.segmentation_head.presence_head.",
        "detector.segmentation_head.pixel_decoder.conv_layers.2.",
        "detector.segmentation_head.pixel_decoder.norms.2.",
        "detector.class_embed.",
        "detector.instance_dot_prod_scoring.",
        "detector.instance_class_embed.",
    ];

    /// <summary>Suffixes of persistent non-learned buffers (precomputed RoPE, causal masks, position ids), dropped without error.</summary>
    private static readonly string[] NonLearnedBufferSuffixes =
    [
        ".freqs_cis",
        ".attn_mask",
        ".position_ids",
    ];

    /// <summary>Splits a SAM 3 state dict into image-path weight groups, using the upstream default sizes.</summary>
    /// <param name="stateDict">Checkpoint tensors keyed by full name. Not modified, copied or disposed.</param>
    /// <returns>Weight groups keyed relative to each component prefix, plus the sorted full names of unused keys.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="stateDict"/> is null.</exception>
    /// <exception cref="InvalidDataException">The dict is empty, holds an unexpected <c>detector.</c> key, or lacks a required key.</exception>
    public static Sam3WeightGroups Convert(IReadOnlyDictionary<string, Tensor> stateDict) => Convert(stateDict, Defaults);

    /// <summary>Splits a SAM 3 state dict into image-path weight groups sized by <paramref name="config"/>.</summary>
    /// <param name="stateDict">Checkpoint tensors keyed by full name. Not modified, copied or disposed.</param>
    /// <param name="config">Hyperparameters that set the expected layer and block counts.</param>
    /// <returns>Weight groups keyed relative to each component prefix, plus the sorted full names of unused keys.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="stateDict"/> or <paramref name="config"/> is null.</exception>
    /// <exception cref="InvalidDataException">The dict is empty, holds an unexpected <c>detector.</c> key, or lacks a required key.</exception>
    public static Sam3WeightGroups Convert(IReadOnlyDictionary<string, Tensor> stateDict, Sam3Config config)
    {
        ArgumentNullException.ThrowIfNull(stateDict);
        ArgumentNullException.ThrowIfNull(config);
        if (stateDict.Count == 0)
        {
            throw new InvalidDataException("SAM 3 state dict is empty; expected detector.* tensors.");
        }

        // Required relative keys per component, sorted ordinally so error messages are deterministic.
        Dictionary<Sam3Part, SortedSet<string>> expectedKeys = BuildExpectedKeys(config);
        Dictionary<Sam3Part, Dictionary<string, Tensor>> buckets = CreateBuckets();
        List<string> unused = [];

        foreach (KeyValuePair<string, Tensor> entry in stateDict)
        {
            string key = StripModelWrapper(entry.Key);
            if (!key.StartsWith(DetectorPrefix, StringComparison.Ordinal))
            {
                // Tracker, video and multiplex modules: the image path never reads them.
                unused.Add(entry.Key);
                continue;
            }

            if (IsDeliberatelyUnused(key))
            {
                unused.Add(entry.Key);
                continue;
            }

            if (!TryRoute(key, out Sam3Part part, out string partKey))
            {
                throw new InvalidDataException("SAM 3 checkpoint key '" + entry.Key + "' matches no image-path component.");
            }

            if (!expectedKeys[part].Contains(partKey))
            {
                throw new InvalidDataException("SAM 3 checkpoint key '" + entry.Key + "' is not an expected key of the " + part + " group.");
            }

            buckets[part].Add(partKey, entry.Value);
        }

        foreach (Sam3Part part in Enum.GetValues<Sam3Part>())
        {
            foreach (string partKey in expectedKeys[part])
            {
                if (!buckets[part].ContainsKey(partKey))
                {
                    throw new InvalidDataException("SAM 3 checkpoint is missing required key '" + PrefixOf(part) + partKey + "'.");
                }
            }
        }

        unused.Sort(StringComparer.Ordinal);
        return new Sam3WeightGroups
        {
            VisionTrunk = buckets[Sam3Part.VisionTrunk],
            VisionNeck = buckets[Sam3Part.VisionNeck],
            TextEncoder = buckets[Sam3Part.TextEncoder],
            FusionEncoder = buckets[Sam3Part.FusionEncoder],
            Decoder = buckets[Sam3Part.Decoder],
            GeometryEncoder = buckets[Sam3Part.GeometryEncoder],
            Scoring = buckets[Sam3Part.Scoring],
            SegmentationHead = buckets[Sam3Part.SegmentationHead],
            UnusedKeys = unused,
        };
    }

    private static Dictionary<Sam3Part, Dictionary<string, Tensor>> CreateBuckets()
    {
        Dictionary<Sam3Part, Dictionary<string, Tensor>> buckets = [];
        foreach (Sam3Part part in Enum.GetValues<Sam3Part>())
        {
            buckets[part] = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        }

        return buckets;
    }

    /// <summary>Removes a leading <c>model.</c> only when the remainder is a detector or tracker key.</summary>
    private static string StripModelWrapper(string key)
    {
        if (!key.StartsWith(ModelWrapperPrefix, StringComparison.Ordinal))
        {
            return key;
        }

        string inner = key[ModelWrapperPrefix.Length..];
        bool isCheckpointRoot = inner.StartsWith(DetectorPrefix, StringComparison.Ordinal)
            || inner.StartsWith(TrackerPrefix, StringComparison.Ordinal);
        return isCheckpointRoot ? inner : key;
    }

    private static bool IsDeliberatelyUnused(string key)
    {
        foreach (string prefix in UnusedPrefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                return true;
            }
        }

        foreach (string suffix in NonLearnedBufferSuffixes)
        {
            if (key.EndsWith(suffix, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }

    private static bool TryRoute(string key, out Sam3Part part, out string partKey)
    {
        foreach (Sam3Part candidate in RouteOrder)
        {
            string prefix = PrefixOf(candidate);
            if (key.StartsWith(prefix, StringComparison.Ordinal))
            {
                part = candidate;
                partKey = key[prefix.Length..];
                return true;
            }
        }

        part = default;
        partKey = string.Empty;
        return false;
    }

    private static string PrefixOf(Sam3Part part) => part switch
    {
        Sam3Part.VisionTrunk => "detector.backbone.vision_backbone.trunk.",
        Sam3Part.VisionNeck => "detector.backbone.vision_backbone.",
        Sam3Part.TextEncoder => "detector.backbone.language_backbone.",
        Sam3Part.FusionEncoder => "detector.transformer.encoder.",
        Sam3Part.Decoder => "detector.transformer.decoder.",
        Sam3Part.GeometryEncoder => "detector.geometry_encoder.",
        Sam3Part.Scoring => "detector.dot_prod_scoring.",
        Sam3Part.SegmentationHead => "detector.segmentation_head.",
        _ => throw new ArgumentOutOfRangeException(nameof(part), part, "Unknown SAM 3 component."),
    };

    private static Dictionary<Sam3Part, SortedSet<string>> BuildExpectedKeys(Sam3Config config)
    {
        Dictionary<Sam3Part, SortedSet<string>> expected = [];
        expected[Sam3Part.VisionTrunk] = VisionTrunkKeys(config);
        expected[Sam3Part.VisionNeck] = VisionNeckKeys();
        expected[Sam3Part.TextEncoder] = TextEncoderKeys(config);
        expected[Sam3Part.FusionEncoder] = FusionEncoderKeys(config);
        expected[Sam3Part.Decoder] = DecoderKeys(config);
        expected[Sam3Part.GeometryEncoder] = GeometryEncoderKeys(config);
        expected[Sam3Part.Scoring] = ScoringKeys();
        expected[Sam3Part.SegmentationHead] = SegmentationHeadKeys();
        return expected;
    }

    private static SortedSet<string> VisionTrunkKeys(Sam3Config config)
    {
        SortedSet<string> keys = NewKeySet();
        keys.Add("patch_embed.proj.weight");
        keys.Add("pos_embed");
        AddPair(keys, "ln_pre");
        for (int block = 0; block < config.VitDepth; block++)
        {
            string prefix = "blocks." + block.ToString(CultureInfo.InvariantCulture) + ".";
            AddPair(keys, prefix + "norm1");
            AddPair(keys, prefix + "attn.qkv");
            AddPair(keys, prefix + "attn.proj");
            AddPair(keys, prefix + "norm2");
            AddPair(keys, prefix + "mlp.fc1");
            AddPair(keys, prefix + "mlp.fc2");
        }

        return keys;
    }

    private static SortedSet<string> VisionNeckKeys()
    {
        SortedSet<string> keys = NewKeySet();
        AddPair(keys, "convs.0.dconv_2x2_0");
        AddPair(keys, "convs.0.dconv_2x2_1");
        AddPair(keys, "convs.0.conv_1x1");
        AddPair(keys, "convs.0.conv_3x3");
        AddPair(keys, "convs.1.dconv_2x2");
        AddPair(keys, "convs.1.conv_1x1");
        AddPair(keys, "convs.1.conv_3x3");
        AddPair(keys, "convs.2.conv_1x1");
        AddPair(keys, "convs.2.conv_3x3");
        return keys;
    }

    private static SortedSet<string> TextEncoderKeys(Sam3Config config)
    {
        SortedSet<string> keys = NewKeySet();
        keys.Add("encoder.token_embedding.weight");
        keys.Add("encoder.positional_embedding");
        for (int layer = 0; layer < config.TextLayers; layer++)
        {
            string prefix = "encoder.transformer.resblocks." + layer.ToString(CultureInfo.InvariantCulture) + ".";
            AddAttention(keys, prefix + "attn");
            AddPair(keys, prefix + "ln_1");
            AddPair(keys, prefix + "ln_2");
            AddPair(keys, prefix + "mlp.c_fc");
            AddPair(keys, prefix + "mlp.c_proj");
        }

        AddPair(keys, "encoder.ln_final");
        keys.Add("encoder.text_projection");
        AddPair(keys, "resizer");
        return keys;
    }

    private static SortedSet<string> FusionEncoderKeys(Sam3Config config)
    {
        SortedSet<string> keys = NewKeySet();
        for (int layer = 0; layer < config.EncoderLayers; layer++)
        {
            string prefix = "layers." + layer.ToString(CultureInfo.InvariantCulture) + ".";
            AddAttention(keys, prefix + "self_attn");
            AddAttention(keys, prefix + "cross_attn_image");
            AddPair(keys, prefix + "linear1");
            AddPair(keys, prefix + "linear2");
            AddPair(keys, prefix + "norm1");
            AddPair(keys, prefix + "norm2");
            AddPair(keys, prefix + "norm3");
        }

        return keys;
    }

    private static SortedSet<string> DecoderKeys(Sam3Config config)
    {
        SortedSet<string> keys = NewKeySet();
        keys.Add("query_embed.weight");
        keys.Add("reference_points.weight");
        keys.Add("presence_token.weight");
        AddPair(keys, "norm");
        AddPair(keys, "bbox_embed.layers.0");
        AddPair(keys, "bbox_embed.layers.1");
        AddPair(keys, "bbox_embed.layers.2");
        AddPair(keys, "ref_point_head.layers.0");
        AddPair(keys, "ref_point_head.layers.1");
        AddPair(keys, "boxRPB_embed_x.layers.0");
        AddPair(keys, "boxRPB_embed_x.layers.1");
        AddPair(keys, "boxRPB_embed_y.layers.0");
        AddPair(keys, "boxRPB_embed_y.layers.1");
        AddPair(keys, "presence_token_head.layers.0");
        AddPair(keys, "presence_token_head.layers.1");
        AddPair(keys, "presence_token_head.layers.2");
        AddPair(keys, "presence_token_out_norm");
        for (int layer = 0; layer < config.DecoderLayers; layer++)
        {
            string prefix = "layers." + layer.ToString(CultureInfo.InvariantCulture) + ".";
            AddAttention(keys, prefix + "cross_attn");
            AddAttention(keys, prefix + "ca_text");
            AddAttention(keys, prefix + "self_attn");
            AddPair(keys, prefix + "linear1");
            AddPair(keys, prefix + "linear2");
            AddPair(keys, prefix + "norm1");
            AddPair(keys, prefix + "norm2");
            AddPair(keys, prefix + "norm3");
            AddPair(keys, prefix + "catext_norm");
        }

        return keys;
    }

    private static SortedSet<string> GeometryEncoderKeys(Sam3Config config)
    {
        SortedSet<string> keys = NewKeySet();
        keys.Add("label_embed.weight");
        keys.Add("cls_embed.weight");
        AddPair(keys, "points_direct_project");
        AddPair(keys, "points_pool_project");
        AddPair(keys, "points_pos_enc_project");
        AddPair(keys, "boxes_direct_project");
        AddPair(keys, "boxes_pool_project");
        AddPair(keys, "boxes_pos_enc_project");
        AddPair(keys, "img_pre_norm");
        AddPair(keys, "final_proj");
        AddPair(keys, "norm");
        for (int layer = 0; layer < config.GeometryLayers; layer++)
        {
            string prefix = "encode." + layer.ToString(CultureInfo.InvariantCulture) + ".";
            AddAttention(keys, prefix + "self_attn");
            AddAttention(keys, prefix + "cross_attn_image");
            AddPair(keys, prefix + "linear1");
            AddPair(keys, prefix + "linear2");
            AddPair(keys, prefix + "norm1");
            AddPair(keys, prefix + "norm2");
            AddPair(keys, prefix + "norm3");
        }

        AddPair(keys, "encode_norm");
        return keys;
    }

    private static SortedSet<string> ScoringKeys()
    {
        SortedSet<string> keys = NewKeySet();
        AddPair(keys, "prompt_mlp.layers.0");
        AddPair(keys, "prompt_mlp.layers.1");
        AddPair(keys, "prompt_mlp.out_norm");
        AddPair(keys, "prompt_proj");
        AddPair(keys, "hs_proj");
        return keys;
    }

    private static SortedSet<string> SegmentationHeadKeys()
    {
        SortedSet<string> keys = NewKeySet();
        AddPair(keys, "pixel_decoder.conv_layers.0");
        AddPair(keys, "pixel_decoder.conv_layers.1");
        AddPair(keys, "pixel_decoder.norms.0");
        AddPair(keys, "pixel_decoder.norms.1");
        AddPair(keys, "mask_predictor.mask_embed.layers.0");
        AddPair(keys, "mask_predictor.mask_embed.layers.1");
        AddPair(keys, "mask_predictor.mask_embed.layers.2");
        AddAttention(keys, "cross_attend_prompt");
        AddPair(keys, "cross_attn_norm");
        AddPair(keys, "instance_seg_head");
        AddPair(keys, "semantic_seg_head");
        return keys;
    }

    private static SortedSet<string> NewKeySet() => new(StringComparer.Ordinal);

    /// <summary>Adds <c>name.weight</c> and <c>name.bias</c>.</summary>
    private static void AddPair(SortedSet<string> keys, string name)
    {
        keys.Add(name + ".weight");
        keys.Add(name + ".bias");
    }

    /// <summary>Adds the four tensors of a <c>nn.MultiheadAttention</c> module stored under <paramref name="name"/>.</summary>
    private static void AddAttention(SortedSet<string> keys, string name)
    {
        keys.Add(name + ".in_proj_weight");
        keys.Add(name + ".in_proj_bias");
        AddPair(keys, name + ".out_proj");
    }

    /// <summary>Image-path components that own a group of checkpoint keys.</summary>
    private enum Sam3Part
    {
        VisionTrunk,
        VisionNeck,
        TextEncoder,
        FusionEncoder,
        Decoder,
        GeometryEncoder,
        Scoring,
        SegmentationHead,
    }
}
