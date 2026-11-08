using System.Text.RegularExpressions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts a BiRefNet checkpoint (<c>ZhengPeng7/BiRefNet-matting</c>, the upstream <c>BiRefNet</c> module
/// state dict) into the weight dictionary the C# BiRefNet loads. The only key rewrite is the Swin fused <c>attn.qkv</c>
/// projection, split into separate <c>attn.q</c>/<c>attn.k</c>/<c>attn.v</c> row blocks (views into the source tensor,
/// valid while its loader is alive). Every checkpoint key must either be routed or be one of the documented
/// inference-unused buffers; an unrecognized key fails the conversion instead of being silently carried along.</summary>
public static class BiRefNetCheckpointConverter
{
    /// <summary>Checkpoint modules the inference path loads: the whole Swin encoder, the squeeze block, and the decoder
    /// sub-modules (the training-only heads are matched by <see cref="IsDocumentedUnused"/> before this runs).</summary>
    private static readonly Regex RoutedModule = new(
        @"^(bb\.|squeeze_module\.0\.|decoder\.(ipt_blk[1-5]|decoder_block[1-4]|lateral_block[2-4]|gdt_convs_[234]|gdt_convs_attn_[234]|conv_out1)\.)",
        RegexOptions.CultureInvariant);

    /// <summary>True for a checkpoint key that inference never reads: the BatchNorm step counter, the Swin relative
    /// position index (a deterministic buffer rebuilt from the window size), and the decoder's multi-scale and
    /// gradient-label heads, which only run when the upstream module is in training mode.</summary>
    public static bool IsDocumentedUnused(string key) =>
        key.EndsWith(".num_batches_tracked", StringComparison.Ordinal)
        || key.EndsWith(".attn.relative_position_index", StringComparison.Ordinal)
        || key.StartsWith("decoder.conv_ms_spvn_", StringComparison.Ordinal)
        || key.StartsWith("decoder.gdt_convs_pred_", StringComparison.Ordinal);

    /// <summary>Routes every checkpoint tensor to its C# weight key. Throws when a key is neither routed nor documented unused.</summary>
    public static Dictionary<string, Tensor> Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        Dictionary<string, Tensor> output = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in all)
        {
            if (IsDocumentedUnused(key))
                continue;
            if (key.EndsWith(".attn.qkv.weight", StringComparison.Ordinal) || key.EndsWith(".attn.qkv.bias", StringComparison.Ordinal))
            {
                SplitQkv(key, tensor, output);
                continue;
            }
            if (!RoutedModule.IsMatch(key))
                throw new InvalidOperationException($"BiRefNet checkpoint key '{key}' is neither routed nor a documented inference-unused buffer.");
            output[key] = tensor;
        }
        return output;
    }

    private static void SplitQkv(string key, Tensor fused, Dictionary<string, Tensor> output)
    {
        int rows = (int)fused.Shape[0];
        if (rows % 3 != 0)
            throw new InvalidOperationException($"BiRefNet '{key}' has {rows} rows; a fused qkv projection needs a multiple of 3.");
        int third = rows / 3;
        string stem = key[..key.LastIndexOf(".qkv.", StringComparison.Ordinal)];
        string suffix = key[(key.LastIndexOf(".qkv.", StringComparison.Ordinal) + ".qkv.".Length)..];
        output[$"{stem}.q.{suffix}"] = fused.SliceRows(0, third);
        output[$"{stem}.k.{suffix}"] = fused.SliceRows(third, third);
        output[$"{stem}.v.{suffix}"] = fused.SliceRows(2 * third, third);
    }
}
