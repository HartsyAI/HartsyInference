using System.Text.RegularExpressions;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Turns a LongCat GGUF repack in the ComfyUI Wan layout (<c>self_attn.q/k/v/o</c>, <c>cross_attn.q/k/v/o</c>, <c>norm3</c>,
/// <c>norm_x</c>, <c>head</c>, <c>patch_embedding</c>, <c>multitalk_audio_proj</c>) into the engine's LongCat key names. The engine reads one
/// <c>attn.qkv</c> and one <c>cross_attn.kv_linear</c> weight, so those parts are joined along the output rows, in that order.</summary>
/// <remarks>Parts of one dtype are joined as stored bytes: every row of a block-quantized weight is a whole number of blocks, so the joined
/// tensor decodes to the parts' own values. Parts of different dtypes (the "_M" and "_S" mixes store <c>v</c> wider than <c>q</c> and
/// <c>k</c>) have no common packed form; they are dequantized and joined as BF16, rounded to nearest, ties to even. Input shapes are the
/// engine's <c>[out, in]</c> order (<see cref="Checkpoints.CheckpointSource"/>).</remarks>
public static partial class LongCatGgufConverter
{
    /// <summary>True for the ComfyUI Wan layout of a LongCat DiT: Wan's split self-attention beside LongCat's SwiGLU <c>ffn.w1</c>.</summary>
    public static bool IsComfyLayout(IEnumerable<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        bool splitSelf = false, swiglu = false;
        foreach (string key in keys)
        {
            splitSelf |= key == "blocks.0.self_attn.q.weight";
            swiglu |= key == "blocks.0.ffn.w1.weight";
        }
        return splitSelf && swiglu;
    }

    /// <summary>The engine name of a tensor, and its place in a joined weight (or -1 for a tensor that is not joined).</summary>
    public static (string Name, int Part) EngineName(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        Match part = SelfPart().Match(key);
        if (part.Success)
            return ($"{part.Groups[1].Value}.attn.qkv.{part.Groups[3].Value}", "qkv".IndexOf(part.Groups[2].Value[0]));
        part = CrossPart().Match(key);
        if (part.Success)
            return ($"{part.Groups[1].Value}.cross_attn.kv_linear.{part.Groups[3].Value}", "kv".IndexOf(part.Groups[2].Value[0]));
        foreach ((string from, string to) in Prefixes)
            if (key.StartsWith(from, StringComparison.Ordinal))
                return (to + key[from.Length..], -1);
        foreach ((string from, string to) in Fragments)
            if (key.Contains(from, StringComparison.Ordinal))
                return (key.Replace(from, to, StringComparison.Ordinal), -1);
        return (key, -1);
    }

    /// <summary>Returns the weights under the engine's names. A tensor that is only renamed is the source's own; a joined one is new and is added
    /// to <paramref name="owned"/>, which the caller disposes. <paramref name="widened"/> counts the joined weights that had to become
    /// BF16.</summary>
    public static Dictionary<string, Tensor> Convert(IReadOnlyDictionary<string, Tensor> source, List<Tensor> owned, out int widened)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(owned);
        widened = 0;
        Dictionary<string, Tensor> result = new(source.Count, StringComparer.Ordinal);
        Dictionary<string, SortedDictionary<int, Tensor>> joins = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in source)
        {
            (string name, int part) = EngineName(key);
            if (part < 0)
            {
                if (!result.TryAdd(name, tensor))
                    throw new InvalidDataException($"Two tensors map to '{name}' (one of them '{key}').");
                continue;
            }
            if (!joins.TryGetValue(name, out SortedDictionary<int, Tensor>? parts))
                joins[name] = parts = new SortedDictionary<int, Tensor>();
            parts.Add(part, tensor);
        }
        foreach ((string name, SortedDictionary<int, Tensor> parts) in joins)
        {
            int expected = name.Contains(".attn.qkv.", StringComparison.Ordinal) ? 3 : 2;
            if (parts.Count != expected)
                throw new InvalidDataException($"'{name}' needs {expected} parts; the file holds {parts.Count}.");
            if (result.ContainsKey(name))
                throw new InvalidDataException($"The file holds both '{name}' and its split parts.");
            Tensor joined = Join(name, [.. parts.Values], out bool asBf16);
            owned.Add(joined);
            result[name] = joined;
            if (asBf16)
                widened++;
        }
        return result;
    }

    private static Tensor Join(string name, Tensor[] parts, out bool asBf16)
    {
        Tensor first = parts[0];
        int rank = first.Shape.Rank;
        long rows = 0;
        bool sameDtype = true;
        foreach (Tensor part in parts)
        {
            if (part.Shape.Rank != rank || rank is < 1 or > 2 || (rank == 2 && part.Shape[1] != first.Shape[1]))
                throw new InvalidDataException($"The parts of '{name}' do not join along their rows: {first.Shape} and {part.Shape}.");
            if (part.QuantInfo is not null)
                throw new InvalidDataException($"A part of '{name}' carries a companion scale; only GGUF block quants and dense floats are joined.");
            rows = checked(rows + part.Shape[0]);
            sameDtype &= part.DType == first.DType;
        }
        TensorShape shape = rank == 1 ? new TensorShape(rows) : new TensorShape(rows, first.Shape[1]);
        asBf16 = !sameDtype;
        if (sameDtype)
        {
            if (first.DType.IsQuantized && (rank != 2 || first.Shape[1] % first.DType.BlockElementCount != 0))
                throw new InvalidDataException($"'{name}' is {first.DType.Name} {first.Shape}; a row must be a whole number of blocks to join.");
            Tensor joined = new(shape, first.DType);
            Span<byte> bytes = joined.AsSpan<byte>();
            int at = 0;
            foreach (Tensor part in parts)
            {
                ReadOnlySpan<byte> from = part.AsReadOnlySpan<byte>();
                from.CopyTo(bytes[at..]);
                at += from.Length;
            }
            return joined;
        }
        Tensor bf16 = new(shape, DType.BF16);
        Span<ushort> bits = bf16.AsSpan<ushort>();
        int written = 0;
        foreach (Tensor part in parts)
        {
            using Tensor? wide = part.DType.IsQuantized ? GgufDequantizer.Dequantize(part, DType.F32)
                : part.DType == DType.F32 ? null : part.CastTo(DType.F32);
            ReadOnlySpan<float> values = (wide ?? part).AsReadOnlySpan<float>();
            for (int i = 0; i < values.Length; i++)
                bits[written + i] = BFloat16Bits.FromFloat(values[i]);
            written += values.Length;
        }
        return bf16;
    }

    private static readonly (string From, string To)[] Prefixes =
    [
        ("patch_embedding.", "x_embedder.proj."), ("text_embedding.", "y_embedder.y_proj."), ("time_embedding.", "t_embedder."),
        ("head.head.", "final_layer.linear."), ("head.modulation.", "final_layer.adaLN_modulation."), ("multitalk_audio_proj.", "audio_proj."),
    ];

    private static readonly (string From, string To)[] Fragments =
    [
        (".modulation.1.", ".adaLN_modulation.1."), (".audio_modulation.1.", ".audio_adaLN_modulation.1."), (".self_attn.o.", ".attn.proj."),
        (".self_attn.norm_q.", ".attn.q_norm."), (".self_attn.norm_k.", ".attn.k_norm."), (".cross_attn.q.", ".cross_attn.q_linear."),
        (".cross_attn.o.", ".cross_attn.proj."), (".cross_attn.norm_q.", ".cross_attn.q_norm."), (".cross_attn.norm_k.", ".cross_attn.k_norm."),
        (".norm3.", ".pre_crs_attn_norm."), (".norm_x.", ".pre_video_crs_attn_norm."),
    ];

    [GeneratedRegex(@"^(blocks\.\d+)\.self_attn\.([qkv])\.(weight|bias)$")]
    private static partial Regex SelfPart();

    // Wan's text cross-attention only: audio_cross_attn already holds one kv_linear.
    [GeneratedRegex(@"^(blocks\.\d+)\.cross_attn\.([kv])\.(weight|bias)$")]
    private static partial Regex CrossPart();
}
