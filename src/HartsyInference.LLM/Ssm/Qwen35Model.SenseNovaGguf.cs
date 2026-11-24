using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

public sealed unsafe partial class Qwen35Model
{
    /// <summary><c>general.architecture</c> of the official SenseNova-Loop GGUFs (OpenSenseNova/llama.cpp <c>sensenova-loop</c>).</summary>
    public const string SenseNovaLoopArchitecture = "sensenova_loop";

    /// <summary>The Loop plan a <c>sensenova_loop</c> GGUF declares. <c>block_count</c> counts execution slots (56) plus the MTP block; the
    /// physical layer count follows from the plan.</summary>
    private static SenseNovaLoopConfig ReadSenseNovaLoop(GgufMetadata m, string arch, int slots)
    {
        int start = (int)m.GetUInt32($"{arch}.loop.start"), end = (int)m.GetUInt32($"{arch}.loop.end"), count = (int)m.GetUInt32($"{arch}.loop.count");
        if (count < 1 || end <= start)
        {
            throw new InvalidDataException($"The GGUF declares an invalid Loop interval [{start}, {end}) x {count}.");
        }
        int physical = slots - (count - 1) * (end - start);
        if (m.GetBool($"{arch}.mhc.fixed_identity"))
        {
            throw new NotSupportedException("A SenseNova-Loop GGUF with a fixed-identity mHC mix is not supported.");
        }
        return new SenseNovaLoopConfig(start, end, count, physical, (int)m.GetUInt32($"{arch}.mhc.start"), (int)m.GetUInt32($"{arch}.mhc.end"),
            (int)m.GetUInt32($"{arch}.mhc.streams"), (int)m.GetUInt32($"{arch}.mhc.iterations"), m.GetFloat32($"{arch}.mhc.pre_scale", 1f));
    }

    /// <summary>The RoPE scaling a GGUF declares: YaRN as <c>rope.scaling.*</c>, or none. HF's YaRN truncates the correction range unless a config
    /// says otherwise, and no GGUF key carries that, so it stays on.</summary>
    private static RopeScaling ReadGgufRopeScaling(GgufMetadata m, string arch)
    {
        string? type = m.GetString($"{arch}.rope.scaling.type");
        if (string.IsNullOrEmpty(type) || type == "none")
        {
            return RopeScaling.None;
        }
        if (type != "yarn")
        {
            throw new NotSupportedException($"RoPE scaling '{type}' is not supported for Qwen3.5 GGUFs (none and yarn are).");
        }
        double original = m.GetUInt32($"{arch}.rope.scaling.original_context_length");
        float factor = m.GetFloat32($"{arch}.rope.scaling.factor", 1f);
        if (original <= 0 || !(factor > 0))
            throw new InvalidDataException("A YaRN GGUF needs a positive scaling factor and original_context_length.");
        return new RopeScaling
        {
            Type = RopeScalingType.Yarn,
            Factor = factor,
            OriginalContextLength = original,
            TruncateYarnCorrectionRange = true,
        };
    }

    /// <summary>Turns the tensors of a <c>sensenova_loop</c> GGUF into the layout <see cref="Qwen35Model"/> reads, in place in <paramref name="w"/>.
    /// The file stores each physical layer once, at its first execution slot, with three differences from a plain Qwen3.5 file:
    /// (1) <c>attn_qkv_gate</c> holds the qkv rows then the z rows and <c>ssm_beta_alpha</c> the beta rows then the alpha rows, fused;
    /// (2) llama.cpp's converter stores the value heads of the linear-attention layers in <i>tiled</i> order when there are more value than
    /// key heads (value head <c>j</c> belongs to key group <c>j % keyHeads</c>), where this engine, like transformers, expects them <i>grouped</i>
    /// (group <c>j / repeat</c>): qkv V rows, z rows, a and b rows, <c>ssm_a</c>, <c>ssm_dt.bias</c>, the V channels of the conv kernel and the
    /// columns of <c>ssm_out</c> are all permuted; (3) the routers are <c>hc_{attn,ffn}_fn.{weight,bias}</c>.
    /// Row permutations copy whole rows, so they are exact for quantized tensors; the <c>ssm_out</c> column permutation is not (it cuts through
    /// quantization blocks), so a quantized <c>ssm_out</c> is widened to F16 first. Later visits of a repeated layer then alias the first visit's tensors.</summary>
    /// <returns>The routers of each physical layer in the mHC range.</returns>
    private static Dictionary<int, MhcRouterWeights> PrepareSenseNovaWeights(Dictionary<string, Tensor> w, IReadOnlyDictionary<string, Tensor> source,
        SenseNovaLoopConfig loop, bool[] linear, int numKHeads, int numVHeads, int headKDim, int headVDim, int convKernel, List<Tensor> owned)
    {
        int repeat = numVHeads % numKHeads == 0 ? numVHeads / numKHeads : throw new InvalidDataException("Value heads must be a multiple of key heads.");
        int keyWidth = numKHeads * headKDim;   // rows of the q block, and of the k block, in the qkv projection and its conv channels
        Dictionary<int, MhcRouterWeights> routers = [];
        for (int slot = 0; slot < loop.Slots; slot++)
        {
            int physical = loop.Plan[slot];
            if (loop.FirstSlot(physical) != slot) continue;
            string p = $"blk.{slot}";
            if (linear[slot])
            {
                Tensor fused = Take(w, $"{p}.attn_qkv_gate.weight");
                int qkvRows = (int)fused.Shape[0] - numVHeads * headVDim;
                // The fused tensors are file-owned; row views taken from them stay valid after Take.
                Put(w, $"{p}.attn_qkv.weight", UntileRows(fused.SliceRows(0, qkvRows), 2 * keyWidth, numKHeads, repeat, headVDim, owned), source);
                Put(w, $"{p}.attn_gate.weight", UntileRows(fused.SliceRows(qkvRows, numVHeads * headVDim), 0, numKHeads, repeat, headVDim, owned), source);
                Tensor ba = Take(w, $"{p}.ssm_beta_alpha.weight");
                Put(w, $"{p}.ssm_beta.weight", UntileRows(ba.SliceRows(0, numVHeads), 0, numKHeads, repeat, 1, owned), source);
                Put(w, $"{p}.ssm_alpha.weight", UntileRows(ba.SliceRows(numVHeads, numVHeads), 0, numKHeads, repeat, 1, owned), source);
                Put(w, $"{p}.ssm_a", UntileRows(w[$"{p}.ssm_a"], 0, numKHeads, repeat, 1, owned), source);
                Put(w, $"{p}.ssm_dt.bias", UntileRows(w[$"{p}.ssm_dt.bias"], 0, numKHeads, repeat, 1, owned), source);
                // The conv kernel is one row of ConvKernel taps per channel; the channels after q and k are the V heads.
                Put(w, $"{p}.ssm_conv1d.weight", UntileRows(w[$"{p}.ssm_conv1d.weight"], 2 * keyWidth, numKHeads, repeat, headVDim, owned, rowElements: convKernel), source);
                Put(w, $"{p}.ssm_out.weight", UntileColumns(w[$"{p}.ssm_out.weight"], numKHeads, repeat, headVDim, owned), source);
            }
            if (physical >= loop.MhcStart && physical < loop.MhcEnd)
            {
                routers[physical] = new MhcRouterWeights(
                    TakeFloats(w, $"{p}.hc_attn_fn.weight"), TakeFloats(w, $"{p}.hc_attn_fn.bias"),
                    TakeFloats(w, $"{p}.hc_ffn_fn.weight"), TakeFloats(w, $"{p}.hc_ffn_fn.bias"));
            }
        }
        for (int slot = 0; slot < loop.Slots; slot++)
        {
            int first = loop.FirstSlot(loop.Plan[slot]);
            if (first == slot) continue;
            string firstPrefix = $"blk.{first}.", prefix = $"blk.{slot}.";
            foreach (string key in w.Keys.Where(k => k.StartsWith(firstPrefix, StringComparison.Ordinal)).ToList())
            {
                w[prefix + key[firstPrefix.Length..]] = w[key];
            }
        }
        // The MTP draft block (slots past the plan) is not run; leaving it in the dictionary would preload it.
        foreach (string key in w.Keys.Where(k => IsSlotBeyondPlan(k, loop.Slots)).ToList())
        {
            Drop(w, key, source);
        }
        return routers;
    }

    /// <summary>True for <c>blk.N.*</c> keys with N at or past <paramref name="slots"/>.</summary>
    private static bool IsSlotBeyondPlan(string key, int slots)
    {
        const string prefix = "blk.";
        if (!key.StartsWith(prefix, StringComparison.Ordinal)) return false;
        int dot = key.IndexOf('.', prefix.Length);
        return dot > prefix.Length && int.TryParse(key.AsSpan(prefix.Length, dot - prefix.Length), out int index) && index >= slots;
    }

    /// <summary>Replaces <paramref name="key"/>'s tensor; a superseded tensor the file does not own is disposed.</summary>
    private static void Put(Dictionary<string, Tensor> w, string key, Tensor value, IReadOnlyDictionary<string, Tensor> source)
    {
        if (w.TryGetValue(key, out Tensor? old) && !ReferenceEquals(old, value) && !OwnedByFile(old, source))
            old.Dispose();
        w[key] = value;
    }

    /// <summary>Removes <paramref name="key"/>; a tensor the file does not own is disposed.</summary>
    private static void Drop(Dictionary<string, Tensor> w, string key, IReadOnlyDictionary<string, Tensor> source)
    {
        if (!w.Remove(key, out Tensor? old)) return;
        if (!OwnedByFile(old, source)) old.Dispose();
    }

    private static bool OwnedByFile(Tensor t, IReadOnlyDictionary<string, Tensor> source)
    {
        foreach (Tensor f in source.Values)
            if (ReferenceEquals(f, t)) return true;
        return false;
    }

    private static Tensor Take(Dictionary<string, Tensor> w, string key)
    {
        if (!w.Remove(key, out Tensor? t)) throw new KeyNotFoundException($"The SenseNova-Loop GGUF has no tensor '{key}'.");
        return t;
    }

    private static float[] TakeFloats(Dictionary<string, Tensor> w, string key)
    {
        Tensor t = Take(w, key);
        if (t.DType != DType.F32) throw new InvalidDataException($"'{key}' is {t.DType.Name}; mHC routers must be F32.");
        return t.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>Source head (in the tiled file layout) of head <paramref name="grouped"/> in transformers' grouped order.</summary>
    internal static int TiledIndex(int grouped, int keyHeads, int repeat) => grouped % repeat * keyHeads + grouped / repeat;

    /// <summary>A copy of <paramref name="src"/> (rows of any dtype, quantized included) whose rows from <paramref name="firstRow"/> on are
    /// value heads of <paramref name="rowsPerHead"/> rows each, moved from tiled to grouped order; the rows before are copied unchanged.
    /// <paramref name="rowElements"/> is the row length when the tensor's shape does not say (a 1-D vector has one element per row).</summary>
    internal static Tensor UntileRows(Tensor src, int firstRow, int keyHeads, int repeat, int rowsPerHead, List<Tensor> owned, int rowElements = 0)
    {
        if (repeat == 1) return src;   // equal head counts: the permutation is the identity, so no copy
        long cols = rowElements > 0 ? rowElements : src.Shape.Rank == 2 ? src.Shape[1] : 1;
        long rows = src.ElementCount / cols;
        long rowBytes = src.DType.ComputeByteCount(cols);
        Tensor dst = new(src.Shape, src.DType);
        owned.Add(dst);
        byte* from = (byte*)src.DataPointer, to = (byte*)dst.DataPointer;
        Buffer.MemoryCopy(from, to, rows * rowBytes, firstRow * rowBytes);
        int heads = keyHeads * repeat;
        if (rows - firstRow != (long)heads * rowsPerHead)
        {
            throw new InvalidDataException($"{rows - firstRow} value rows do not divide into {heads} heads of {rowsPerHead}.");
        }
        for (int g = 0; g < heads; g++)
        {
            long srcRow = firstRow + (long)TiledIndex(g, keyHeads, repeat) * rowsPerHead, dstRow = firstRow + (long)g * rowsPerHead;
            Buffer.MemoryCopy(from + srcRow * rowBytes, to + dstRow * rowBytes, rowsPerHead * rowBytes, rowsPerHead * rowBytes);
        }
        return dst;
    }

    /// <summary>A copy of the <c>[out, in]</c> matrix <paramref name="src"/> whose input columns, grouped in heads of <paramref name="headCols"/>,
    /// are moved from tiled to grouped order. A quantized matrix is widened to F16 first (the permutation cuts through blocks).</summary>
    internal static Tensor UntileColumns(Tensor src, int keyHeads, int repeat, int headCols, List<Tensor> owned)
    {
        if (repeat == 1) return src;   // identity: keep the file's quantization, no widening
        Tensor source = src;
        if (src.DType.IsQuantized)
        {
            source = GgufDequantizer.Dequantize(src, DType.F32);
        }
        long rows = src.Shape[0], cols = src.Shape[1];
        int heads = keyHeads * repeat;
        if (cols != (long)heads * headCols)
        {
            throw new InvalidDataException($"{cols} columns do not divide into {heads} heads of {headCols}.");
        }
        long elementBytes = source.DType.ComputeByteCount(1);
        Tensor dst = new(source.Shape, source.DType);
        byte* from = (byte*)source.DataPointer, to = (byte*)dst.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            for (int g = 0; g < heads; g++)
            {
                long srcOffset = (r * cols + (long)TiledIndex(g, keyHeads, repeat) * headCols) * elementBytes;
                long dstOffset = (r * cols + (long)g * headCols) * elementBytes;
                Buffer.MemoryCopy(from + srcOffset, to + dstOffset, headCols * elementBytes, headCols * elementBytes);
            }
        }
        if (!src.DType.IsQuantized)
        {
            owned.Add(dst);
            return dst;
        }
        Tensor narrow = dst.CastTo(DType.F16);
        dst.Dispose();
        source.Dispose();
        owned.Add(narrow);
        return narrow;
    }
}
