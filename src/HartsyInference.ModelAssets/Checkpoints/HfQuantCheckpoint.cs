using System.Globalization;
using System.Text.RegularExpressions;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Nf4;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>A Hugging Face safetensors checkpoint read as logical <c>*.weight</c> tensors, whatever their storage: plain float,
/// block FP8 with an F32 <c>weight_scale_inv</c> grid, or symmetric 4-bit GPTQ (<c>qweight</c> / <c>scales</c> / <c>qzeros</c>).
/// Fused MoE expert stacks (<c>mlp.experts.gate_up_proj</c> <c>[E, 2I, H]</c> and <c>mlp.experts.down_proj</c> <c>[E, H, I]</c>)
/// read as per-expert <c>mlp.experts.{e}.{gate,up,down}_proj.weight</c>. <see cref="Produce"/> returns any weight in any dtype,
/// taking the exact route where one exists (a GPTQ weight to Q4_0, a float weight to its own dtype).</summary>
public sealed partial class HfQuantCheckpoint : IDisposable
{
    private const string Weight = ".weight";
    private readonly ShardedSafeTensorSet _set;
    private readonly Dictionary<string, HfQuantWeight> _weights = new(StringComparer.Ordinal);
    private int _disposed;

    private HfQuantCheckpoint(ShardedSafeTensorSet set, int gptqGroupSize, int fp8BlockSize)
    {
        _set = set;
        GptqGroupSize = gptqGroupSize;
        Fp8BlockSize = fp8BlockSize;
        Index();
    }

    /// <summary>Inputs per GPTQ scale group.</summary>
    public int GptqGroupSize { get; }

    /// <summary>Edge of an FP8 scale block.</summary>
    public int Fp8BlockSize { get; }

    /// <summary>Every logical weight, by name.</summary>
    public IReadOnlyDictionary<string, HfQuantWeight> Weights => _weights;

    /// <summary>Names the files the weights come from: each shard's name, size and last write time. A cache built from this
    /// checkpoint records it, so a replaced shard is noticed even when the new one has the same tensors.</summary>
    public string SourceIdentity => string.Join(';', _set.Shards.Select(static shard =>
    {
        FileInfo file = new(shard.Path);
        return string.Create(CultureInfo.InvariantCulture, $"{shard.FileName}:{file.Length}:{file.LastWriteTimeUtc.Ticks}");
    }));

    /// <summary>Opens <paramref name="directory"/>'s <c>model.safetensors.index.json</c> shards, or its single <c>model.safetensors</c>.</summary>
    public static HfQuantCheckpoint Open(string directory, int gptqGroupSize = 128, int fp8BlockSize = Fp8BlockF32Codec.DefaultBlock)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ShardedSafeTensorSet set = File.Exists(Path.Combine(directory, "model.safetensors.index.json"))
            ? ShardedSafeTensorSet.OpenIndex(directory)
            : ShardedSafeTensorSet.OpenFiles([Path.Combine(directory, "model.safetensors")]);
        try
        {
            return new HfQuantCheckpoint(set, gptqGroupSize, fp8BlockSize);
        }
        catch
        {
            set.Dispose();
            throw;
        }
    }

    /// <summary>Opens the given safetensors files as one checkpoint.</summary>
    public static HfQuantCheckpoint OpenFiles(IReadOnlyList<string> files, int gptqGroupSize = 128, int fp8BlockSize = Fp8BlockF32Codec.DefaultBlock)
    {
        ArgumentNullException.ThrowIfNull(files);
        ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles(files);
        try
        {
            return new HfQuantCheckpoint(set, gptqGroupSize, fp8BlockSize);
        }
        catch
        {
            set.Dispose();
            throw;
        }
    }

    /// <summary>Whether <paramref name="path"/> names one of this checkpoint's files, through a symbolic link in any component.</summary>
    public bool IsSourceFile(string path) => _set.Shards.Any(shard => FileSystemPathIdentity.SamePath(shard.Path, path));

    /// <summary>A plain weight as stored, as a fresh borrowed view of the mapped checkpoint (an expert of a fused stack is a row view).</summary>
    public Tensor Borrow(string name)
    {
        HfQuantWeight w = Get(name);
        if (w.Storage != HfWeightStorage.Plain)
        {
            throw new InvalidOperationException($"'{name}' is stored as {w.Storage}; it has no float view to borrow.");
        }
        if (w.StackedSource is null) return _set.GetTensor(w.SourceKey);
        Tensor stacked = _set.GetTensor(w.StackedSource);
        long rowsPerExpert = stacked.Shape[1];
        return stacked.Reshape(new TensorShape(stacked.Shape[0] * rowsPerExpert, stacked.Shape[2]))
            .SliceRows(w.StackedExpert * rowsPerExpert + w.StackedRowOffset, w.Shape[0]);
    }

    /// <summary>The weight as F32 <c>[out, in]</c>, a new tensor the caller owns.</summary>
    public Tensor DequantizeF32(string name)
    {
        HfQuantWeight w = Get(name);
        switch (w.Storage)
        {
            case HfWeightStorage.Fp8Block:
            {
                using Tensor q = _set.GetTensor(w.SourceKey);
                using Tensor scale = _set.GetTensor(w.SourceKey + "_scale_inv");
                return Fp8BlockF32Codec.Dequantize(q, scale, Fp8BlockSize, Fp8BlockSize);
            }
            case HfWeightStorage.GptqInt4:
            {
                (Tensor qweight, Tensor scales, Tensor? qzeros, Tensor? gIdx) = GptqParts(w);
                using (qweight) using (scales) using (qzeros) using (gIdx)
                {
                    return GptqInt4Codec.Dequantize(qweight, scales, qzeros, gIdx, GptqGroupSize);
                }
            }
            case HfWeightStorage.Nf4:
            {
                Tensor decoded = DecodeNf4(w);
                if (decoded.DType == DType.F32) return decoded;
                using (decoded) return decoded.CastTo(DType.F32);
            }
            default:
            {
                using Tensor view = Borrow(name);
                return view.CastTo(DType.F32);
            }
        }
    }

    private Tensor DecodeNf4(HfQuantWeight w)
    {
        Dictionary<string, Tensor> parts = new(StringComparer.Ordinal);
        try
        {
            foreach (string key in Nf4Keys(w.SourceKey)) parts[key] = _set.GetTensor(key);
            return Nf4CompanionFold.DecodeWeight(parts, w.SourceKey);
        }
        finally
        {
            foreach (Tensor part in parts.Values) part.Dispose();
        }
    }

    private IEnumerable<string> Nf4Keys(string module) =>
        new[] { module, module + ".absmax", module + ".quant_map", module + ".nested_absmax", module + ".nested_quant_map",
            module + Nf4CompanionFold.QuantStatePrefix + "nf4" }.Where(_set.Inventory.ContainsKey);

    /// <summary>The weight in <paramref name="target"/> dtype, a tensor the caller owns: a fresh view for a plain weight already in it,
    /// a bit-exact repack for a GPTQ weight to Q4_0, else a cast or quantization of the dequantized values. When
    /// <paramref name="error"/> is given, a quantization folds its round-trip error in.</summary>
    public Tensor Produce(string name, DType target, GgufQuantError? error = null)
    {
        HfQuantWeight w = Get(name);
        long rowLength = w.Shape[w.Shape.Rank - 1];
        if (target.IsQuantized && rowLength % target.BlockElementCount != 0)
        {
            // The quantizer blocks the flattened tensor, so a ragged row would silently share blocks with the next one.
            throw new ArgumentException(
                $"'{name}' has rows of {rowLength} values, not whole {target.Name} blocks of {target.BlockElementCount}.", nameof(target));
        }
        if (w.Storage == HfWeightStorage.Plain && w.StoredDType == target)
        {
            return Borrow(name);
        }
        if (w.Storage == HfWeightStorage.GptqInt4 && target == DType.Q4_0)
        {
            (Tensor qweight, Tensor scales, Tensor? qzeros, Tensor? gIdx) = GptqParts(w);
            using (qweight) using (scales) using (qzeros) using (gIdx)
            {
                return GptqInt4Codec.ToQ4_0(qweight, scales, qzeros, gIdx, GptqGroupSize);
            }
        }
        if (w.Storage == HfWeightStorage.Plain && !target.IsQuantized)
        {
            using Tensor view = Borrow(name);
            return view.CastTo(target);
        }
        Tensor f32 = DequantizeF32(name);
        if (target == DType.F32) return f32;
        try
        {
            if (!target.IsQuantized) return f32.CastTo(target);
            return error is null ? GgufQuantizer.Quantize(f32, target) : error.Quantize(f32, target);
        }
        finally
        {
            f32.Dispose();
        }
    }

    private HfQuantWeight Get(string name) =>
        _weights.TryGetValue(name, out HfQuantWeight? w) ? w : throw new KeyNotFoundException($"The checkpoint has no weight '{name}'.");

    private (Tensor QWeight, Tensor Scales, Tensor? QZeros, Tensor? GIdx) GptqParts(HfQuantWeight w)
    {
        string module = w.SourceKey;
        Tensor? qzeros = _set.Inventory.ContainsKey($"{module}.qzeros") ? _set.GetTensor($"{module}.qzeros") : null;
        Tensor? gIdx = _set.Inventory.ContainsKey($"{module}.g_idx") ? _set.GetTensor($"{module}.g_idx") : null;
        return (_set.GetTensor($"{module}.qweight"), _set.GetTensor($"{module}.scales"), qzeros, gIdx);
    }

    private HashSet<string> IndexNf4()
    {
        HashSet<string> consumed = new(StringComparer.Ordinal);
        foreach (string key in _set.Inventory.Keys)
        {
            int marker = key.IndexOf(Nf4CompanionFold.QuantStatePrefix, StringComparison.Ordinal);
            if (marker < 0) continue;
            string module = key[..marker];
            string quantType = key[(marker + Nf4CompanionFold.QuantStatePrefix.Length)..];
            if (!string.Equals(quantType, "nf4", StringComparison.Ordinal))
            {
                throw new NotSupportedException($"'{module}' is bitsandbytes '{quantType}', not nf4. Only NF4 4-bit checkpoints are read.");
            }
            if (!_set.Inventory.ContainsKey(module))
            {
                throw new InvalidDataException($"'{key}' has a quant state but no weight tensor '{module}'.");
            }
            RequireCompanion(module + ".absmax", module);
            RequireCompanion(module + ".quant_map", module);
            // A non-F32 absmax is double-quantized and decodes only through both nested companions.
            if (_set.Inventory[module + ".absmax"].DType != DType.F32)
            {
                RequireCompanion(module + ".nested_absmax", module);
                RequireCompanion(module + ".nested_quant_map", module);
            }
            TensorShape shape;
            using (Tensor state = _set.GetTensor(key))
            {
                shape = Nf4CompanionFold.ReadShape(state, key);
            }
            long packedBytes = _set.Inventory[module].Shape.ElementCount;
            long expectedBytes = (shape.ElementCount + 1) / 2;
            if (packedBytes != expectedBytes)
            {
                throw new InvalidDataException($"'{module}' holds {packedBytes} NF4 bytes; its quant state's shape {shape} needs {expectedBytes}.");
            }
            Add(new HfQuantWeight(module, shape, HfWeightStorage.Nf4, DType.U8, module));
            foreach (string part in Nf4Keys(module)) consumed.Add(part);
        }
        return consumed;
    }

    private void Index()
    {
        HashSet<string> nf4Parts = IndexNf4();
        foreach ((string key, TensorLocation location) in _set.Inventory)
        {
            if (nf4Parts.Contains(key) || key.EndsWith("_scale_inv", StringComparison.Ordinal) || key.EndsWith(".scales", StringComparison.Ordinal)
                || key.EndsWith(".qzeros", StringComparison.Ordinal) || key.EndsWith(".g_idx", StringComparison.Ordinal))
            {
                continue;
            }
            if (key.EndsWith(".qweight", StringComparison.Ordinal))
            {
                string module = key[..^".qweight".Length];
                RequireCompanion(module + ".scales", key);
                Add(new HfQuantWeight(module + Weight, new TensorShape(location.Shape[1], location.Shape[0] * 8), HfWeightStorage.GptqInt4, DType.I32, module));
                continue;
            }
            Match fused = FusedExperts().Match(key);
            if (fused.Success)
            {
                AddFusedExperts(fused, key, location);
                continue;
            }
            if (location.DType == DType.F8E4M3)
            {
                RequireCompanion(key + "_scale_inv", key);
                Add(new HfQuantWeight(key, location.Shape, HfWeightStorage.Fp8Block, location.DType, key));
                continue;
            }
            Add(new HfQuantWeight(key, location.Shape, HfWeightStorage.Plain, location.DType, key));
        }
    }

    private void AddFusedExperts(Match fused, string key, TensorLocation location)
    {
        if (location.Shape.Rank != 3 || location.DType == DType.F8E4M3)
        {
            throw new NotSupportedException($"The fused expert stack '{key}' is {location.Shape} {location.DType.Name}; only float [E, rows, cols] stacks are read.");
        }
        string prefix = fused.Groups["prefix"].Value;
        int experts = (int)location.Shape[0];
        long rows = location.Shape[1], cols = location.Shape[2];
        bool gateUp = fused.Groups["proj"].Value == "gate_up_proj";
        if (gateUp && rows % 2 != 0)
        {
            throw new InvalidDataException($"The fused gate/up stack '{key}' has an odd {rows} rows.");
        }
        for (int e = 0; e < experts; e++)
        {
            string expert = $"{prefix}.{e.ToString(CultureInfo.InvariantCulture)}";
            if (gateUp)
            {
                long half = rows / 2;
                Add(new HfQuantWeight($"{expert}.gate_proj.weight", new TensorShape(half, cols), HfWeightStorage.Plain, location.DType, key, key, e, 0));
                Add(new HfQuantWeight($"{expert}.up_proj.weight", new TensorShape(half, cols), HfWeightStorage.Plain, location.DType, key, key, e, half));
            }
            else
            {
                Add(new HfQuantWeight($"{expert}.down_proj.weight", new TensorShape(rows, cols), HfWeightStorage.Plain, location.DType, key, key, e, 0));
            }
        }
    }

    private void Add(HfQuantWeight weight)
    {
        if (!_weights.TryAdd(weight.Name, weight))
        {
            throw new InvalidDataException($"The checkpoint stores '{weight.Name}' twice.");
        }
    }

    private void RequireCompanion(string companion, string key)
    {
        if (!_set.Inventory.ContainsKey(companion))
        {
            throw new InvalidDataException($"'{key}' has no '{companion}' beside it.");
        }
    }

    [GeneratedRegex(@"^(?<prefix>.+\.mlp\.experts)\.(?<proj>gate_up_proj|down_proj)$", RegexOptions.CultureInvariant)]
    private static partial Regex FusedExperts();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _set.Dispose();
    }
}
