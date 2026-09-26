using System.Numerics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>Bakes adapters into checkpoint weights, producing a standalone merged file: the offline counterpart of
/// <see cref="LoraStack"/>. Adapter keys are grouped through <see cref="LoraRoleSuffix"/>, so every decomposition the
/// engine reads (LoRA, DoRA, LoHa, LoKr, full-weight diffs) bakes the same way, whatever the file's naming.
/// <para>A low-rank pair is multiplied on the host in float32 with <c>k</c> ascending and fused multiply-add, the order
/// single-threaded PyTorch CPU matmul accumulates in, then scaled once and added once. The merged weight therefore
/// matches torch's <c>W + (B @ A) * scale</c> bit for bit when torch runs on one thread; threaded torch splits the sum
/// and differs in the last bit of about 1% of values.</para></summary>
public static class LoraBaker
{
    /// <summary>One adapter module: its key root, what it adds, and the file keys it was built from.</summary>
    public sealed class Patch
    {
        /// <summary>The adapter key with its role suffix removed, e.g. <c>adapter.layers.0.attn.key_proj</c>.</summary>
        public required string Root { get; init; }

        /// <summary>The low-rank (or LoHa/LoKr) delta for the module's <c>.weight</c>, or null for a diff-only module.</summary>
        public LoraDelta? Delta { get; init; }

        /// <summary>A full-weight <c>.diff</c> for the module's <c>.weight</c>.</summary>
        public Tensor? Diff { get; init; }

        /// <summary>A full-weight <c>.diff_b</c> for the module's <c>.bias</c>.</summary>
        public Tensor? BiasDiff { get; init; }

        /// <summary>Every adapter key this patch consumed.</summary>
        public required IReadOnlyList<string> SourceKeys { get; init; }
    }

    /// <summary>How a bake scales each delta.</summary>
    public sealed class Options
    {
        /// <summary>User strength, multiplied into the scale once (1 = as trained).</summary>
        public float Strength { get; init; } = 1.0f;

        /// <summary>Alpha for every module, overriding any <c>.alpha</c> tensor. PEFT keeps it in
        /// <c>adapter_config.json</c> rather than the weights file, so it has to come from outside.</summary>
        public float? Alpha { get; init; }

        /// <summary>Rank-stabilized scaling, <c>alpha / sqrt(rank)</c> instead of <c>alpha / rank</c> (PEFT <c>use_rslora</c>).</summary>
        public bool RsLora { get; init; }

        /// <summary>Computes LoHa and LoKr deltas; a plain LoRA or diff never touches it. Null refuses those decompositions.</summary>
        public IBackend? Backend { get; init; }
    }

    /// <summary>Groups adapter tensors into patches by root. Keys without a LoRA role suffix are returned in
    /// <paramref name="unrecognized"/>; a module missing half its pair throws, naming it.</summary>
    public static List<Patch> Group(IEnumerable<KeyValuePair<string, Tensor>> adapter, List<string>? unrecognized = null)
    {
        Dictionary<string, Pending> groups = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in adapter)
        {
            if (!LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role))
            {
                unrecognized?.Add(key);
                continue;
            }
            if (!groups.TryGetValue(root, out Pending? group))
                groups[root] = group = new Pending { Buffer = new LoraGroupBuffer { Target = LoraTarget.Transformer, FirstSourceKey = key } };
            group.Keys.Add(key);
            if (role == LoraRole.Diff)
                group.Diff = tensor;
            else if (role == LoraRole.BiasDiff)
                group.BiasDiff = tensor;
            else
                group.Buffer.Assign(role, tensor);
        }
        List<Patch> patches = new(groups.Count);
        foreach ((string root, Pending group) in groups)
        {
            LoraDelta? delta = group.Buffer.BuildDelta();
            bool hasMatrices = group.Keys.Any(k => LoraRoleSuffix.TryStrip(k, out _, out LoraRole r) && LoraRoleSuffix.IsDecompositionMatrix(r));
            if (delta is null && hasMatrices)
                throw new InvalidDataException($"Adapter module '{root}' is incomplete: it has only {string.Join(", ", group.Keys)}.");
            if (delta is null && group.Diff is null && group.BiasDiff is null)
                throw new InvalidDataException($"Adapter module '{root}' has no delta, only {string.Join(", ", group.Keys)}.");
            patches.Add(new Patch { Root = root, Delta = delta, Diff = group.Diff, BiasDiff = group.BiasDiff, SourceKeys = group.Keys });
        }
        return patches;
    }

    private sealed class Pending
    {
        public required LoraGroupBuffer Buffer { get; init; }
        public List<string> Keys { get; } = [];
        public Tensor? Diff { get; set; }
        public Tensor? BiasDiff { get; set; }
    }

    /// <summary>Resolves adapter roots to modules of <paramref name="weightKeys"/> without a mapping. Adapter files name
    /// modules as the model does but under their own wrapper (<c>base_model.model.</c>, <c>lora_unet_</c>) while the
    /// checkpoint may carry another (<c>model.diffusion_model.</c>), so one prefix pair is inferred from all
    /// <paramref name="roots"/> together (the pair most of them share) and then applied strictly: a root that does not
    /// fit it resolves to null rather than to whichever weight happens to end the same way. Kohya's underscore
    /// spelling is matched against the weights with their dots written as underscores.</summary>
    /// <returns>A map from root to module name without <c>.weight</c>, or null.</returns>
    public static Func<string, string?> AutoTargets(IEnumerable<string> weightKeys, IEnumerable<string> roots)
    {
        List<string> modules = [.. weightKeys.Where(k => k.EndsWith(".weight", StringComparison.Ordinal)).Select(k => k[..^".weight".Length])];
        List<string> rootList = [.. roots];
        bool kohya = rootList.Count > 0 && rootList.All(r => !r.Contains('.', StringComparison.Ordinal));
        char separator = kohya ? '_' : '.';
        Dictionary<string, string> spelled = new(StringComparer.Ordinal);
        HashSet<string> clashes = new(StringComparer.Ordinal);
        foreach (string module in modules)
        {
            string key = kohya ? module.Replace('.', '_') : module;
            if (!spelled.TryAdd(key, module))
                clashes.Add(key);
        }
        Dictionary<string, List<string>> bySuffix = new(StringComparer.Ordinal);
        foreach (string key in spelled.Keys)
        {
            for (int i = 0; i >= 0; i = key.IndexOf(separator, i) is int next && next >= 0 ? next + 1 : -1)
            {
                string suffix = key[i..];
                if (!bySuffix.TryGetValue(suffix, out List<string>? list))
                    bySuffix[suffix] = list = [];
                list.Add(key);
            }
        }
        // Each root votes for the heads that its longest shared tail implies.
        Dictionary<(string RootHead, string ModuleHead), int> votes = [];
        foreach (string root in rootList)
        {
            for (int i = 0; i >= 0; i = root.IndexOf(separator, i) is int next && next >= 0 ? next + 1 : -1)
            {
                if (!bySuffix.TryGetValue(root[i..], out List<string>? hits))
                    continue;
                foreach (string hit in hits)
                {
                    (string, string) heads = (root[..i], hit[..^(root.Length - i)]);
                    votes[heads] = votes.GetValueOrDefault(heads) + 1;
                }
                break;
            }
        }
        if (votes.Count == 0)
            return _ => null;
        int best = votes.Values.Max();
        List<(string RootHead, string ModuleHead)> winners = [.. votes.Where(v => v.Value == best).Select(v => v.Key)];
        if (winners.Count > 1)
        {
            throw new InvalidDataException($"The adapter's module names fit the weights {winners.Count} ways ("
                + string.Join(", ", winners.Take(3).Select(w => $"'{w.RootHead}' -> '{w.ModuleHead}'")) + "); map it explicitly.");
        }
        (string rootHead, string moduleHead) = winners[0];
        return root =>
        {
            if (!root.StartsWith(rootHead, StringComparison.Ordinal))
                return null;
            string key = moduleHead + root[rootHead.Length..];
            if (clashes.Contains(key))
                throw new InvalidDataException($"Adapter module '{root}' matches several weights spelled '{key}'; map it explicitly.");
            return spelled.GetValueOrDefault(key);
        };
    }

    /// <summary>Bakes <paramref name="patches"/> into <paramref name="weights"/>, replacing each target with a new
    /// tensor of the same dtype (F32, BF16 or F16; BF16/F16 merge in F32 and round to nearest even). New tensors are
    /// added to <paramref name="owned"/>. Throws when a patch has no target or its shape does not fit.</summary>
    /// <param name="target">Maps a patch root to the module it modifies, without <c>.weight</c>; null means unresolved.</param>
    /// <returns>The number of weight and bias tensors changed.</returns>
    public static int Apply(IDictionary<string, Tensor> weights, IReadOnlyList<Patch> patches, Func<string, string?> target, Options options,
        List<Tensor> owned)
    {
        List<string> unresolved = [];
        int changed = 0;
        foreach (Patch patch in patches)
        {
            if (target(patch.Root) is not string module)
            {
                unresolved.Add(patch.Root);
                continue;
            }
            if (patch.Delta is not null || patch.Diff is not null)
            {
                string key = module + ".weight";
                Tensor weight = weights.TryGetValue(key, out Tensor? w) ? w : throw new InvalidDataException($"Adapter '{patch.Root}' targets '{key}', which does not exist.");
                weights[key] = Bake(key, weight, patch, options, owned);
                changed++;
            }
            if (patch.BiasDiff is not null)
            {
                string key = module + ".bias";
                Tensor bias = weights.TryGetValue(key, out Tensor? b) ? b : throw new InvalidDataException($"Adapter '{patch.Root}' targets '{key}', which does not exist.");
                weights[key] = AddDiff(key, bias, patch.BiasDiff, options.Strength, owned);
                changed++;
            }
        }
        if (unresolved.Count > 0)
            throw new InvalidDataException($"{unresolved.Count} adapter module(s) match no weight, e.g. {string.Join(", ", unresolved.Take(3))}. Check the adapter was trained for this model.");
        return changed;
    }

    /// <summary>The <c>scale</c> a plain low-rank pair is multiplied by: <c>strength · alpha / rank</c>, or
    /// <c>/ sqrt(rank)</c> under rsLoRA, evaluated in double and rounded once, as PEFT's Python float is.</summary>
    public static float ScaleFor(float alpha, int rank, Options options) =>
        (float)(options.Strength * (double)alpha / (options.RsLora ? Math.Sqrt(rank) : rank));

    private static Tensor Bake(string key, Tensor weight, Patch patch, Options options, List<Tensor> owned)
    {
        Tensor accum = ToF32(key, weight);
        try
        {
            if (patch.Diff is not null)
                AddScaled(accum, patch.Diff, options.Strength, key);
            if (patch.Delta is LoraDelta delta)
            {
                if (!delta.MatchesShape(weight))
                    throw new InvalidDataException($"Adapter '{patch.Root}' makes a [{delta.OutFeatures}, {delta.InFeatures}] delta; '{key}' is {weight.Shape}.");
                BakeDelta(accum, delta, options, key);
            }
            Tensor result = weight.DType == DType.F32 ? accum : FromF32(accum, weight.DType);
            if (!ReferenceEquals(result, accum))
                accum.Dispose();
            owned.Add(result);
            return result;
        }
        catch
        {
            accum.Dispose();
            throw;
        }
    }

    private static void BakeDelta(Tensor accum, LoraDelta delta, Options options, string key)
    {
        float alpha;
        int rank;
        Tensor product;
        if (delta is StandardLoraDelta standard)
        {
            alpha = options.Alpha ?? standard.Alpha;
            rank = standard.Rank;
            product = MatMulFma(standard.Up, standard.Down);
        }
        else
        {
            IBackend backend = options.Backend
                ?? throw new NotSupportedException($"'{key}' has a {delta.Variant} adapter, which needs a backend to compute; pass one in Options.Backend.");
            if (options.Alpha is not null || options.RsLora)
                throw new NotSupportedException($"'{key}' has a {delta.Variant} adapter; an alpha override or rsLoRA applies to plain LoRA only.");
            alpha = delta.Scale;
            rank = 1;
            product = delta.ComputeF32(backend);
        }
        try
        {
            if (delta.DoraScale is not null)
            {
                if (accum.Shape.Rank != 2)
                    throw new NotSupportedException($"'{key}' has a DoRA adapter on a {accum.Shape.Rank}-D weight; DoRA is defined for 2-D weights only.");
                LoraDoraDecompose.Apply(accum, product, delta.DoraScale, (float)(alpha / (options.RsLora ? Math.Sqrt(rank) : rank)), options.Strength);
                return;
            }
            float scale = ScaleFor(alpha, rank, options);
            Span<float> w = accum.AsSpan<float>();
            ReadOnlySpan<float> d = product.AsReadOnlySpan<float>();
            for (int i = 0; i < w.Length; i++)
            {
                float step = d[i] * scale;
                w[i] += step;
            }
        }
        finally
        {
            product.Dispose();
        }
    }

    /// <summary><c>up @ down</c> in F32, <c>k</c> ascending with fused multiply-add per element — single-threaded PyTorch's CPU order.
    /// Conv adapters fold their trailing axes into the columns first.</summary>
    public static unsafe Tensor MatMulFma(Tensor up, Tensor down)
    {
        long rows = up.Shape[0];
        int rank = (int)(up.Shape.ElementCount / rows);
        long columns = down.Shape.ElementCount / down.Shape[0];
        if (down.Shape[0] != rank)
            throw new InvalidDataException($"LoRA up {up.Shape} and down {down.Shape} disagree on the rank.");
        using Tensor b = ToF32("up", up);
        using Tensor a = ToF32("down", down);
        Tensor result = new(new TensorShape(rows, columns), DType.F32);
        float* bp = (float*)b.DataPointer, ap = (float*)a.DataPointer, cp = (float*)result.DataPointer;
        int width = Vector<float>.Count;
        Parallel.For(0, rows, row =>
        {
            float* c = cp + row * columns;
            new Span<float>(c, (int)columns).Clear();
            for (int k = 0; k < rank; k++)
            {
                float bk = bp[row * rank + k];
                float* ak = ap + (long)k * columns;
                Vector<float> bv = new(bk);
                long i = 0;
                for (; i + width <= columns; i += width)
                {
                    Vector<float> acc = Vector.Load(c + i);
                    Vector.Store(Vector.FusedMultiplyAdd(bv, Vector.Load(ak + i), acc), c + i);
                }
                for (; i < columns; i++)
                    c[i] = MathF.FusedMultiplyAdd(bk, ak[i], c[i]);
            }
        });
        return result;
    }

    private static Tensor AddDiff(string key, Tensor weight, Tensor diff, float strength, List<Tensor> owned)
    {
        Tensor accum = ToF32(key, weight);
        try
        {
            AddScaled(accum, diff, strength, key);
            Tensor result = weight.DType == DType.F32 ? accum : FromF32(accum, weight.DType);
            if (!ReferenceEquals(result, accum))
                accum.Dispose();
            owned.Add(result);
            return result;
        }
        catch
        {
            accum.Dispose();
            throw;
        }
    }

    private static void AddScaled(Tensor accum, Tensor diff, float strength, string key)
    {
        if (diff.Shape.ElementCount != accum.Shape.ElementCount)
            throw new InvalidDataException($"Diff {diff.Shape} does not fit '{key}' {accum.Shape}.");
        using Tensor d32 = ToF32(key, diff);
        Span<float> w = accum.AsSpan<float>();
        ReadOnlySpan<float> d = d32.AsReadOnlySpan<float>();
        for (int i = 0; i < w.Length; i++)
        {
            float step = d[i] * strength;
            w[i] += step;
        }
    }

    // Owned F32 copy; the merger's converter keeps BF16/F16 exact on the way up.
    private static unsafe Tensor ToF32(string key, Tensor tensor)
    {
        if (!SafeTensorsMerger.IsCastable(tensor.DType))
            throw new NotSupportedException($"'{key}' is {tensor.DType.Name}; adapters bake into F32, BF16 or F16 weights only. Dequantize it first.");
        Tensor copy = new(tensor.Shape, DType.F32);
        long count = tensor.Shape.ElementCount;
        byte* src = (byte*)tensor.DataPointer, dst = (byte*)copy.DataPointer;
        int inSize = tensor.DType.SizeInBytes;
        for (long done = 0; done < count; done += int.MaxValue / 4)
        {
            int n = (int)Math.Min(int.MaxValue / 4, count - done);
            SafeTensorsMerger.Convert(src + done * inSize, tensor.DType, dst + done * 4, DType.F32, n);
        }
        return copy;
    }

    private static unsafe Tensor FromF32(Tensor accum, DType dtype)
    {
        Tensor result = new(accum.Shape, dtype);
        long count = accum.Shape.ElementCount;
        byte* src = (byte*)accum.DataPointer, dst = (byte*)result.DataPointer;
        for (long done = 0; done < count; done += int.MaxValue / 4)
        {
            int n = (int)Math.Min(int.MaxValue / 4, count - done);
            SafeTensorsMerger.Convert(src + done * 4, DType.F32, dst + done * dtype.SizeInBytes, dtype, n);
        }
        return result;
    }
}
