using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Converts official Diffusers PDD targets into Hartsy's fused MiniMax-H3 weight names without skipped tensors.</summary>
public static unsafe class MiniMaxH3PddKeyConverter
{
    private const int PublishedBlocks = 50;
    private const int PublishedRefinerBlocks = 2;
    private const string DownSuffix = ".lora_down";
    private const string DownWeightSuffix = ".lora_down.weight";
    private const string UpSuffix = ".lora_up";
    private const string UpWeightSuffix = ".lora_up.weight";
    private const string ASuffix = ".lora_A.weight";
    private const string BSuffix = ".lora_B.weight";
    private const string ADefaultSuffix = ".lora_A.default.weight";
    private const string BDefaultSuffix = ".lora_B.default.weight";
    private const string AlphaSuffix = ".alpha";

    /// <summary>Strictly converts every trunk/refiner target and rejects any unrecognized source tensor.</summary>
    public static MiniMaxH3PddTrunkConversion Convert(IReadOnlyDictionary<string, Tensor> tensors,
        IReadOnlySet<string> headKeys, float globalAlpha, bool requireMainAdaln = true)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(headKeys);
        if (!(globalAlpha > 0.0f) || !float.IsFinite(globalAlpha))
            throw new ArgumentOutOfRangeException(nameof(globalAlpha), "PDD LoRA alpha must be finite and positive.");

        bool original = tensors.Keys.Any(key => key.StartsWith("transformer_blocks.", StringComparison.Ordinal)
            || key.StartsWith("token_refiner.refiner_blocks.", StringComparison.Ordinal));
        return original ? ConvertOfficial(tensors, headKeys, globalAlpha, requireMainAdaln)
            : ConvertCanonical(tensors, headKeys, globalAlpha);
    }

    private static MiniMaxH3PddTrunkConversion ConvertOfficial(IReadOnlyDictionary<string, Tensor> tensors,
        IReadOnlySet<string> headKeys, float alpha, bool requireMainAdaln)
    {
        HashSet<string> consumed = new(headKeys, StringComparer.Ordinal);
        List<LoraLayer> layers = new(PublishedBlocks * 5 + PublishedRefinerBlocks * 4);
        List<Tensor> owned = [];
        try
        {
            for (int i = 0; i < PublishedBlocks; i++)
            {
                string source = $"transformer_blocks.{i}";
                string target = $"blocks.{i}";
                AddQkv(tensors, consumed, owned, layers, source, target, alpha);
                AddPlain(tensors, consumed, layers, $"{source}.attn.to_out.0", $"{target}.attn.out_proj", alpha);
                AddPlain(tensors, consumed, layers, $"{source}.ff.net.0.proj", $"{target}.mlp.fc1", alpha,
                    swapUpHalves: true, owned);
                AddPlain(tensors, consumed, layers, $"{source}.ff.net.2", $"{target}.mlp.fc2", alpha);
                if (requireMainAdaln)
                {
                    AddPlain(tensors, consumed, layers, $"{source}.adaln_proj.linear",
                        $"{target}.adaln_proj.linear", alpha);
                }
            }
            for (int i = 0; i < PublishedRefinerBlocks; i++)
            {
                string source = $"token_refiner.refiner_blocks.{i}";
                string target = $"token_refiner.blocks.{i}";
                AddQkv(tensors, consumed, owned, layers, source, target, alpha);
                AddPlain(tensors, consumed, layers, $"{source}.attn.to_out.0", $"{target}.attn.out_proj", alpha);
                AddPlain(tensors, consumed, layers, $"{source}.ff.net.0.proj", $"{target}.mlp.fc1", alpha,
                    swapUpHalves: true, owned);
                AddPlain(tensors, consumed, layers, $"{source}.ff.net.2", $"{target}.mlp.fc2", alpha);
            }
            RejectLeftovers(tensors, consumed);
            return new MiniMaxH3PddTrunkConversion(layers, [], owned);
        }
        catch
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
            throw;
        }
    }

    private static MiniMaxH3PddTrunkConversion ConvertCanonical(IReadOnlyDictionary<string, Tensor> tensors,
        IReadOnlySet<string> headKeys, float globalAlpha)
    {
        Dictionary<string, CanonicalGroup> groups = new(StringComparer.Ordinal);
        Dictionary<string, Tensor> diffs = new(StringComparer.Ordinal);
        Dictionary<string, Tensor> biasDiffs = new(StringComparer.Ordinal);
        HashSet<string> consumed = new(headKeys, StringComparer.Ordinal);

        foreach ((string key, Tensor tensor) in tensors)
        {
            if (headKeys.Contains(key)) continue;
            string root;
            CanonicalGroup group;
            if (key.EndsWith(ASuffix, StringComparison.Ordinal) || key.EndsWith(DownWeightSuffix, StringComparison.Ordinal))
            {
                string suffix = key.EndsWith(ASuffix, StringComparison.Ordinal) ? ASuffix : DownWeightSuffix;
                root = NormalizeCanonicalRoot(key[..^suffix.Length]);
                group = GetGroup(groups, root);
                if (group.Down is not null) throw Duplicate(key, root, "down");
                group.Down = tensor;
                consumed.Add(key);
            }
            else if (key.EndsWith(BSuffix, StringComparison.Ordinal) || key.EndsWith(UpWeightSuffix, StringComparison.Ordinal))
            {
                string suffix = key.EndsWith(BSuffix, StringComparison.Ordinal) ? BSuffix : UpWeightSuffix;
                root = NormalizeCanonicalRoot(key[..^suffix.Length]);
                group = GetGroup(groups, root);
                if (group.Up is not null) throw Duplicate(key, root, "up");
                group.Up = tensor;
                consumed.Add(key);
            }
            else if (key.EndsWith(AlphaSuffix, StringComparison.Ordinal))
            {
                root = NormalizeCanonicalRoot(key[..^AlphaSuffix.Length]);
                group = GetGroup(groups, root);
                group.Alpha = KohyaSdMapper.ReadScalar(tensor);
                consumed.Add(key);
            }
            else if (key.EndsWith(".diff_b", StringComparison.Ordinal))
            {
                root = NormalizeCanonicalRoot(key[..^".diff_b".Length]);
                if (!biasDiffs.TryAdd(root, tensor)) throw Duplicate(key, root, "bias diff");
                consumed.Add(key);
            }
            else if (key.EndsWith(".diff", StringComparison.Ordinal))
            {
                root = NormalizeCanonicalRoot(key[..^".diff".Length]);
                if (!diffs.TryAdd(root, tensor)) throw Duplicate(key, root, "weight diff");
                consumed.Add(key);
            }
        }

        List<LoraLayer> layers = new(groups.Count);
        foreach ((string root, CanonicalGroup group) in groups)
        {
            if (group.Down is null || group.Up is null)
                throw new HartsyInferenceException($"PDD target '{root}' is missing its down or up matrix.");
            ValidatePair(root, group.Down, group.Up);
            int rank = (int)group.Down.Shape[0];
            layers.Add(new LoraLayer
            {
                TargetKey = root + ".weight",
                Target = LoraTarget.Transformer,
                LoraDown = group.Down,
                LoraUp = group.Up,
                Alpha = group.Alpha ?? globalAlpha,
                Rank = rank,
                Variant = LoraVariant.StandardLora,
            });
        }

        List<LoraFullWeightDiff> fullDiffs = [];
        foreach ((string root, Tensor diff) in diffs)
        {
            if (!root.EndsWith(".adaln_proj.linear", StringComparison.Ordinal))
                throw new HartsyInferenceException($"PDD full-weight diff '{root}' is not an AdaLN rebase target.");
            if (!biasDiffs.TryGetValue(root, out Tensor? bias))
                throw new HartsyInferenceException($"PDD AdaLN rebase '{root}' is missing mandatory DC-bias diff_b.");
            fullDiffs.Add(new LoraFullWeightDiff
            {
                TargetKey = root + ".weight",
                Target = LoraTarget.Transformer,
                Diff = diff,
                IsBias = false,
            });
            fullDiffs.Add(new LoraFullWeightDiff
            {
                TargetKey = root + ".bias",
                Target = LoraTarget.Transformer,
                Diff = bias,
                IsBias = true,
            });
        }
        foreach (string root in biasDiffs.Keys)
        {
            if (!diffs.ContainsKey(root))
                throw new HartsyInferenceException($"PDD AdaLN DC-bias diff '{root}' has no matching weight diff.");
        }

        ValidateCompleteTargets(layers, fullDiffs);
        RejectLeftovers(tensors, consumed);
        return new MiniMaxH3PddTrunkConversion(layers, fullDiffs, []);
    }

    private static void AddQkv(IReadOnlyDictionary<string, Tensor> tensors, HashSet<string> consumed,
        List<Tensor> owned, List<LoraLayer> layers, string source, string target, float alpha)
    {
        Tensor[] down = new Tensor[3];
        Tensor[] up = new Tensor[3];
        string[] parts = ["q", "k", "v"];
        for (int i = 0; i < parts.Length; i++)
        {
            down[i] = TakeRole(tensors, consumed, $"{source}.attn.to_{parts[i]}", down: true);
            up[i] = TakeRole(tensors, consumed, $"{source}.attn.to_{parts[i]}", down: false);
            ValidatePair($"{source}.attn.to_{parts[i]}", down[i], up[i]);
        }
        int rank = (int)down[0].Shape[0];
        int input = (int)down[0].Shape[1];
        int output = (int)up[0].Shape[0];
        for (int i = 1; i < 3; i++)
        {
            if (down[i].DType != down[0].DType || up[i].DType != up[0].DType
                || down[i].Shape[0] != rank || down[i].Shape[1] != input
                || up[i].Shape[0] != output || up[i].Shape[1] != rank)
            {
                throw new HartsyInferenceException($"PDD Q/K/V ranks and projection dimensions disagree at '{source}'.");
            }
        }
        Tensor fusedDown = ConcatRows(down, rank, input);
        Tensor fusedUp = BlockDiagonal(up, output, rank);
        owned.Add(fusedDown);
        owned.Add(fusedUp);
        layers.Add(new LoraLayer
        {
            TargetKey = target + ".attn.qkv_proj.weight",
            Target = LoraTarget.Transformer,
            LoraDown = fusedDown,
            LoraUp = fusedUp,
            Alpha = alpha * 3.0f,
            Rank = rank * 3,
            Variant = LoraVariant.StandardLora,
        });
    }

    private static void AddPlain(IReadOnlyDictionary<string, Tensor> tensors, HashSet<string> consumed,
        List<LoraLayer> layers, string source, string target, float alpha, bool swapUpHalves = false,
        List<Tensor>? owned = null)
    {
        Tensor down = TakeRole(tensors, consumed, source, down: true);
        Tensor up = TakeRole(tensors, consumed, source, down: false);
        ValidatePair(source, down, up);
        if (swapUpHalves)
        {
            if (owned is null) throw new ArgumentNullException(nameof(owned));
            up = SwapRowHalves(up);
            owned.Add(up);
        }
        layers.Add(new LoraLayer
        {
            TargetKey = target + ".weight",
            Target = LoraTarget.Transformer,
            LoraDown = down,
            LoraUp = up,
            Alpha = alpha,
            Rank = (int)down.Shape[0],
            Variant = LoraVariant.StandardLora,
        });
    }

    private static Tensor TakeRole(IReadOnlyDictionary<string, Tensor> tensors, HashSet<string> consumed,
        string root, bool down)
    {
        string[] suffixes = down
            ? [DownSuffix, DownWeightSuffix, ASuffix, ADefaultSuffix]
            : [UpSuffix, UpWeightSuffix, BSuffix, BDefaultSuffix];
        Tensor? found = null;
        string? foundKey = null;
        foreach (string suffix in suffixes)
        {
            string key = root + suffix;
            if (!tensors.TryGetValue(key, out Tensor? tensor)) continue;
            if (found is not null)
                throw new HartsyInferenceException($"PDD target '{root}' carries duplicate role spellings.");
            found = tensor;
            foundKey = key;
        }
        if (found is null)
            throw new HartsyInferenceException($"PDD target '{root}' is missing its {(down ? "down" : "up")} matrix.");
        consumed.Add(foundKey!);
        return found;
    }

    private static void ValidatePair(string root, Tensor down, Tensor up)
    {
        if (down.Shape.Rank != 2 || up.Shape.Rank != 2)
            throw new HartsyInferenceException($"PDD LoRA target '{root}' must use rank-two matrices.");
        if (down.Shape[0] != up.Shape[1])
        {
            throw new HartsyInferenceException(
                $"PDD LoRA target '{root}' rank mismatch: down {down.Shape}, up {up.Shape}.");
        }
        if (down.DType != up.DType || (down.DType != DType.F32 && down.DType != DType.F16
            && down.DType != DType.BF16))
        {
            throw new HartsyInferenceException(
                $"PDD LoRA target '{root}' requires matching F32/F16/BF16 matrices; got {down.DType}/{up.DType}.");
        }
    }

    private static Tensor ConcatRows(Tensor[] inputs, int rows, int columns)
    {
        Tensor result = new Tensor(new TensorShape(rows * inputs.Length, columns), inputs[0].DType);
        long bytes = inputs[0].DType.ComputeByteCount((long)rows * columns);
        byte* destination = (byte*)result.DataPointer;
        for (int i = 0; i < inputs.Length; i++)
        {
            Buffer.MemoryCopy((void*)inputs[i].DataPointer, destination + i * bytes, bytes, bytes);
        }
        return result;
    }

    private static Tensor BlockDiagonal(Tensor[] inputs, int rows, int columns)
    {
        DType dtype = inputs[0].DType;
        Tensor result = new Tensor(new TensorShape(rows * inputs.Length, columns * inputs.Length), dtype);
        byte* destination = (byte*)result.DataPointer;
        long sourceRowBytes = dtype.ComputeByteCount(columns);
        long destinationRowBytes = dtype.ComputeByteCount(columns * inputs.Length);
        for (int block = 0; block < inputs.Length; block++)
        {
            byte* source = (byte*)inputs[block].DataPointer;
            for (int row = 0; row < rows; row++)
            {
                byte* target = destination + (block * rows + row) * destinationRowBytes + block * sourceRowBytes;
                Buffer.MemoryCopy(source + row * sourceRowBytes, target, sourceRowBytes, sourceRowBytes);
            }
        }
        return result;
    }

    private static Tensor SwapRowHalves(Tensor source)
    {
        if (source.Shape[0] % 2 != 0)
            throw new HartsyInferenceException($"PDD SwiGLU up matrix has odd output rows: {source.Shape}.");
        int half = (int)source.Shape[0] / 2;
        long columns = source.Shape[1];
        Tensor result = new Tensor(source.Shape, source.DType);
        long halfBytes = source.DType.ComputeByteCount((long)half * columns);
        byte* input = (byte*)source.DataPointer;
        byte* output = (byte*)result.DataPointer;
        Buffer.MemoryCopy(input + halfBytes, output, halfBytes, halfBytes);
        Buffer.MemoryCopy(input, output + halfBytes, halfBytes, halfBytes);
        return result;
    }

    private static CanonicalGroup GetGroup(Dictionary<string, CanonicalGroup> groups, string root)
    {
        if (!groups.TryGetValue(root, out CanonicalGroup? group))
        {
            group = new CanonicalGroup();
            groups[root] = group;
        }
        return group;
    }

    private static string NormalizeCanonicalRoot(string root)
    {
        string result = root;
        string[] prefixes = ["model.diffusion_model.", "diffusion_model.", "transformer."];
        foreach (string prefix in prefixes)
        {
            if (result.StartsWith(prefix, StringComparison.Ordinal))
            {
                result = result[prefix.Length..];
                break;
            }
        }
        result = result.Replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", StringComparison.Ordinal);
        return result;
    }

    private static void ValidateCompleteTargets(IReadOnlyList<LoraLayer> layers,
        IReadOnlyList<LoraFullWeightDiff> fullDiffs)
    {
        HashSet<string> actual = layers.Select(layer => layer.TargetKey).ToHashSet(StringComparer.Ordinal);
        foreach (LoraFullWeightDiff diff in fullDiffs.Where(diff => !diff.IsBias)) actual.Add(diff.TargetKey);
        HashSet<string> expected = [];
        for (int i = 0; i < PublishedBlocks; i++)
        {
            string root = $"blocks.{i}";
            expected.Add($"{root}.attn.qkv_proj.weight");
            expected.Add($"{root}.attn.out_proj.weight");
            expected.Add($"{root}.mlp.fc1.weight");
            expected.Add($"{root}.mlp.fc2.weight");
            expected.Add($"{root}.adaln_proj.linear.weight");
        }
        for (int i = 0; i < PublishedRefinerBlocks; i++)
        {
            string root = $"token_refiner.blocks.{i}";
            expected.Add($"{root}.attn.qkv_proj.weight");
            expected.Add($"{root}.attn.out_proj.weight");
            expected.Add($"{root}.mlp.fc1.weight");
            expected.Add($"{root}.mlp.fc2.weight");
        }
        string[] missing = expected.Except(actual, StringComparer.Ordinal).Take(6).ToArray();
        string[] extra = actual.Except(expected, StringComparer.Ordinal).Take(6).ToArray();
        if (missing.Length > 0 || extra.Length > 0 || actual.Count != expected.Count)
        {
            throw new HartsyInferenceException(
                $"PDD trunk target set is incomplete or drifted (expected {expected.Count}, got {actual.Count}; "
                + $"missing: {string.Join(", ", missing)}; extra: {string.Join(", ", extra)}). No tensor may be skipped.");
        }
    }

    private static void RejectLeftovers(IReadOnlyDictionary<string, Tensor> tensors, HashSet<string> consumed)
    {
        string[] leftovers = tensors.Keys.Where(key => !consumed.Contains(key)).Take(8).ToArray();
        if (leftovers.Length > 0)
        {
            int count = tensors.Keys.Count(key => !consumed.Contains(key));
            throw new HartsyInferenceException(
                $"PDD adapter contains {count} unrecognized tensor(s), including {string.Join(", ", leftovers)}. "
                + "Refusing a partial acceleration merge.");
        }
    }

    private static HartsyInferenceException Duplicate(string key, string root, string role) =>
        new($"PDD tensor '{key}' duplicates the {role} for '{root}'.");

    private sealed class CanonicalGroup
    {
        public Tensor? Down { get; set; }
        public Tensor? Up { get; set; }
        public float? Alpha { get; set; }
    }
}
