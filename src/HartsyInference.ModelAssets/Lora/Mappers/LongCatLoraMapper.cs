using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Lora.Mappers;

/// <summary>Parses LongCat-Video LoRA files (the base distill and refinement LoRAs, the Avatar DMD and OPSD LoRAs) onto the DiT's
/// own weight names. The reference
/// (<c>longcat_video/modules/lora_utils.py</c> @ 58eb336) names a module by its path with <c>.</c> spelled
/// <c>___lorahyphen___</c> and a leading <c>lora</c>,
/// stores <c>lora_down.weight</c>, <c>lora_up.weight</c> and an <c>alpha_scale</c> scalar (alpha / rank, already divided), and applies
/// <c>y += multiplier · alpha_scale · up(down(x))</c>.
/// <para>A fused projection (<c>attn.qkv</c>, three outputs; <c>cross_attn.kv_linear</c>, two) is stored split: <c>lora_down [n·rank, in]</c> and
/// <c>lora_up.blocks.{i}.weight [out/n, rank]</c>, where chunk <c>i</c> of the down output feeds up block <c>i</c> and the
/// results concatenate. That is one
/// ordinary low-rank pair of rank <c>n·rank</c> whose up matrix is block-diagonal, so the mapper builds exactly that (the
/// off-diagonal zeros cost one extra
/// small GEMM at LoRA ranks) and the stack's adjunct path carries it unchanged. A negative stored scale is folded into the up
/// matrix, because an adjunct
/// gate takes only a non-negative strength.</para></summary>
public static class LongCatLoraMapper
{
    /// <summary>Every key of the reference's LoRA state dict starts with this.</summary>
    public const string KeyPrefix = "lora___lorahyphen___";

    private const string Hyphen = "___lorahyphen___";
    private const string DownSuffix = ".lora_down.weight";
    private const string UpSuffix = ".lora_up.weight";
    private const string UpBlockMarker = ".lora_up.blocks.";
    private const string AlphaSuffix = ".alpha_scale";

    /// <summary>Whether <paramref name="key"/> is spelled the way the reference's LoRA state dict spells it.</summary>
    public static bool IsDistinctive(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return key.StartsWith(KeyPrefix, StringComparison.Ordinal);
    }

    /// <summary>The DiT weight key a LoRA module name addresses
    /// (<c>lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv</c> gives
    /// <c>blocks.0.attn.qkv.weight</c>).</summary>
    public static string TargetKeyOf(string moduleName)
    {
        ArgumentNullException.ThrowIfNull(moduleName);
        if (!moduleName.StartsWith(KeyPrefix, StringComparison.Ordinal))
            throw new ArgumentException($"'{moduleName}' does not start with '{KeyPrefix}'.", nameof(moduleName));
        return moduleName[KeyPrefix.Length..].Replace(Hyphen, ".", StringComparison.Ordinal) + ".weight";
    }

    /// <summary>Builds one layer per adapted module on <see cref="LoraTarget.Transformer"/>. Tensors the mapper allocates (a
    /// block-diagonal up matrix, a negated one)
    /// are added to <paramref name="ownedTensors"/> for the file to dispose.</summary>
    public static IReadOnlyList<LoraLayer> ParseLayers(SafeTensorsLoader loader, List<Tensor> ownedTensors)
    {
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(ownedTensors);
        SortedDictionary<string, Module> modules = new(StringComparer.Ordinal);
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!key.StartsWith(KeyPrefix, StringComparison.Ordinal))
                throw new InvalidDataException(
                    $"LongCat LoRA key '{key}' does not start with '{KeyPrefix}'; the file mixes in tensors of another layout.");
            if (key.EndsWith(DownSuffix, StringComparison.Ordinal))
                Of(modules, key[..^DownSuffix.Length]).Down = key;
            else if (key.EndsWith(UpSuffix, StringComparison.Ordinal))
                Of(modules, key[..^UpSuffix.Length]).Up = key;
            else if (key.EndsWith(AlphaSuffix, StringComparison.Ordinal))
                Of(modules, key[..^AlphaSuffix.Length]).Alpha = key;
            else if (TrySplitBlock(key, out string module, out int index))
                Of(modules, module).UpBlocks[index] = key;
            else
                throw new InvalidDataException($"LongCat LoRA key '{key}' is not a lora_down, lora_up, lora_up.blocks.N or alpha_scale tensor.");
        }

        List<LoraLayer> layers = new(modules.Count);
        foreach ((string name, Module module) in modules)
            layers.Add(Build(loader, name, module, ownedTensors));
        if (layers.Count == 0)
            throw new InvalidDataException("The LongCat LoRA file holds no modules.");
        return layers;
    }

    private static LoraLayer Build(SafeTensorsLoader loader, string name, Module module, List<Tensor> ownedTensors)
    {
        string target = TargetKeyOf(name);
        if (module.Down is null)
            throw new InvalidDataException($"LongCat LoRA module '{target}' has no lora_down matrix.");
        bool split = module.UpBlocks.Count > 0;
        if (split == (module.Up is not null))
            throw new InvalidDataException($"LongCat LoRA module '{target}' must have either lora_up.weight or lora_up.blocks.N, not "
                + (split ? "both." : "neither."));

        Tensor down = loader.GetTensor(module.Down);
        if (down.Shape.Rank != 2)
            throw new InvalidDataException($"LongCat LoRA '{module.Down}' must be rank 2; got {down.Shape}.");
        Tensor up = split ? BlockDiagonalUp(loader, target, module, (int)down.Shape[0], ownedTensors) : loader.GetTensor(module.Up!);
        if (up.Shape.Rank != 2 || up.Shape[1] != down.Shape[0])
            throw new InvalidDataException($"LongCat LoRA '{target}': up {up.Shape} does not pair with down {down.Shape}.");

        if (module.Alpha is null)
            Logs.Warning($"LongCat LoRA module '{target}' has no alpha_scale; applying it at scale 1.");
        float scale = module.Alpha is null ? 1f : ReadScalar(loader, module.Alpha);
        if (!float.IsFinite(scale))
            throw new InvalidDataException($"LongCat LoRA '{module.Alpha}' is not finite.");
        if (scale < 0f)
        {
            if (split)
                Negate(up);
            else
                up = Negated(up, ownedTensors);
            scale = -scale;
        }
        return new LoraLayer
        {
            TargetKey = target,
            Target = LoraTarget.Transformer,
            Delta = new StandardLoraDelta { Down = down, Up = up, Alpha = scale * down.Shape[0] },
        };
    }

    /// <summary>The block-diagonal <c>[Σ out_i, n·rank]</c> up matrix whose block <c>i</c> is <c>lora_up.blocks.i</c> over
    /// columns <c>[i·rank, (i+1)·rank)</c>.</summary>
    private static unsafe Tensor BlockDiagonalUp(SafeTensorsLoader loader, string target, Module module, int downRows, List<Tensor> ownedTensors)
    {
        int n = module.UpBlocks.Count;
        for (int i = 0; i < n; i++)
        {
            if (!module.UpBlocks.ContainsKey(i))
                throw new InvalidDataException(
                    $"LongCat LoRA module '{target}' has lora_up.blocks up to {module.UpBlocks.Keys.Max()} but no block {i}.");
        }
        if (downRows % n != 0)
            throw new InvalidDataException($"LongCat LoRA module '{target}': lora_down has {downRows} rows, which {n} up blocks do not divide.");
        int rank = downRows / n;

        Tensor[] blocks = new Tensor[n];
        long rows = 0;
        for (int i = 0; i < n; i++)
        {
            blocks[i] = loader.GetTensor(module.UpBlocks[i]);
            if (blocks[i].Shape.Rank != 2 || blocks[i].Shape[1] != rank)
                throw new InvalidDataException($"LongCat LoRA '{module.UpBlocks[i]}' must be [out, {rank}]; got {blocks[i].Shape}.");
            rows += blocks[i].Shape[0];
        }

        Tensor up = new(new TensorShape(rows, downRows), DType.F32);
        ownedTensors.Add(up);
        new Span<float>((float*)up.DataPointer, checked((int)(rows * downRows))).Clear();
        long rowOffset = 0;
        for (int i = 0; i < n; i++)
        {
            using Tensor block = blocks[i].CastTo(DType.F32);
            ReadOnlySpan<float> source = block.AsReadOnlySpan<float>();
            Span<float> destination = up.AsSpan<float>();
            long blockRows = block.Shape[0];
            for (long r = 0; r < blockRows; r++)
                source.Slice((int)(r * rank), rank).CopyTo(destination.Slice((int)((rowOffset + r) * downRows + (long)i * rank), rank));
            rowOffset += blockRows;
        }
        return up;
    }

    /// <summary>Negates a mapper-owned F32 tensor in place.</summary>
    private static unsafe void Negate(Tensor owned)
    {
        Span<float> values = owned.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = -values[i];
    }

    private static unsafe Tensor Negated(Tensor up, List<Tensor> ownedTensors)
    {
        using Tensor f32 = up.CastTo(DType.F32);
        Tensor negated = new(f32.Shape, DType.F32);
        ownedTensors.Add(negated);
        ReadOnlySpan<float> source = f32.AsReadOnlySpan<float>();
        Span<float> destination = negated.AsSpan<float>();
        for (int i = 0; i < source.Length; i++)
            destination[i] = -source[i];
        return negated;
    }

    private static float ReadScalar(SafeTensorsLoader loader, string key)
    {
        using Tensor f32 = loader.GetTensor(key).CastTo(DType.F32);
        if (f32.ElementCount != 1)
            throw new InvalidDataException($"LongCat LoRA '{key}' must be a scalar; got {f32.Shape}.");
        return f32.AsReadOnlySpan<float>()[0];
    }

    private static bool TrySplitBlock(string key, out string module, out int index)
    {
        module = string.Empty;
        index = -1;
        int at = key.IndexOf(UpBlockMarker, StringComparison.Ordinal);
        if (at < 0 || !key.EndsWith(".weight", StringComparison.Ordinal))
            return false;
        ReadOnlySpan<char> number = key.AsSpan(at + UpBlockMarker.Length, key.Length - at - UpBlockMarker.Length - ".weight".Length);
        // Canonical decimal only: "blocks.01" would otherwise parse to block 1 and collide with it silently.
        if (!int.TryParse(number, NumberStyles.None, CultureInfo.InvariantCulture, out index) || index < 0
            || index.ToString(CultureInfo.InvariantCulture) != number.ToString())
            return false;
        module = key[..at];
        return true;
    }

    private static Module Of(SortedDictionary<string, Module> modules, string name)
    {
        if (!modules.TryGetValue(name, out Module? module))
            modules[name] = module = new Module();
        return module;
    }

    private sealed class Module
    {
        public string? Down, Up, Alpha;
        public readonly SortedDictionary<int, string> UpBlocks = [];
    }
}
