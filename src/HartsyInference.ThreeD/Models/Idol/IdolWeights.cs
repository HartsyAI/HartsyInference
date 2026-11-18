using System.Security.Cryptography;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>The F32 weights of Idol's denoiser, from a Comfy-Org <c>idol_fp16</c> repack or a BudgetAny <c>budget3d_idol</c> file (both use
/// idol-sdk's key names; the second adds <c>bp_*</c> tensors this class leaves alone). Each tensor is widened once at load, which is exact.
/// The token positions <c>pos_pe</c> are in neither file, so they are read from the pinned raw file beside the weights. Owns the
/// safetensors mapping and every widened copy.</summary>
public sealed class IdolWeights : IDisposable
{
    /// <summary>The file name of the 98,304-byte raw F32 <c>[1, 8192, 3]</c> token positions, a slice of <c>speridlabs/idol</c>'s
    /// <c>denoiser.safetensors</c>; it equals <c>SobolEngine(3, scramble=True, seed=123).draw(8192)</c>. The engine carries it as an
    /// embedded resource, so a model folder needs no copy.</summary>
    public const string PosPeFileName = "pos_pe_f32_1x8192x3.bin";

    /// <summary>SHA-256 of <see cref="PosPeFileName"/>.</summary>
    public const string PosPeSha256 = "884df4db664a680c100d032c9855397ced6afa28f8984efbee6b77f6ba3efe6b";

    private readonly SafeTensorsLoader _loader;
    private readonly List<Tensor> _owned;
    private bool _disposed;

    private IdolWeights(IdolConfig config, Dictionary<string, Tensor> tensors, SafeTensorsLoader loader, List<Tensor> owned)
    {
        Config = config;
        Tensors = tensors;
        _loader = loader;
        _owned = owned;
    }

    /// <summary>The architecture these weights were checked against.</summary>
    public IdolConfig Config { get; }

    /// <summary>The tensors by idol-sdk key, all F32, plus <c>pos_pe</c>.</summary>
    public IReadOnlyDictionary<string, Tensor> Tensors { get; }

    /// <summary>The tensor under idol-sdk key <paramref name="key"/>.</summary>
    /// <exception cref="HartsyInferenceException">There is none.</exception>
    public Tensor this[string key]
        => Tensors.TryGetValue(key, out Tensor? t) ? t : throw new HartsyInferenceException($"The Idol checkpoint has no tensor '{key}'.");

    /// <summary>Opens <paramref name="checkpointPath"/> and the pinned token positions (the engine's embedded copy, or the file at
    /// <paramref name="posPePath"/>), and checks every tensor the architecture reads.</summary>
    /// <exception cref="FileNotFoundException">A file is missing.</exception>
    /// <exception cref="HartsyInferenceException">A tensor is missing or mis-shaped, or the token positions do not match their pin.</exception>
    /// <param name="linearDType">What the blocks' linear weights (qkv, out and the MLP of the refiners and joint blocks, nearly all of the 370 M parameters) are held as:
    /// F32 (an exact widening of the file, what the parity tests use) or F16, which halves their memory and runs the GEMMs on 16-bit tensor cores with F32 accumulation, as idol-sdk does
    /// (its blocks are F16). Embedders, the rotary layers, norms and heads stay F32.</param>
    public static IdolWeights Load(string checkpointPath, string? posPePath = null, IdolConfig? config = null, DType? linearDType = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"Idol checkpoint not found: '{checkpointPath}'.", checkpointPath);
        if (posPePath is not null && !File.Exists(posPePath))
            throw new FileNotFoundException($"Idol token positions not found: '{posPePath}'.", posPePath);
        config ??= new IdolConfig();

        SafeTensorsLoader loader = new();
        List<Tensor> owned = [];
        try
        {
            loader.Load(checkpointPath);
            Dictionary<string, Tensor> source = loader.GetAllTensors();
            Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
            foreach (string key in ExpectedShapes(config).Keys)
            {
                if (!source.TryGetValue(key, out Tensor? t))
                    throw new HartsyInferenceException($"The Idol checkpoint has no tensor '{key}'.");
                DType target = linearDType is { } narrow && narrow != DType.F32 && IsBlockLinear(key) ? narrow : DType.F32;
                Tensor converted = t.DType == target ? t : t.CastTo(target);
                if (!ReferenceEquals(converted, t))
                    owned.Add(converted);
                tensors[key] = converted;
            }

            Tensor posPe = ReadPosPe(posPePath, config);
            owned.Add(posPe);
            tensors["pos_pe"] = posPe;
            IdolWeights weights = new(config, tensors, loader, owned);
            weights.CheckShapes();
            return weights;
        }
        catch
        {
            foreach (Tensor t in owned)
                t.Dispose();
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Wraps tensors already in memory (tests); borrowed, and <c>pos_pe</c> must be among them.</summary>
    internal static IdolWeights FromTensors(IdolConfig config, Dictionary<string, Tensor> tensors)
    {
        IdolWeights weights = new(config, tensors, new SafeTensorsLoader(), []);
        weights.CheckShapes();
        return weights;
    }

    private void CheckShapes()
    {
        foreach ((string key, long[] shape) in ExpectedShapes(Config))
        {
            Tensor t = this[key];
            bool match = t.Shape.Rank == shape.Length;
            for (int d = 0; match && d < shape.Length; d++)
                match = t.Shape[d] == shape[d];
            if (!match)
                throw new HartsyInferenceException($"Idol tensor '{key}' must have shape [{string.Join(", ", shape)}]; got {t.Shape}.");
        }
    }

    private static unsafe Tensor ReadPosPe(string? path, IdolConfig config)
    {
        byte[] raw = path is not null ? File.ReadAllBytes(path) : EmbeddedPosPe();
        string hash = Convert.ToHexString(SHA256.HashData(raw)).ToLowerInvariant();
        if (config.Tokens == 8192 && hash != PosPeSha256)
            throw new HartsyInferenceException($"'{path ?? "the embedded copy"}' is not Idol's token positions: sha256 {hash}, expected {PosPeSha256}.");
        long expected = (long)config.Tokens * 3 * sizeof(float);
        if (raw.Length != expected)
            throw new HartsyInferenceException($"Idol token positions must be {expected} bytes; '{path ?? "the embedded copy"}' has {raw.Length}.");
        Tensor t = new(new TensorShape(1, config.Tokens, 3), DType.F32);
        fixed (byte* src = raw)
            Buffer.MemoryCopy(src, (void*)t.DataPointer, expected, expected);
        return t;
    }

    // The matrices of the attention and MLP in the refiners and the joint blocks.
    private static bool IsBlockLinear(string key)
        => (key.StartsWith("blocks.", StringComparison.Ordinal) || key.StartsWith("noise_refiner.", StringComparison.Ordinal) || key.StartsWith("context_refiner.", StringComparison.Ordinal))
            && (key.EndsWith(".attn.qkv.weight", StringComparison.Ordinal) || key.EndsWith(".attn.out.weight", StringComparison.Ordinal)
                || key.EndsWith(".mlp.mlp.0.weight", StringComparison.Ordinal) || key.EndsWith(".mlp.mlp.2.weight", StringComparison.Ordinal));

    /// <summary>Idol's 8192 token positions, <c>[8192 × 3]</c> floats, from the engine's embedded copy; Budget3D's checkpoints carry no <c>pos_pe</c> and use these.</summary>
    public static float[] EmbeddedPositions()
    {
        byte[] raw = EmbeddedPosPe();
        float[] values = new float[raw.Length / sizeof(float)];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return values;
    }

    private static byte[] EmbeddedPosPe()
    {
        using Stream stream = typeof(IdolWeights).Assembly.GetManifestResourceStream($"idol/{PosPeFileName}")
            ?? throw new HartsyInferenceException("The engine's embedded copy of Idol's token positions is missing.");
        using MemoryStream copy = new();
        stream.CopyTo(copy);
        return copy.ToArray();
    }

    /// <summary>Every tensor the denoiser reads and its shape (<c>pos_pe</c> excluded: it comes from its own file).</summary>
    internal static Dictionary<string, long[]> ExpectedShapes(IdolConfig c)
    {
        int d = c.ModelChannels, h = c.MlpHidden, hd = c.HeadDim, heads = c.Heads;
        Dictionary<string, long[]> s = new(StringComparer.Ordinal)
        {
            ["t_embedder.mlp.0.weight"] = [d, 256], ["t_embedder.mlp.0.bias"] = [d],
            ["t_embedder.mlp.2.weight"] = [d, d], ["t_embedder.mlp.2.bias"] = [d],
            ["adaLN_modulation.1.weight"] = [6 * d, d], ["adaLN_modulation.1.bias"] = [6 * d],
            ["input_layer.weight"] = [d, c.LatentChannels], ["input_layer.bias"] = [d],
            ["cond_embedder.weight"] = [d, c.CondChannels], ["cond_embedder.bias"] = [d],
            ["cond_embedder2.weight"] = [d, c.Cond2Channels], ["cond_embedder2.bias"] = [d],
            ["cam_refiner.mlp.0.weight"] = [d, c.CameraChannels], ["cam_refiner.mlp.0.bias"] = [d],
            ["cam_refiner.mlp.2.weight"] = [d, d], ["cam_refiner.mlp.2.bias"] = [d],
            ["shift_table"] = [1, 2, d],
            ["out_layer.weight"] = [c.LatentChannels, d], ["out_layer.bias"] = [c.LatentChannels],
            ["cam_out_layer.weight"] = [c.CameraChannels, d], ["cam_out_layer.bias"] = [c.CameraChannels],
        };
        void Repo(string p)
        {
            s[$"{p}.norm.weight"] = [d]; s[$"{p}.norm.bias"] = [d];
            s[$"{p}.gate_map.weight"] = [d / 8, d]; s[$"{p}.content_map.weight"] = [d / 8, d];
            s[$"{p}.final_map.weight"] = [3 * heads, d / 8];
            s[$"{p}.freqs_0"] = [2 * (hd / 6) / 2]; s[$"{p}.freqs_1"] = [2 * (hd / 6) / 2]; s[$"{p}.freqs_2"] = [(hd - 2 * (2 * (hd / 6))) / 2];
        }
        void Block(string p, bool modulated)
        {
            s[$"{p}.attn.qkv.weight"] = [3 * d, d]; s[$"{p}.attn.qkv.bias"] = [3 * d];
            s[$"{p}.attn.out.weight"] = [d, d]; s[$"{p}.attn.out.bias"] = [d];
            s[$"{p}.attn.q_norm.gamma"] = [heads, hd]; s[$"{p}.attn.k_norm.gamma"] = [heads, hd];
            s[$"{p}.mlp.mlp.0.weight"] = [h, d]; s[$"{p}.mlp.mlp.0.bias"] = [h];
            s[$"{p}.mlp.mlp.2.weight"] = [d, h]; s[$"{p}.mlp.mlp.2.bias"] = [d];
            if (modulated)
            {
                s[$"{p}.shift_table"] = [1, 6 * d];
            }
            else
            {
                s[$"{p}.norm1.weight"] = [d]; s[$"{p}.norm1.bias"] = [d];
                s[$"{p}.norm2.weight"] = [d]; s[$"{p}.norm2.bias"] = [d];
            }
        }
        for (int i = 0; i < c.RefinerBlocks; i++)
        {
            Repo($"noise_repo_layers.{i}"); Repo($"context_repo_layers.{i}");
            Block($"noise_refiner.{i}", modulated: true); Block($"context_refiner.{i}", modulated: false);
        }
        for (int i = 0; i < c.Blocks; i++)
        {
            Repo($"repo_layers.{i}");
            Block($"blocks.{i}", modulated: true);
        }
        return s;
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor t in _owned)
            t.Dispose();
        _loader.Dispose();
    }
}
