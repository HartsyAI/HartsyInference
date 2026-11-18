using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>The 7.56 M parameters Budget3D adds to Idol's denoiser (the <c>bp_*</c> tensors of its checkpoint, all F32): the two scale mixers, the integrated position encoder, the per-scale embedding and the two
/// refine stages. <c>bp_feat_proj</c> is a training-only head that inference never calls, so it is not read. Owns the tensors it loads.</summary>
public sealed class BudgetAdapterWeights : IDisposable
{
    /// <summary>Hidden width of the mixers' bottleneck (<c>mix_bott</c>).</summary>
    public const int MixerBottleneck = 256;

    /// <summary>Hidden width of the position encoder (<c>ipe_bottleneck</c>).</summary>
    public const int IpeBottleneck = 256;

    /// <summary>Octaves of the integrated position encoding.</summary>
    public const int IpeOctaves = 8;

    /// <summary>Width of the refine stages (<c>refine_width</c>).</summary>
    public const int RefineWidth = 128;

    /// <summary>Blocks of each refine stage.</summary>
    public const int RefineDepth = 2;

    /// <summary>Attention heads of the refine blocks.</summary>
    public const int RefineHeads = 4;

    private readonly SafeTensorsLoader _loader;
    private readonly List<Tensor> _owned;
    private int _disposed;

    private BudgetAdapterWeights(IdolConfig config, Dictionary<string, Tensor> tensors, SafeTensorsLoader loader, List<Tensor> owned)
    {
        Config = config;
        Tensors = tensors;
        _loader = loader;
        _owned = owned;
    }

    /// <summary>The Idol architecture these adapters sit on.</summary>
    public IdolConfig Config { get; }

    /// <summary>The adapters' tensors by checkpoint key.</summary>
    public IReadOnlyDictionary<string, Tensor> Tensors { get; }

    /// <summary>The tensor under <paramref name="key"/>.</summary>
    /// <exception cref="HartsyInferenceException">There is none.</exception>
    public Tensor this[string key]
        => Tensors.TryGetValue(key, out Tensor? t) ? t : throw new HartsyInferenceException($"The Budget3D checkpoint has no tensor '{key}'.");

    /// <summary>Opens a Budget3D checkpoint and reads its adapter tensors, checking each against its shape.</summary>
    /// <exception cref="FileNotFoundException">The file is missing.</exception>
    /// <exception cref="HartsyInferenceException">A tensor is missing or mis-shaped: the file is not a Budget3D checkpoint of this architecture.</exception>
    public static BudgetAdapterWeights Load(string checkpointPath, IdolConfig? config = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException("The Budget3D checkpoint is missing.", checkpointPath);
        config ??= new IdolConfig();
        SafeTensorsLoader loader = new();
        List<Tensor> owned = [];
        try
        {
            loader.Load(checkpointPath);
            Dictionary<string, Tensor> source = loader.GetAllTensors();
            Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
            foreach ((string key, long[] shape) in ExpectedShapes(config))
            {
                if (!source.TryGetValue(key, out Tensor? t))
                    throw new HartsyInferenceException($"The Budget3D checkpoint has no tensor '{key}'.");
                bool match = t.Shape.Rank == shape.Length;
                for (int d = 0; match && d < shape.Length; d++)
                    match = t.Shape[d] == shape[d];
                if (!match)
                    throw new HartsyInferenceException($"Budget3D tensor '{key}' must have shape [{string.Join(", ", shape)}]; got {t.Shape}.");
                Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
                if (!ReferenceEquals(f32, t))
                    owned.Add(f32);
                tensors[key] = f32;
            }

            return new BudgetAdapterWeights(config, tensors, loader, owned);
        }
        catch
        {
            foreach (Tensor t in owned)
                t.Dispose();
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Wraps tensors already in memory (tests); borrowed.</summary>
    internal static BudgetAdapterWeights FromTensors(IdolConfig config, Dictionary<string, Tensor> tensors) => new(config, tensors, new SafeTensorsLoader(), []);

    /// <summary>Every adapter tensor the forward reads and its shape.</summary>
    internal static Dictionary<string, long[]> ExpectedShapes(IdolConfig c)
    {
        long d = c.ModelChannels, lat = c.LatentChannels;
        Dictionary<string, long[]> s = new(StringComparer.Ordinal) { ["bp_scale_emb"] = [4, d] };
        foreach (int n in new[] { 2, 4 })
        {
            int sub = n == 2 ? 2 : 8;
            string p = $"bp_mixers.{n}";
            s[$"{p}.pos"] = [sub, d];
            s[$"{p}.down.weight"] = [MixerBottleneck, d];
            s[$"{p}.down.bias"] = [MixerBottleneck];
            s[$"{p}.spatial_w"] = [MixerBottleneck, sub, sub];
            s[$"{p}.spatial_b"] = [sub, MixerBottleneck];
            s[$"{p}.up.weight"] = [d, (long)sub * MixerBottleneck];
            s[$"{p}.up.bias"] = [d];
        }

        s["bp_ipe.mlp.0.weight"] = [IpeBottleneck, 12 * IpeOctaves];
        s["bp_ipe.mlp.0.bias"] = [IpeBottleneck];
        s["bp_ipe.mlp.2.weight"] = [d, IpeBottleneck];
        s["bp_ipe.mlp.2.bias"] = [d];
        foreach (int n in new[] { 2, 4 })
        {
            int children = n == 2 ? 2 : 8;
            string p = $"bp_refiners.{n}";
            s[$"{p}.embed.weight"] = [RefineWidth, lat];
            s[$"{p}.embed.bias"] = [RefineWidth];
            s[$"{p}.embed_zt.weight"] = [RefineWidth, lat];
            s[$"{p}.embed_zt.bias"] = [RefineWidth];
            s[$"{p}.pos_proj.weight"] = [RefineWidth, 24];
            s[$"{p}.pos_proj.bias"] = [RefineWidth];
            s[$"{p}.slots"] = [children, RefineWidth];
            for (int b = 0; b < RefineDepth; b++)
            {
                string q = $"{p}.blocks.{b}";
                s[$"{q}.qkv.weight"] = [3 * RefineWidth, RefineWidth];
                s[$"{q}.qkv.bias"] = [3 * RefineWidth];
                s[$"{q}.proj.weight"] = [RefineWidth, RefineWidth];
                s[$"{q}.proj.bias"] = [RefineWidth];
                s[$"{q}.mlp.0.weight"] = [4 * RefineWidth, RefineWidth];
                s[$"{q}.mlp.0.bias"] = [4 * RefineWidth];
                s[$"{q}.mlp.2.weight"] = [RefineWidth, 4 * RefineWidth];
                s[$"{q}.mlp.2.bias"] = [RefineWidth];
                s[$"{q}.ada.weight"] = [6 * RefineWidth, d];
                s[$"{q}.ada.bias"] = [6 * RefineWidth];
            }

            s[$"{p}.out_norm.weight"] = [RefineWidth];
            s[$"{p}.out.weight"] = [lat, RefineWidth];
            s[$"{p}.out.bias"] = [lat];
        }

        return s;
    }

    /// <summary>Releases the converted tensors and the file.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor t in _owned)
            t.Dispose();
        _loader.Dispose();
    }
}
