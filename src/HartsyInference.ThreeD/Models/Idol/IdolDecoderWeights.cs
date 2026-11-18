using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>The F32 weights of Idol's two decoders from the Comfy-Org <c>idol_decoder_fp16</c> repack: one file with the octree
/// decoder under <c>octree.</c> and the Gaussian decoder under <c>gs.</c>. Each tensor is widened once at load, which is exact. Owns the
/// safetensors mapping and every widened copy.</summary>
public sealed class IdolDecoderWeights : IDisposable
{
    private readonly SafeTensorsLoader _loader;
    private readonly List<Tensor> _owned;
    private bool _disposed;

    private IdolDecoderWeights(IdolDecoderConfig config, Dictionary<string, Tensor> tensors, SafeTensorsLoader loader, List<Tensor> owned)
    {
        Config = config;
        Tensors = tensors;
        _loader = loader;
        _owned = owned;
    }

    public IdolDecoderConfig Config { get; }

    /// <summary>The tensors by key (<c>octree.*</c> and <c>gs.*</c>), all F32.</summary>
    public IReadOnlyDictionary<string, Tensor> Tensors { get; }

    public Tensor this[string key]
        => Tensors.TryGetValue(key, out Tensor? t) ? t : throw new HartsyInferenceException($"The Idol decoder checkpoint has no tensor '{key}'.");

    public static IdolDecoderWeights Load(string checkpointPath, IdolDecoderConfig? config = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"Idol decoder checkpoint not found: '{checkpointPath}'.", checkpointPath);
        config ??= new IdolDecoderConfig();
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
                    throw new HartsyInferenceException($"The Idol decoder checkpoint has no tensor '{key}'.");
                Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
                if (!ReferenceEquals(f32, t))
                    owned.Add(f32);
                tensors[key] = f32;
            }

            IdolDecoderWeights weights = new(config, tensors, loader, owned);
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

    /// <summary>Wraps tensors already in memory (tests); borrowed.</summary>
    internal static IdolDecoderWeights FromTensors(IdolDecoderConfig config, Dictionary<string, Tensor> tensors)
    {
        IdolDecoderWeights weights = new(config, tensors, new SafeTensorsLoader(), []);
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
                throw new HartsyInferenceException($"Idol decoder tensor '{key}' must have shape [{string.Join(", ", shape)}]; got {t.Shape}.");
        }
    }

    internal static Dictionary<string, long[]> ExpectedShapes(IdolDecoderConfig c)
    {
        int d = c.ModelChannels, h = c.MlpHidden, hd = d / c.Heads, lat = c.LatentChannels;
        Dictionary<string, long[]> s = new(StringComparer.Ordinal)
        {
            ["octree.in_proj.weight"] = [d, 3], ["octree.in_proj.bias"] = [d],
            ["octree.input_layer.weight"] = [d, d], ["octree.input_layer.bias"] = [d],
            ["octree.l_embedder.mlp.0.weight"] = [d, 256], ["octree.l_embedder.mlp.0.bias"] = [d],
            ["octree.l_embedder.mlp.2.weight"] = [d, d], ["octree.l_embedder.mlp.2.bias"] = [d],
            ["octree.adaLN_modulation.1.weight"] = [6 * d, d], ["octree.adaLN_modulation.1.bias"] = [6 * d],
            ["octree.out_proj.weight"] = [8, d], ["octree.out_proj.bias"] = [8],
            ["gs.in_proj.weight"] = [d, 3], ["gs.in_proj.bias"] = [d],
            ["gs.input_layer.weight"] = [d, d], ["gs.input_layer.bias"] = [d],
            ["gs.out_proj.weight"] = [c.GaussianOutputChannels, d], ["gs.out_proj.bias"] = [c.GaussianOutputChannels],
            ["gs.points_offset_perturbation"] = [c.GaussiansPerAnchor, 3], ["gs.base_offset_scale"] = [],
        };
        void Cross(string p)
        {
            s[$"{p}.to_q.weight"] = [d, d]; s[$"{p}.to_q.bias"] = [d];
            s[$"{p}.to_kv.weight"] = [2 * d, lat]; s[$"{p}.to_kv.bias"] = [2 * d];
            s[$"{p}.to_out.weight"] = [d, d]; s[$"{p}.to_out.bias"] = [d];
            s[$"{p}.q_rms_norm.gamma"] = [c.Heads, hd]; s[$"{p}.k_rms_norm.gamma"] = [c.Heads, hd];
        }
        void Mlp(string p)
        {
            s[$"{p}.mlp.mlp.0.weight"] = [h, d]; s[$"{p}.mlp.mlp.0.bias"] = [h];
            s[$"{p}.mlp.mlp.2.weight"] = [d, h]; s[$"{p}.mlp.mlp.2.bias"] = [d];
        }
        for (int i = 0; i < c.OctreeBlocks; i++)
        {
            Cross($"octree.blocks.{i}.cross_attn");
            Mlp($"octree.blocks.{i}");
        }

        for (int i = 0; i < c.GaussianBlocks; i++)
        {
            string p = $"gs.blocks.{i}";
            s[$"{p}.self_attn.to_qkv.weight"] = [3 * d, d]; s[$"{p}.self_attn.to_qkv.bias"] = [3 * d];
            s[$"{p}.self_attn.to_out.weight"] = [d, d]; s[$"{p}.self_attn.to_out.bias"] = [d];
            s[$"{p}.self_attn.q_rms_norm.gamma"] = [c.Heads, hd]; s[$"{p}.self_attn.k_rms_norm.gamma"] = [c.Heads, hd];
            Cross($"{p}.cross_attn");
            s[$"{p}.norm2.weight"] = [d]; s[$"{p}.norm2.bias"] = [d];
            Mlp(p);
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
