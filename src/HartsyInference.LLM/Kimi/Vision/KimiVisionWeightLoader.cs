using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>Reads MoonViT-V2 and projector tensors (<c>vision_tower.*</c>, <c>mm_projector.*</c>) from the same source as the text decoder, converting each to F32.</summary>
public static class KimiVisionWeightLoader
{
    private const string TowerRoot = "vision_tower.";
    private const string ProjectorRoot = "mm_projector.";

    /// <summary>Loads every tower and projector tensor. The result owns them; dispose it with the encoder.</summary>
    /// <exception cref="KeyNotFoundException">The source lacks tensors the tower needs; the message names every one.</exception>
    /// <exception cref="InvalidDataException">A tensor's shape differs from <paramref name="config"/>.</exception>
    public static KimiVisionWeights Load(KimiVisionConfig config, IKimiK31WeightSource source)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        config.Validate();

        List<string> missing = [];
        foreach ((string name, long[] _) in Requirements(config))
            if (!source.Has(name)) missing.Add(name);
        if (missing.Count > 0)
            throw new KeyNotFoundException($"The checkpoint lacks {missing.Count} vision tensor(s): {string.Join(", ", missing)}.");

        Reader reader = new(source);
        try
        {
            return reader.Build(config);
        }
        catch
        {
            // A failed load releases what it already created; the caller never sees partial weights.
            reader.ReleaseOwned();
            throw;
        }
    }

    /// <summary>The first reason the checkpoint's tensor inventory cannot supply the tower, or null when every tensor the tower reads is present with
    /// its shape. Reads tensor headers only.</summary>
    internal static string? Refusal(KimiVisionConfig config, IReadOnlyDictionary<string, TensorLocation> inventory)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(inventory);
        config.Validate();

        List<string> missing = [];
        foreach ((string name, long[] dims) in Requirements(config))
        {
            if (!inventory.TryGetValue(name, out TensorLocation? location))
                missing.Add(name);
            else if (location.Shape != new TensorShape(dims))
                return Mismatch(name, location.Shape, dims).Message;
        }
        return missing.Count == 0 ? null : $"The checkpoint lacks {missing.Count} vision tensor(s): {string.Join(", ", missing)}.";
    }

    internal static List<(string Name, long[] Dims)> Requirements(KimiVisionConfig c)
    {
        long h = c.Hidden, q = c.QkvHidden, m = c.MlpDim, g = c.InitPosGrid, p = c.PatchSize, mm = c.MergedDim, t = c.TextHidden;
        List<(string Name, long[] Dims)> list =
        [
            (TowerRoot + "patch_embed.proj.weight", [h, 3, p, p]),
            (TowerRoot + "patch_embed.pos_emb.weight", [g, g, h]),
            (TowerRoot + "encoder.final_layernorm.weight", [h]),
            (ProjectorRoot + "proj.0.weight", [mm, mm]),
            (ProjectorRoot + "proj.2.weight", [t, mm]),
            (ProjectorRoot + "post_norm.weight", [t]),
        ];
        for (int i = 0; i < c.Layers; i++)
        {
            string b = BlockRoot(i);
            list.Add((b + "wqkv.weight", [3 * q, h]));
            list.Add((b + "wo.weight", [h, q]));
            list.Add((b + "norm0.weight", [h]));
            list.Add((b + "norm1.weight", [h]));
            list.Add((b + "mlp.fc0.weight", [m, h]));
            list.Add((b + "mlp.fc1.weight", [h, m]));
        }
        return list;
    }

    private static string BlockRoot(int i) => $"{TowerRoot}encoder.blocks.{i}.";

    private static InvalidDataException Mismatch(string name, TensorShape got, long[] want) =>
        new($"Tensor '{name}' has shape {got}, the config needs [{string.Join(", ", want)}].");

    /// <summary>Creates each tensor in turn, so at most one host copy is alive during a load.</summary>
    private sealed class Reader
    {
        private readonly IKimiK31WeightSource _source;
        private readonly List<Tensor> _owned = [];

        public Reader(IKimiK31WeightSource source) => _source = source;

        public KimiVisionWeights Build(KimiVisionConfig c)
        {
            long h = c.Hidden, q = c.QkvHidden, m = c.MlpDim, g = c.InitPosGrid, p = c.PatchSize, mm = c.MergedDim, t = c.TextHidden;
            Tensor patch = Load(TowerRoot + "patch_embed.proj.weight", [h, 3, p, p], [h, 3 * p * p]);
            Tensor pos = Load(TowerRoot + "patch_embed.pos_emb.weight", [g, g, h], [g * g, h]);
            List<KimiVisionBlockWeights> blocks = new(c.Layers);
            for (int i = 0; i < c.Layers; i++) blocks.Add(Block(c, i));
            Tensor finalNorm = Load(TowerRoot + "encoder.final_layernorm.weight", [h]);
            Tensor up = Load(ProjectorRoot + "proj.0.weight", [mm, mm]);
            Tensor down = Load(ProjectorRoot + "proj.2.weight", [t, mm]);
            Tensor postNorm = Load(ProjectorRoot + "post_norm.weight", [t]);
            return new KimiVisionWeights(_owned, patch, pos, blocks, finalNorm, up, down, postNorm);
        }

        public void ReleaseOwned()
        {
            foreach (Tensor tensor in _owned) tensor.Dispose();
            _owned.Clear();
        }

        private KimiVisionBlockWeights Block(KimiVisionConfig c, int i)
        {
            string b = BlockRoot(i);
            long h = c.Hidden, q = c.QkvHidden, m = c.MlpDim;
            int n = (int)(q * h);

            // The fused projection stacks q, k and v along its output rows.
            float[] fused = Host(b + "wqkv.weight", [3 * q, h]);
            Tensor wq = Make(fused.AsSpan(0, n), q, h);
            Tensor wk = Make(fused.AsSpan(n, n), q, h);
            Tensor wv = Make(fused.AsSpan(2 * n, n), q, h);
            return new KimiVisionBlockWeights(
                wq,
                wk,
                wv,
                Load(b + "wo.weight", [h, q]),
                Load(b + "norm0.weight", [h]),
                Load(b + "norm1.weight", [h]),
                Load(b + "mlp.fc0.weight", [m, h]),
                Load(b + "mlp.fc1.weight", [h, m]));
        }

        private Tensor Load(string name, long[] dims) => Load(name, dims, dims);

        private Tensor Load(string name, long[] dims, long[] shape) => Make(Host(name, dims), shape);

        // Reads one tensor as host F32 values and checks its exact shape; the source's tensor is disposed here.
        private float[] Host(string name, long[] dims)
        {
            Tensor raw = _source.Read(name);
            Tensor f = TensorCasts.EnsureF32(raw);
            try
            {
                if (f.Shape.Rank != dims.Length) throw Mismatch(name, f.Shape, dims);
                for (int d = 0; d < dims.Length; d++)
                    if (f.Shape[d] != dims[d]) throw Mismatch(name, f.Shape, dims);
                return f.AsReadOnlySpan<float>().ToArray();
            }
            finally
            {
                if (!ReferenceEquals(f, raw)) f.Dispose();
                raw.Dispose();
            }
        }

        private Tensor Make(ReadOnlySpan<float> values, params long[] shape)
        {
            Tensor t = new(new TensorShape(shape), DType.F32);
            values.CopyTo(t.AsSpan<float>());
            _owned.Add(t);
            return t;
        }
    }
}
