using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using static HartsyInference.ThreeD.Models.CoLa3D.CoLa3DOps;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>CoLa3D decomposer, feed-forward CPU reference. Takes a sparse latent with precomputed Utonia features and returns
/// per-voxel point features, per-voxel mask logits for each query, and the query features. Borrows the weights it is built
/// from, which the caller keeps alive for the decomposer's lifetime.</summary>
/// <remarks>Implements the q20 path only: <c>latent_utonia_1024</c> input fusion, 3D RoPE in both attention directions, the
/// cosine mask head, and no image cross-attention, coordinate PE or MLP mask decoder.</remarks>
public sealed class CoLa3DDecomposer : IDisposable
{
    private readonly CoLa3DConfig _config;
    private readonly RopeFrequencyPlan _plan;
    private readonly CoLa3DBackbone _backbone;
    private readonly CoLa3DMaskHead _head;
    private readonly CoLa3DFixedQueries? _fixed;
    private readonly CoLa3DMaskPromptEncoder? _maskPrompt;
    private readonly CoLa3DPointPromptEncoder? _pointPrompt;
    private int _disposed;

    /// <param name="config">Validated architecture; the weight set must match it exactly.</param>
    /// <param name="weights">Parameters by name from <see cref="CoLa3DKeyMap.Remap"/>.</param>
    public CoLa3DDecomposer(CoLa3DConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        CoLa3DKeyMap.ValidateNamed(config, weights);
        _config = config;
        _plan = CoLa3DRope.Plan(config.HeadDim);
        _backbone = new CoLa3DBackbone(weights, config);
        _head = new CoLa3DMaskHead(weights, config);
        switch (config.QuerySource)
        {
            case CoLa3DConfig.QuerySourceKind.Fixed:
                _fixed = new CoLa3DFixedQueries(weights, config);
                break;
            case CoLa3DConfig.QuerySourceKind.PromptMask:
                _maskPrompt = new CoLa3DMaskPromptEncoder(weights, config);
                break;
            case CoLa3DConfig.QuerySourceKind.PromptPoint:
                _pointPrompt = new CoLa3DPointPromptEncoder(weights, config);
                break;
        }
    }

    /// <summary>Runs the decomposer on a batch of latents. <paramref name="latent"/> holds voxels sorted by batch index
    /// (contiguous from 0), with features <c>[N, LatentChannels]</c>; <paramref name="utonia"/> is <c>[N, UtoniaFeatureDim]</c>
    /// on the same voxels. Prompt inputs are required for the prompt query sources and rejected for <c>Fixed</c>.</summary>
    public CoLa3DOutput Forward(IBackend backend, SparseTensor latent, Tensor utonia, CoLa3DQueryPrompts? prompts = null)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentNullException.ThrowIfNull(utonia);
        int n = latent.Count;
        ValidateInputs(latent, utonia, n);
        (int Start, int End)[] ranges = BatchRanges(latent.Coords, n);
        int queries = ValidatePrompts(prompts, latent, ranges);
        bool[] active = ActiveMask(prompts, ranges.Length, queries);
        int width = _config.HiddenDim;

        Tensor latent3 = latent.Feats.Reshape(new TensorShape(1, n, _config.LatentChannels));
        Tensor utonia3 = utonia.Reshape(new TensorShape(1, n, _config.UtoniaFeatureDim));
        Tensor fused = _backbone.FuseInput(backend, latent3, utonia3);
        Tensor? queryRows = null, points = null, logits = null, queryFeatures = null;
        try
        {
            queryRows = QueryRows(backend, latent, fused, ranges, prompts, queries);
            points = Alloc(n, width);
            logits = Alloc(n, queries);
            queryFeatures = Alloc(ranges.Length, queries, width);
            for (int b = 0; b < ranges.Length; b++)
            {
                (int s, int e) = ranges[b];
                int nb = e - s;
                int[] slots = ActiveSlotIndices(active, b, queries);
                Tensor pIn = Alloc(1, nb, width);
                CopyRows(fused, s, nb, pIn, 0);
                if (slots.Length == 0)
                {
                    Tensor pOnly = _backbone.FinalPoints(backend, pIn);
                    CopyRows(pOnly, 0, nb, points, s);
                    pOnly.Dispose();
                    continue;
                }

                Tensor qIn = Alloc(1, slots.Length, width);
                GatherRows(queryRows, b * queries, slots, qIn, 0);
                using CoLa3DRopeTable pRope = CoLa3DRope.Build(_plan, PointPositions(latent.Coords, s, nb), nb);
                using CoLa3DRopeTable qRope = CoLa3DRope.Build(_plan, SlotPositions(prompts, b, queries, slots), slots.Length);
                (Tensor pOut, Tensor qOut) = _backbone.Run(backend, pIn, qIn, pRope, qRope);
                CopyRows(pOut, 0, nb, points, s);
                ScatterRows(qOut, 0, slots, queryFeatures, b * queries);

                Tensor lg = _head.Logits(backend, pOut, qOut);
                ScatterLogits(lg, s, nb, slots, queries, logits);
                lg.Dispose();
                pOut.Dispose();
                qOut.Dispose();
            }

            CoLa3DOutput output = new(latent.Replace(points), logits, queryFeatures, active);
            points = null;
            logits = null;
            queryFeatures = null;
            return output;
        }
        finally
        {
            queryRows?.Dispose();
            points?.Dispose();
            logits?.Dispose();
            queryFeatures?.Dispose();
            fused.Dispose();
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _backbone.Dispose();
    }

    /// <summary>Query rows <c>[1, Batch·Q, C]</c> for the configured query source.</summary>
    private Tensor QueryRows(
        IBackend backend,
        SparseTensor latent,
        Tensor fused,
        (int Start, int End)[] ranges,
        CoLa3DQueryPrompts? prompts,
        int queries)
    {
        switch (_config.QuerySource)
        {
            case CoLa3DConfig.QuerySourceKind.PromptMask:
                return _maskPrompt!.Encode(backend, prompts!.Masks!);
            case CoLa3DConfig.QuerySourceKind.PromptPoint:
                Tensor source = _config.PointFeatureSource == CoLa3DConfig.PointFeatureSourceKind.LatentUtonia ? fused : latent.Feats;
                Tensor gathered = CoLa3DPointPromptEncoder.GatherLatent(
                    source, ranges, latent.Coords, prompts!.Points!, prompts.Valid, queries, _config.LatentResolution);
                Tensor rows = _pointPrompt!.Encode(backend, prompts.Points!, gathered);
                gathered.Dispose();
                return rows;
            default:
                Tensor fixedRows = _fixed!.Compute(backend);
                Tensor all = Alloc(1, ranges.Length * queries, _config.HiddenDim);
                for (int b = 0; b < ranges.Length; b++)
                    CopyRows(fixedRows, 0, queries, all, b * queries);
                fixedRows.Dispose();
                return all;
        }
    }

    private void ValidateInputs(SparseTensor latent, Tensor utonia, int n)
    {
        if (latent.Coords.Length != n * 4)
            throw new HartsyInferenceException($"Latent coords hold {latent.Coords.Length} values; expected {n} voxels of 4.");
        if (latent.Feats.DType != DType.F32 || latent.Feats.Shape.Rank != 2 || latent.Feats.Shape[0] != n
            || latent.Feats.Shape[1] != _config.LatentChannels)
            throw new HartsyInferenceException(
                $"Latent features have shape {latent.Feats.Shape}; expected F32 [{n}, {_config.LatentChannels}].");
        if (utonia.DType != DType.F32 || utonia.Shape.Rank != 2 || utonia.Shape[0] != n || utonia.Shape[1] != _config.UtoniaFeatureDim)
            throw new HartsyInferenceException(
                $"Utonia features have shape {utonia.Shape}; expected F32 [{n}, {_config.UtoniaFeatureDim}].");
    }

    /// <summary>Validates the prompt inputs for the configured source and returns the query count per sample. Point prompts
    /// are checked against the latent here, so a rejected request fails before any tensor is allocated.</summary>
    private int ValidatePrompts(CoLa3DQueryPrompts? prompts, SparseTensor latent, (int Start, int End)[] ranges)
    {
        int batch = ranges.Length;
        if (_config.QuerySource == CoLa3DConfig.QuerySourceKind.Fixed)
        {
            if (prompts is not null)
                throw new HartsyInferenceException("Fixed queries take no prompts; pass prompts only for a prompt query source.");
            return _config.MaxParts;
        }

        if (prompts is null)
            throw new HartsyInferenceException($"Query source {_config.QuerySource} requires prompts.");
        if (prompts.Batch != batch)
            throw new HartsyInferenceException($"Prompts describe {prompts.Batch} samples; the latent holds {batch}.");
        if (prompts.Queries < 1)
            throw new HartsyInferenceException("Prompts need at least one query slot per sample.");
        if (prompts.Queries > _config.MaxParts)
            throw new HartsyInferenceException($"Prompts name {prompts.Queries} queries; this checkpoint takes at most {_config.MaxParts}.");
        if (prompts.Valid is not null && prompts.Valid.Length != batch * prompts.Queries)
            throw new HartsyInferenceException($"Valid mask holds {prompts.Valid.Length} entries; expected {batch * prompts.Queries}.");

        if (_config.QuerySource == CoLa3DConfig.QuerySourceKind.PromptMask)
        {
            Tensor? masks = prompts.Masks;
            if (masks is null || prompts.Points is not null)
                throw new HartsyInferenceException("PromptMask queries take masks and no points.");
            if (masks.DType != DType.F32 || masks.Shape.Rank != 4 || masks.Shape[0] != batch || masks.Shape[1] != prompts.Queries)
                throw new HartsyInferenceException($"Prompt masks have shape {masks.Shape}; expected F32 [{batch}, {prompts.Queries}, H, W].");
            return prompts.Queries;
        }

        if (prompts.Points is null || prompts.Masks is not null)
            throw new HartsyInferenceException("PromptPoint queries take points and no masks.");
        if (prompts.Points.Length != batch * prompts.Queries * 3)
            throw new HartsyInferenceException($"Prompt points hold {prompts.Points.Length} values; expected {batch * prompts.Queries * 3}.");
        if (latent.Resolution != _config.LatentResolution)
            throw new HartsyInferenceException(
                $"Point prompts index a {_config.LatentResolution}^3 latent; this latent is {latent.Resolution}^3.");
        CoLa3DPointPromptEncoder.ValidatePoints(
            latent.Coords, ranges, prompts.Points, prompts.Valid, prompts.Queries, _config.LatentResolution);
        return prompts.Queries;
    }

    /// <summary>Contiguous voxel range of each sample; batch indices must run 0, 1, … with no gaps.</summary>
    private static (int Start, int End)[] BatchRanges(int[] coords, int n)
    {
        if (n == 0)
            throw new HartsyInferenceException("The latent holds no voxels.");
        List<(int Start, int End)> ranges = new();
        int current = 0, start = 0;
        if (coords[0] != 0)
            throw new HartsyInferenceException($"The first voxel belongs to batch {coords[0]}; batch indices must start at 0.");
        for (int r = 1; r < n; r++)
        {
            int b = coords[r * 4];
            if (b == current)
                continue;
            if (b != current + 1)
                throw new HartsyInferenceException(
                    $"Voxel {r} belongs to batch {b} after batch {current}; voxels must be grouped by contiguous batch index.");
            ranges.Add((start, r));
            current = b;
            start = r;
        }

        ranges.Add((start, n));
        return ranges.ToArray();
    }

    /// <summary>Slot validity flat <c>[Batch·Queries]</c>; every slot is active when the prompts carry no valid mask.</summary>
    private static bool[] ActiveMask(CoLa3DQueryPrompts? prompts, int batch, int queries)
    {
        bool[]? valid = prompts?.Valid;
        bool[] active = new bool[batch * queries];
        for (int i = 0; i < active.Length; i++)
            active[i] = valid is null || valid[i];
        return active;
    }

    /// <summary>Slot indices that take part for sample <paramref name="b"/>.</summary>
    private static int[] ActiveSlotIndices(bool[] active, int b, int queries)
    {
        List<int> slots = new();
        for (int q = 0; q < queries; q++)
        {
            if (active[b * queries + q])
                slots.Add(q);
        }

        return slots.ToArray();
    }

    /// <summary>RoPE positions of voxels <c>[start, start + count)</c>: their <c>(x, y, z)</c> coordinates.</summary>
    private static float[] PointPositions(int[] coords, int start, int count)
    {
        float[] positions = new float[count * 3];
        for (int r = 0; r < count; r++)
        {
            positions[r * 3] = coords[(start + r) * 4 + 1];
            positions[r * 3 + 1] = coords[(start + r) * 4 + 2];
            positions[r * 3 + 2] = coords[(start + r) * 4 + 3];
        }

        return positions;
    }

    /// <summary>RoPE positions of the active query slots: the slot index on all three axes for fixed and mask queries, the prompt
    /// voxel for point queries.</summary>
    private float[] SlotPositions(CoLa3DQueryPrompts? prompts, int b, int queries, int[] slots)
    {
        float[] positions = new float[slots.Length * 3];
        for (int i = 0; i < slots.Length; i++)
        {
            if (_config.QuerySource == CoLa3DConfig.QuerySourceKind.PromptPoint)
            {
                int at = (b * queries + slots[i]) * 3;
                positions[i * 3] = prompts!.Points![at];
                positions[i * 3 + 1] = prompts.Points[at + 1];
                positions[i * 3 + 2] = prompts.Points[at + 2];
            }
            else
            {
                positions[i * 3] = slots[i];
                positions[i * 3 + 1] = slots[i];
                positions[i * 3 + 2] = slots[i];
            }
        }

        return positions;
    }

    /// <summary>Writes <c>logits[s + r, slots[i]] = block[r, i]</c>; other entries stay zero.</summary>
    private static void ScatterLogits(Tensor block, int s, int nb, int[] slots, int queries, Tensor logits)
    {
        ReadOnlySpan<float> src = block.AsReadOnlySpan<float>();
        Span<float> dst = logits.AsSpan<float>();
        for (int r = 0; r < nb; r++)
        {
            for (int i = 0; i < slots.Length; i++)
                dst[(s + r) * queries + slots[i]] = src[r * slots.Length + i];
        }
    }
}
