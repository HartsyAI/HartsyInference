using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>Budget3D on Idol (upstream <c>Budget3DModel</c>): the denoiser run on an adaptive number of latent tokens. A <see cref="BudgetLayout"/> chooses which regions keep one token per Sobol point and which are merged into
/// 16³ cells or 8³ cubes; the trunk (Idol's own noise refiner and 24 joint blocks, <see cref="IdolDenoiser.ForwardTokens"/>) runs on the shorter sequence, and token-local refiners expand every merged token's single velocity back
/// into one velocity per Sobol point, so the output is always the full <c>[1, 8192, 16]</c> latent velocity Idol's decoders expect. The dense layout takes Idol's own path. Borrows the trunk and the weights.</summary>
public sealed class BudgetDenoiser : IDisposable
{
    private const float HfRho = 0.25f;

    private readonly IdolDenoiser _trunk;
    private readonly BudgetAdapterWeights _adapters;
    private readonly BudgetScaleMixer _cellMixer, _cubeMixer;
    private readonly BudgetIpe _ipe;
    private readonly BudgetRefineStage _cellRefiner, _cubeRefiner;
    private readonly int[] _toOctree, _fromOctree, _cubeToCells;
    private readonly float[] _cellCentres, _cubeCentres;
    private int _disposed;

    /// <param name="trunk">The denoiser over the Budget3D checkpoint's backbone (the same key names as Idol's).</param>
    /// <param name="adapters">The checkpoint's <c>bp_*</c> tensors.</param>
    public BudgetDenoiser(IdolDenoiser trunk, BudgetAdapterWeights adapters)
    {
        _trunk = trunk ?? throw new ArgumentNullException(nameof(trunk));
        _adapters = adapters ?? throw new ArgumentNullException(nameof(adapters));
        _cellMixer = new BudgetScaleMixer(adapters, 2);
        _cubeMixer = new BudgetScaleMixer(adapters, 8);
        _ipe = new BudgetIpe(adapters);
        _cellRefiner = new BudgetRefineStage(adapters, 2);
        _cubeRefiner = new BudgetRefineStage(adapters, 8);
        (_toOctree, _fromOctree) = BudgetOctree.Permutation(IdolWeights.EmbeddedPositions());
        _cubeToCells = BudgetOctree.CubeToCells();
        _cellCentres = BudgetOctree.CellCenters();
        _cubeCentres = BudgetOctree.CubeCenters();
    }

    /// <summary>The tensors the adapters hand to the backend, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c> (the trunk lists its own).</summary>
    public IEnumerable<Tensor> EnumerateWeights()
        => _cellMixer.EnumerateWeights().Concat(_cubeMixer.EnumerateWeights()).Concat(_ipe.EnumerateWeights()).Concat(_cellRefiner.EnumerateWeights()).Concat(_cubeRefiner.EnumerateWeights());

    /// <summary>The velocity of <paramref name="latent"/> <c>[1, 8192, 16]</c> and <paramref name="camera"/> <c>[1, 1, 5]</c> at the model time <paramref name="t"/> (1000 times the flow time) on <paramref name="layout"/>,
    /// conditioned on <paramref name="context"/> from <see cref="IdolDenoiser.EmbedContext"/>. The caller owns both results.</summary>
    public (Tensor Latent, Tensor Camera) Forward(IBackend backend, Tensor latent, Tensor camera, float t, Tensor context, BudgetLayout layout)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(layout);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        IdolConfig config = _adapters.Config;
        if (latent.Shape.Rank != 3 || latent.Shape[0] != 1 || latent.Shape[1] != BudgetOctree.Points || latent.Shape[2] != config.LatentChannels)
            throw new ArgumentException($"Budget3D takes a [1, {BudgetOctree.Points}, {config.LatentChannels}] latent; got {latent.Shape}.", nameof(latent));

        Tensor embedded = _trunk.EmbedLatent(backend, latent);
        if (layout.IsDense)
        {
            (Tensor v, Tensor c, Tensor? unused) = _trunk.ForwardTokens(backend, embedded, camera, t, context, wantHidden: false);
            unused?.Dispose();
            return (v, c);
        }

        Tensor tokens = EmbedTokens(backend, embedded, layout);
        (Tensor velocity, Tensor cameraVelocity, Tensor? hidden) = _trunk.ForwardTokens(backend, tokens, camera, t, context, wantHidden: true);
        try
        {
            Tensor expanded = Expand(backend, velocity.AsReadOnlySpan<float>(), hidden!.AsReadOnlySpan<float>(), latent.AsReadOnlySpan<float>(), t, layout, config);
            return (expanded, cameraVelocity);
        }
        catch
        {
            cameraVelocity.Dispose();
            throw;
        }
        finally
        {
            velocity.Dispose();
            hidden?.Dispose();
        }
    }

    // The token sequence [s = 4 | s = 2 | s = 1] of a layout from the dense embedding (consumed): every cell and cube gets its mixed feature, the layout picks the rows, and the merged tokens add their scale's
    // position encoding and every token its scale's embedding row.
    private Tensor EmbedTokens(IBackend backend, Tensor dense, BudgetLayout layout)
    {
        int d = _adapters.Config.ModelChannels;
        float[] octree = new float[BudgetOctree.Points * d];
        using (dense)
        {
            ReadOnlySpan<float> h = dense.AsReadOnlySpan<float>();
            for (int k = 0; k < BudgetOctree.Points; k++)
                h.Slice(_toOctree[k] * d, d).CopyTo(octree.AsSpan(k * d, d));
        }

        float[] cells = _cellMixer.Apply(backend, octree, BudgetOctree.Cells, d);
        float[] cubeChildren = new float[BudgetOctree.Cubes * 8 * d];
        for (int j = 0; j < BudgetOctree.Cubes; j++)
            for (int o = 0; o < 8; o++)
                cells.AsSpan(_cubeToCells[j * 8 + o] * d, d).CopyTo(cubeChildren.AsSpan((j * 8 + o) * d, d));
        float[] cubes = _cubeMixer.Apply(backend, cubeChildren, BudgetOctree.Cubes, d);

        float[] tokens = new float[layout.Tokens * d];
        int row = 0;
        foreach (int j in layout.Cubes)
            cubes.AsSpan(j * d, d).CopyTo(tokens.AsSpan(row++ * d, d));
        foreach (int c in layout.Cells)
            cells.AsSpan(c * d, d).CopyTo(tokens.AsSpan(row++ * d, d));
        foreach (int c in layout.Pairs)
        {
            octree.AsSpan(c * 2 * d, 2 * d).CopyTo(tokens.AsSpan(row * d, 2 * d));
            row += 2;
        }

        if (layout.CubeTokens > 0)
            AddRows(tokens, 0, _ipe.Encode(backend, Gather(_cubeCentres, layout.Cubes, 3), layout.CubeTokens, 4, d), layout.CubeTokens, d);
        if (layout.CellTokens > 0)
            AddRows(tokens, layout.CubeTokens, _ipe.Encode(backend, Gather(_cellCentres, layout.Cells, 3), layout.CellTokens, 2, d), layout.CellTokens, d);
        ReadOnlySpan<float> scaleRows = _adapters["bp_scale_emb"].AsReadOnlySpan<float>();
        AddScaleRow(tokens, 0, layout.CubeTokens, scaleRows.Slice(BudgetOctree.ScaleRow(4) * d, d), d);
        AddScaleRow(tokens, layout.CubeTokens, layout.CellTokens, scaleRows.Slice(BudgetOctree.ScaleRow(2) * d, d), d);
        AddScaleRow(tokens, layout.CubeTokens + layout.CellTokens, layout.PointTokens, scaleRows.Slice(BudgetOctree.ScaleRow(1) * d, d), d);
        return BudgetHost.Make(tokens, 1, layout.Tokens, d);
    }

    private static float[] Gather(float[] table, int[] rows, int width)
    {
        float[] gathered = new float[rows.Length * width];
        for (int i = 0; i < rows.Length; i++)
            Array.Copy(table, rows[i] * width, gathered, i * width, width);
        return gathered;
    }

    private static void AddRows(float[] tokens, int firstRow, float[] addend, int count, int d)
    {
        for (int i = 0; i < count * d; i++)
            tokens[firstRow * d + i] += addend[i];
    }

    private static void AddScaleRow(float[] tokens, int firstRow, int count, ReadOnlySpan<float> row, int d)
    {
        for (int i = 0; i < count; i++)
            for (int c = 0; c < d; c++)
                tokens[(firstRow + i) * d + c] += row[c];
    }

    // Merged-token velocities to one velocity per Sobol point (upstream _expand): s = 1 tokens are the released prediction; an s = 2 token's two points come from the cell refiner; an s = 4 token goes through the
    // cube refiner to its eight cells and each cell through the cell refiner, carrying the first stage's hidden state. A Wiener gain then adds back the within-region detail the noisy latent already shows.
    private Tensor Expand(IBackend backend, ReadOnlySpan<float> velocity, ReadOnlySpan<float> hidden, ReadOnlySpan<float> latent, float t, BudgetLayout layout, IdolConfig config)
    {
        int c = config.LatentChannels, d = config.ModelChannels, n4 = layout.CubeTokens, n2 = layout.CellTokens;
        // The latent in octree order, [4096, 2, c].
        float[] zOct = new float[BudgetOctree.Points * c];
        for (int k = 0; k < BudgetOctree.Points; k++)
            latent.Slice(_toOctree[k] * c, c).CopyTo(zOct.AsSpan(k * c, c));
        float[] points = new float[BudgetOctree.Points * c];
        float time = t / 1000f;
        float gain = time / MathF.Max(time * time + HfRho * HfRho * (1f - time) * (1f - time), 1e-6f);

        // s = 1: the released prediction, two tokens (slots) per opened cell.
        for (int i = 0; i < layout.K2; i++)
            velocity.Slice((n4 + n2 + 2 * i) * c, 2 * c).CopyTo(points.AsSpan(layout.Pairs[i] * 2 * c, 2 * c));

        // s = 2: one cell to its two points.
        if (n2 > 0)
        {
            float[] noisy = new float[n2 * 2 * c];
            for (int i = 0; i < n2; i++)
                Array.Copy(zOct, layout.Cells[i] * 2 * c, noisy, i * 2 * c, 2 * c);
            (float[] v, _) = _cellRefiner.Forward(backend, velocity.Slice(n4 * c, n2 * c), n2, hidden.Slice(n4 * d, n2 * d), d, 1, BudgetOctree.CellChildOffsets(), noisy, default);
            for (int i = 0; i < n2; i++)
                for (int ch = 0; ch < c; ch++)
                {
                    float mean = 0.5f * (noisy[i * 2 * c + ch] + noisy[i * 2 * c + c + ch]);
                    for (int slot = 0; slot < 2; slot++)
                        v[(i * 2 + slot) * c + ch] += gain * (noisy[(i * 2 + slot) * c + ch] - mean);
                }

            for (int i = 0; i < n2; i++)
                Array.Copy(v, i * 2 * c, points, layout.Cells[i] * 2 * c, 2 * c);
        }

        // s = 4: one cube to its eight cells (cube refiner), each cell to its two points (cell refiner, with the carried state).
        if (n4 > 0)
        {
            int cells = n4 * 8;
            float[] noisyCells = new float[cells * 2 * c], cellMeans = new float[cells * c];
            for (int i = 0; i < n4; i++)
                for (int o = 0; o < 8; o++)
                {
                    int cell = _cubeToCells[layout.Cubes[i] * 8 + o], r = i * 8 + o;
                    Array.Copy(zOct, cell * 2 * c, noisyCells, r * 2 * c, 2 * c);
                    for (int ch = 0; ch < c; ch++)
                        cellMeans[r * c + ch] = 0.5f * (zOct[cell * 2 * c + ch] + zOct[cell * 2 * c + c + ch]);
                }

            (float[] cellV, float[] cellH) = _cubeRefiner.Forward(backend, velocity[..(n4 * c)], n4, hidden[..(n4 * d)], d, 1, BudgetOctree.CubeChildOffsets(), cellMeans, default);
            (float[] v, _) = _cellRefiner.Forward(backend, cellV, cells, hidden[..(n4 * d)], d, 8, BudgetOctree.CellChildOffsets(), noisyCells, ParentRows(cellH, cells));
            for (int i = 0; i < n4; i++)
            {
                // The merged region is the whole cube: subtract the mean over its sixteen points.
                float[] cubeMean = new float[c];
                for (int o = 0; o < 8; o++)
                    for (int slot = 0; slot < 2; slot++)
                        for (int ch = 0; ch < c; ch++)
                            cubeMean[ch] += noisyCells[((i * 8 + o) * 2 + slot) * c + ch];
                for (int ch = 0; ch < c; ch++)
                    cubeMean[ch] /= 16f;
                for (int o = 0; o < 8; o++)
                    for (int slot = 0; slot < 2; slot++)
                        for (int ch = 0; ch < c; ch++)
                            v[((i * 8 + o) * 2 + slot) * c + ch] += gain * (noisyCells[((i * 8 + o) * 2 + slot) * c + ch] - cubeMean[ch]);
            }

            for (int i = 0; i < n4; i++)
                for (int o = 0; o < 8; o++)
                    Array.Copy(v, (i * 8 + o) * 2 * c, points, _cubeToCells[layout.Cubes[i] * 8 + o] * 2 * c, 2 * c);
        }

        // Back to Idol's token order: the point at octree slot k is Sobol point _toOctree[k]; latent velocity of point p is the slot _fromOctree[p].
        Tensor result = new(new TensorShape(1, BudgetOctree.Points, c), DType.F32);
        Span<float> output = result.AsSpan<float>();
        for (int p = 0; p < BudgetOctree.Points; p++)
            points.AsSpan(_fromOctree[p] * c, c).CopyTo(output.Slice(p * c, c));
        return result;
    }

    // The first stage's hidden state of each cell, [cells, 8 → ...]: the carry the second stage takes is the hidden of that cell's token, [cells × width].
    private static float[] ParentRows(float[] hidden, int cells) => hidden.Length == cells * BudgetAdapterWeights.RefineWidth
        ? hidden
        : throw new HartsyInferenceException($"The cube refiner's hidden state has {hidden.Length} values for {cells} cells.");

    /// <summary>Marks the denoiser disposed; the trunk and the weights belong to their owners.</summary>
    public void Dispose() => Interlocked.Exchange(ref _disposed, 1);
}
