using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Duo joint-media denoise pass on the H3 block stack, without a KV cache (PR 2f).
/// The graph is H3's block, token refiner and output head under the upstream names. What differs is the packed layout:
/// Duo rows arrive in any order, each with its own timestep index and modality tag, so the stream is gathered from the
/// three modality sources rather than spliced segment by segment. Padding rows stay zero, as upstream's zero buffer does.
/// RoPE comes from <see cref="DuoRope"/>, which casts positions to F32 before the angle, as upstream does.
/// The Duo attention bias is the square [S, S] mask of the live layout; the refiner has its own graph.</summary>
public sealed unsafe partial class MiniMaxH3Transformer
{
    /// <summary>The longest packed layout that takes one dense [S, S] bias (4·S² bytes). Longer layouts take the chunked route with a
    /// per-tile bias instead; the dense bias is never built past this length.</summary>
    internal const int MaxDuoDenseBiasRows = 32768;

    /// <summary>Byte budget for one query tile's bias ([rows, S] F32). Tile rows are this budget divided by 4·S, so a longer layout
    /// gets a narrower tile rather than a larger mask. 256 MiB is 1024 rows at S = 65536.</summary>
    internal const long DuoTileBudgetBytes = 256L * 1024 * 1024;

    /// <summary>The narrowest tile the budget may force. Below it the layout is refused rather than tiled into slivers.</summary>
    internal const int MinDuoTileRows = 64;

    /// <summary>The route one Duo forward takes: the chunk width attention runs at, and whether that width is a bias tile.</summary>
    internal readonly record struct DuoRoute(int ChunkRows, bool Tiled);

    /// <summary>Picks the route for a sequence of <paramref name="seq"/> rows given the policy's chunk width. A sequence that fits that
    /// width and the dense cap is dense. Anything longer is tiled, and the tile is the chunk width clamped to the byte budget for one
    /// [tile, seq] mask, so the mask lives inside the budget and the chunk policy's own choice is never exceeded.</summary>
    /// <exception cref="NotSupportedException">The budget leaves fewer than <see cref="MinDuoTileRows"/> rows per tile.</exception>
    internal static DuoRoute ResolveDuoRoute(int seq, int chunkRows)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seq);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkRows);
        bool tiled = seq > chunkRows || seq > MaxDuoDenseBiasRows;
        if (!tiled)
        {
            return new DuoRoute(chunkRows, false);
        }
        int budgetRows = DuoTileRowsFor(seq);
        if (budgetRows < MinDuoTileRows)
        {
            throw new NotSupportedException(
                $"A Duo layout of {seq} rows leaves {budgetRows} tile rows in the {DuoTileBudgetBytes}-byte mask budget; "
                + $"the minimum is {MinDuoTileRows}. Shorten the clip or the reference.");
        }
        return new DuoRoute(Math.Min(chunkRows, budgetRows), true);
    }

    /// <summary>Query rows one [rows, seq] F32 mask may take inside <see cref="DuoTileBudgetBytes"/>.</summary>
    internal static int DuoTileRowsFor(int seq) => (int)Math.Min(int.MaxValue, DuoTileBudgetBytes / ((long)seq * sizeof(float)));

    /// <summary>Per-forward tile masks: one reused [rows, seq] F32 buffer per tile height in use (the full tile and the shorter last one).
    /// Each call refills the buffer from the rectangles, so host work is O(rows·seq) per tile per layer, while memory stays at two tiles
    /// however long the layout is.</summary>
    private sealed class DuoTileMasks : IDisposable
    {
        private readonly IReadOnlyList<DuoMaskRect> _rects;
        private readonly int _seq;
        private readonly Dictionary<int, Tensor> _byRows = new();

        public DuoTileMasks(IReadOnlyList<DuoMaskRect> rects, int seq)
        {
            _rects = rects;
            _seq = seq;
        }

        /// <summary>Fills the tile of rows <c>[rowStart, rowStart + rows)</c> and returns it. Borrowed: valid until the next call or <see cref="Dispose"/>.</summary>
        public Tensor Tile(int rowStart, int rows)
        {
            if (!_byRows.TryGetValue(rows, out Tensor? tile))
            {
                tile = new Tensor(new TensorShape(rows, _seq), DType.F32);
                _byRows[rows] = tile;
            }
            DuoAttentionBias.BuildRowsInto(_rects, _seq, rowStart, rows, _seq, tile.AsSpan<float>());
            return tile;
        }

        public void Dispose()
        {
            foreach (Tensor tile in _byRows.Values)
            {
                tile.Dispose();
            }
            _byRows.Clear();
        }
    }

    /// <summary>Runs one Duo denoise evaluation over <paramref name="layout"/>.</summary>
    /// <param name="mode">The visibility rule: causal for a denoise against a committed history, bidirectional for a full-context pass.</param>
    /// <param name="refinerGroups">The text groups over the refiner's own rows (<paramref name="textStates"/>), in source coordinates.</param>
    /// <param name="textStates">Text conditioning rows [T, textDim] before <c>context_embedder</c> (upstream <c>refiner_text</c>).</param>
    /// <param name="videoRows">Patchified video rows [layout.VideoRows.Length, videoPatchDim], in <c>VideoRows</c> order.</param>
    /// <param name="audioRows">Audio rows [layout.AudioRows.Length, audioDim], in <c>AudioRows</c> order; null only when the layout has none.</param>
    /// <param name="times">The distinct timesteps in [0, 1], unscaled; <c>layout.TimeIds</c> indexes this array.</param>
    /// <param name="roleEmbeddingEnabled">Upstream keeps role embeddings disabled and zero-initialised. Passing true is refused, not ignored.</param>
    /// <param name="chunkRowsOverride">Test seam: forces the chunk width instead of the VRAM policy. A width below the sequence takes the
    /// tiled route, so the tiled bias can be checked against the dense bias on a small layout.</param>
    /// <remarks>A sequence that fits the chunk width and <see cref="MaxDuoDenseBiasRows"/> takes one dense [S, S] bias through the dense
    /// attention path. Any longer sequence takes the chunked route with a bias built per query tile, the tile sized by
    /// <see cref="DuoTileBudgetBytes"/>, so no S x S matrix is built. All inputs are validated before the first device allocation. The body is F32. The bias and index
    /// tensors are written through host pointers, as <c>BuildModulationIndex</c> does, so they must live where the host can write. The tiled
    /// bias is applied through the chunked path's score mask, which CPU SDPA reads; the GPU backends' mask support is not verified here.</remarks>
    internal (Tensor Video, Tensor? Audio) ForwardDuo(IBackend backend, DuoLayout layout, DuoAttentionMode mode,
        IReadOnlyList<DuoGroup> refinerGroups, Tensor textStates, Tensor videoRows, Tensor? audioRows,
        float[] times, bool roleEmbeddingEnabled = false, int? chunkRowsOverride = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(refinerGroups);
        ArgumentNullException.ThrowIfNull(textStates);
        ArgumentNullException.ThrowIfNull(videoRows);
        ArgumentNullException.ThrowIfNull(times);
        if (roleEmbeddingEnabled)
        {
            throw new NotSupportedException("Duo role embeddings are disabled upstream and are not ported; refusing rather than ignoring them.");
        }
        if (BodyDType != DType.F32)
        {
            throw new NotSupportedException("The Duo forward runs an F32 body; its zero-row gather is F32-only.");
        }
        if (layout.VideoRows.Length == 0)
        {
            throw new NotSupportedException("A Duo layout with no video rows is not supported; every target carries video.");
        }
        if (audioRows is null && layout.AudioRows.Length > 0)
        {
            throw new ArgumentException("The layout has audio rows, so audioRows must be given.", nameof(audioRows));
        }
        if (audioRows is not null && !_config.HasAudio)
        {
            throw new ArgumentException("This checkpoint has no audio stream, so audioRows must be null.", nameof(audioRows));
        }
        if (videoRows.Shape[0] != layout.VideoRows.Length)
        {
            throw new ArgumentException($"Video rows {videoRows.Shape[0]} do not match the layout's {layout.VideoRows.Length}.", nameof(videoRows));
        }
        if (audioRows is not null && audioRows.Shape[0] != layout.AudioRows.Length)
        {
            throw new ArgumentException($"Audio rows {audioRows.Shape[0]} do not match the layout's {layout.AudioRows.Length}.", nameof(audioRows));
        }
        if (times.Length == 0)
        {
            throw new ArgumentException("At least one timestep is needed.", nameof(times));
        }
        ValidateDuoRefinerGroups(refinerGroups, (int)textStates.Shape[0]);
        ValidateDuoTimeIds(layout, times.Length);
        if (chunkRowsOverride is int forcedRows)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(forcedRows);
        }

        int hidden = _config.HiddenSize;
        int seq = layout.Length;
        int headDim = _config.AttentionHeadDim;
        int textCount = layout.TextRows.Length, videoCount = layout.VideoRows.Length, audioCount = layout.AudioRows.Length;

        // Route first: one dense bias for a sequence that fits, a reused per-tile bias for anything longer. No S x S matrix is built
        // on the tiled route. Every allocation below is a `using`, so a throw disposes whatever was already built.
        (long freeBytes, _) = backend.GetVramInfo();
        int chunkRows = chunkRowsOverride ?? MiniMaxH3ChunkPolicy.ResolveChunkRows(seq, _config, BodyDType, freeBytes, backend);
        DuoRoute route = ResolveDuoRoute(seq, chunkRows);
        IReadOnlyList<DuoMaskRect> rects = DuoMask.Build(layout.Groups, layout.Groups, mode);
        DuoAttentionBias.ValidateRows(rects, seq, seq);
        using Tensor? bias = route.Tiled ? null : HostTensor(new TensorShape(seq, seq), DuoAttentionBias.BuildRows(rects, seq, 0, seq, seq));
        using DuoTileMasks? tiles = route.Tiled ? new DuoTileMasks(rects, seq) : null;
        Func<int, int, Tensor>? biasTile = null;
        if (tiles is not null)
        {
            biasTile = (r0, rows) => tiles.Tile(r0, rows);
        }
        using Tensor tEmb = BuildTimeEmbedding(backend, times);
        using Tensor modIndex = BuildDuoModulationIndex(layout);
        (float[] cosValues, float[] sinValues) = DuoRope.Tables(layout.Positions, headDim);
        using Tensor cos = HostTensor(new TensorShape(seq, headDim), cosValues);
        using Tensor sin = HostTensor(new TensorShape(seq, headDim), sinValues);

        Tensor stream = BuildDuoStream(backend, layout, refinerGroups, mode, textStates, videoRows, audioRows, hidden,
            textCount, videoCount, audioCount);
        try
        {
            for (int i = 0; i < _config.NumLayers; i++)
            {
                ForwardBlock(backend, ref stream, tEmb, modIndex, cos, sin, i, route.ChunkRows, null, bias, biasTile);
            }

            // norm_out: the shift/scale table has one row per timestep; every row is addressed by its own time index.
            Tensor normed = Norm(backend, stream, "final_layer.norm.weight", _config.FinalNormEps, DType.F32);
            try
            {
                Tensor[] headMod = Adaln(backend, tEmb, "final_layer.adaln_proj", expand: 2, modalities: 1);
                try
                {
                    using Tensor timeIndex = BuildDuoTimeIndex(layout);
                    using Tensor modulated = Modulate(backend, normed, headMod[0], headMod[1], timeIndex);
                    Tensor video = LinearDuoHead(backend, Gather(backend, modulated, layout.VideoRows), "final_layer.video_out");
                    try
                    {
                        Tensor? audio = audioCount > 0
                            ? LinearDuoHead(backend, Gather(backend, modulated, layout.AudioRows), "final_layer.audio_out")
                            : null;
                        return (video, audio);
                    }
                    catch
                    {
                        video.Dispose();
                        throw;
                    }
                }
                finally
                {
                    foreach (Tensor t in headMod) t.Dispose();
                }
            }
            finally
            {
                normed.Dispose();
            }
        }
        finally
        {
            stream.Dispose();
        }
    }

    /// <summary>Bounds-checks each refiner group against the text rows it names, before any allocation.</summary>
    private static void ValidateDuoRefinerGroups(IReadOnlyList<DuoGroup> refinerGroups, int textRows)
    {
        foreach (DuoGroup group in refinerGroups)
        {
            if (group.Start < 0 || group.End < group.Start || group.End > textRows)
            {
                throw new ArgumentException($"Refiner group [{group.Start}, {group.End}) lies outside the {textRows} text rows.", nameof(refinerGroups));
            }
        }
    }

    /// <summary>Refuses a timestep index beyond <paramref name="timestepCount"/>, before any allocation.</summary>
    private static void ValidateDuoTimeIds(DuoLayout layout, int timestepCount)
    {
        for (int r = 0; r < layout.Length; r++)
        {
            int time = layout.TimeIds[r];
            if (time < 0 || time >= timestepCount)
            {
                throw new ArgumentOutOfRangeException(nameof(layout), time, $"Row {r} names timestep {time}; only {timestepCount} were given.");
            }
        }
    }

    /// <summary>Refines the text rows, projects video and audio, and gathers all three into the packed stream [S, hidden].
    /// Source row <c>k</c> of the concatenation is text <c>k</c>, then video, then audio; the last source row is zero for padding.</summary>
    private Tensor BuildDuoStream(IBackend backend, DuoLayout layout, IReadOnlyList<DuoGroup> refinerGroups,
        DuoAttentionMode mode, Tensor textStates, Tensor videoRows, Tensor? audioRows, int hidden,
        int textCount, int videoCount, int audioCount)
    {
        using Tensor refined = RefineDuoText(backend, textStates, refinerGroups, mode);
        using Tensor? selected = textCount > 0 ? Gather(backend, refined, layout.TextSourceRows) : null;
        using Tensor videoEmbed = ProjectIntoStream(backend, videoRows, "video_patch_proj", hidden);
        using Tensor? audioEmbed = audioCount > 0 ? ProjectIntoStream(backend, audioRows!, "audio_patch_proj", hidden) : null;
        using Tensor zero = new Tensor(new TensorShape(1, hidden), DType.F32);
        zero.AsSpan<float>().Clear();

        List<Tensor> pieces = new List<Tensor>(4);
        if (selected is not null) pieces.Add(selected);
        if (videoCount > 0) pieces.Add(videoEmbed);
        if (audioEmbed is not null) pieces.Add(audioEmbed);
        pieces.Add(zero);

        int sourceRows = textCount + videoCount + audioCount + 1;
        using Tensor source = new Tensor(new TensorShape(sourceRows, hidden), DType.F32);
        backend.Concat(source, CollectionsMarshal.AsSpan(pieces), 0);

        int zeroRow = sourceRows - 1;
        int[] packed = new int[layout.Length];
        Array.Fill(packed, zeroRow);
        for (int k = 0; k < textCount; k++) packed[layout.TextRows[k]] = k;
        for (int k = 0; k < videoCount; k++) packed[layout.VideoRows[k]] = textCount + k;
        for (int k = 0; k < audioCount; k++) packed[layout.AudioRows[k]] = textCount + videoCount + k;
        return Gather(backend, source, packed);
    }

    /// <summary>The Duo token refiner: context_embedder (H3's condition_proj), then the masked refiner blocks, then final_norm.
    /// The refiner graph is its own: upstream builds it from <paramref name="refinerGroups"/>, not from the packed layout.
    /// Empty text returns no rows, as upstream skips the refiner for it.</summary>
    private Tensor RefineDuoText(IBackend backend, Tensor textStates, IReadOnlyList<DuoGroup> refinerGroups,
        DuoAttentionMode mode)
    {
        int hidden = _config.HiddenSize;
        int rows = (int)textStates.Shape[0];
        Tensor x = ProjectIntoStream(backend, textStates, "condition_proj", hidden);
        if (rows == 0)
        {
            return x;
        }

        try
        {
            using Tensor? bias = _config.TokenRefinerNumLayers > 0
                ? HostTensor(new TensorShape(rows, rows),
                    DuoAttentionBias.Build(DuoMask.Build(refinerGroups, refinerGroups, mode, refiner: true), rows, rows))
                : null;
            for (int i = 0; i < _config.TokenRefinerNumLayers; i++)
            {
                string p = $"token_refiner.blocks.{i}";
                using Tensor normed = Norm(backend, x, $"{p}.norm1.weight", _config.NormEps, BodyDType);
                Tensor attn = Attention(backend, normed, $"{p}.attn", cos: null, sin: null, bias);
                x = Residual(backend, x, attn);

                using Tensor normed2 = Norm(backend, x, $"{p}.norm2.weight", _config.NormEps, BodyDType);
                Tensor mlp = Mlp(backend, normed2, $"{p}.mlp");
                x = Residual(backend, x, mlp);
            }

            return Norm(backend, x, "token_refiner.final_norm.weight", _config.FinalNormEps, BodyDType);
        }
        finally
        {
            x.Dispose();
        }
    }

    /// <summary>Per-row AdaLN index <c>timeIds * 3 + max(tag, 0)</c>, the row layout the block tables are built for.
    /// Same definition as <see cref="DuoLayout.AdalnRow"/>; the layout's time ids must already be validated.</summary>
    private static Tensor BuildDuoModulationIndex(DuoLayout layout)
    {
        Tensor index = new Tensor(new TensorShape(layout.Length), DType.I32);
        int* p = (int*)index.DataPointer;
        for (int r = 0; r < layout.Length; r++)
        {
            p[r] = layout.AdalnRow(r);
        }
        return index;
    }

    /// <summary>Per-row time index for norm_out, which has one table row per timestep and no modality.</summary>
    private static Tensor BuildDuoTimeIndex(DuoLayout layout)
    {
        Tensor index = new Tensor(new TensorShape(layout.Length), DType.I32);
        int* p = (int*)index.DataPointer;
        for (int r = 0; r < layout.Length; r++)
        {
            p[r] = layout.TimeIds[r];
        }
        return index;
    }

    /// <summary>Gathers the rows at <paramref name="rows"/> into a fresh [rows.Length, cols] tensor of <paramref name="src"/>'s dtype.</summary>
    private static Tensor Gather(IBackend backend, Tensor src, int[] rows)
    {
        Tensor outT = new Tensor(new TensorShape(rows.Length, src.Shape[1]), src.DType);
        backend.GatherRows(outT, src, rows);
        return outT;
    }

    /// <summary>Output head: a bias-carrying linear from the stream to one modality's patch width. Takes ownership of <paramref name="x"/>.</summary>
    private Tensor LinearDuoHead(IBackend backend, Tensor x, string prefix)
    {
        try
        {
            Tensor weight = Require($"{prefix}.weight");
            Tensor outT = new Tensor(new TensorShape(x.Shape[0], weight.Shape[0]), DType.F32);
            backend.Linear(outT, x, weight, Optional($"{prefix}.bias"));
            return outT;
        }
        finally
        {
            x.Dispose();
        }
    }

    /// <summary>A host-written F32 tensor with the given values (row-major, length must equal the shape's element count).</summary>
    private static Tensor HostTensor(TensorShape shape, float[] values)
    {
        Tensor t = new Tensor(shape, DType.F32);
        values.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }
}
