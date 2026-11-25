using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Duo joint-media denoise pass on the H3 block stack, without a KV cache (PR 2f).
/// The graph is H3's block, token refiner and output head under the upstream names. What differs is the packed layout:
/// Duo rows arrive in any order, each with its own timestep index and modality tag, so the stream is gathered from the
/// three modality sources rather than spliced segment by segment. Padding rows stay zero, as upstream's zero buffer does.
/// The Duo attention bias is the square [S, S] mask of the live layout; the refiner has its own graph.</summary>
public sealed unsafe partial class MiniMaxH3Transformer
{
    /// <summary>Runs one Duo denoise evaluation over <paramref name="layout"/>.</summary>
    /// <param name="mode">The visibility rule: causal for a denoise against a committed history, bidirectional for a full-context pass.</param>
    /// <param name="refinerGroups">The text groups over the refiner's own rows (<paramref name="textStates"/>), in source coordinates.</param>
    /// <param name="textStates">Text conditioning rows [T, textDim] before <c>context_embedder</c> (upstream <c>refiner_text</c>).</param>
    /// <param name="videoRows">Patchified video rows [layout.VideoRows.Length, videoPatchDim], in <c>VideoRows</c> order.</param>
    /// <param name="audioRows">Audio rows [layout.AudioRows.Length, audioDim], in <c>AudioRows</c> order; null only when the layout has none.</param>
    /// <param name="times">The distinct timesteps in [0, 1], unscaled; <c>layout.TimeIds</c> indexes this array.</param>
    /// <param name="roleEmbeddingEnabled">Upstream keeps role embeddings disabled and zero-initialised. Passing true is refused, not ignored.</param>
    /// <remarks>Every Duo forward is a bias forward: the dense attention path runs with the bias, so a sequence longer than the chunk
    /// width is refused (see <see cref="CheckBiasedRoute"/>). The body is F32. The bias and index tensors are written through host pointers,
    /// as <c>BuildModulationIndex</c> does, so they must live where the host can write.</remarks>
    internal (Tensor Video, Tensor? Audio) ForwardDuo(IBackend backend, DuoLayout layout, DuoAttentionMode mode,
        IReadOnlyList<DuoGroup> refinerGroups, Tensor textStates, Tensor videoRows, Tensor? audioRows,
        float[] times, bool roleEmbeddingEnabled = false)
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
        if (audioRows is null ? layout.AudioRows.Length > 0 : !_config.HasAudio)
        {
            throw new ArgumentException("Audio rows must be given exactly when the checkpoint has an audio stream and the layout has audio rows.", nameof(audioRows));
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

        int hidden = _config.HiddenSize;
        int seq = layout.Length;
        int textCount = layout.TextRows.Length, videoCount = layout.VideoRows.Length, audioCount = layout.AudioRows.Length;

        Tensor h = BuildDuoStream(backend, layout, refinerGroups, mode, textStates, videoRows, audioRows, hidden, textCount, videoCount, audioCount);
        Tensor tEmb = BuildTimeEmbedding(backend, times);
        Tensor modIndex = BuildDuoModulationIndex(layout, times.Length);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(layout.Positions,
            MiniMaxH3Rope.DefaultInvFreq(_config.RopeInvFreqLen), _config.AttentionHeadDim);
        Tensor bias = HostTensor(new TensorShape(seq, seq),
            DuoAttentionBias.Build(DuoMask.Build(layout.Groups, layout.Groups, mode), seq, seq));
        try
        {
            (long freeBytes, _) = backend.GetVramInfo();
            int chunkRows = MiniMaxH3ChunkPolicy.ResolveChunkRows(seq, _config, BodyDType, freeBytes, backend);
            CheckBiasedRoute(seq, chunkRows, freeBytes, sparse: false, chunkForced: MiniMaxH3ChunkPolicy.ChunkRowsForced());
            for (int i = 0; i < _config.NumLayers; i++)
            {
                ForwardBlock(backend, ref h, tEmb, modIndex, cos, sin, i, chunkRows, null, bias);
            }

            // norm_out: the shift/scale table has one row per timestep; every row is addressed by its own time index.
            Tensor normed = Norm(backend, h, "final_layer.norm.weight", _config.FinalNormEps, DType.F32);
            Tensor[] headMod = Adaln(backend, tEmb, "final_layer.adaln_proj", expand: 2, modalities: 1);
            try
            {
                using Tensor timeIndex = BuildDuoTimeIndex(layout, times.Length);
                Tensor modulated = Modulate(backend, normed, headMod[0], headMod[1], timeIndex);
                normed.Dispose();
                try
                {
                    Tensor video = LinearDuoHead(backend, Gather(backend, modulated, layout.VideoRows), "final_layer.video_out");
                    Tensor? audio = audioCount > 0
                        ? LinearDuoHead(backend, Gather(backend, modulated, layout.AudioRows), "final_layer.audio_out")
                        : null;
                    return (video, audio);
                }
                finally
                {
                    modulated.Dispose();
                }
            }
            finally
            {
                foreach (Tensor t in headMod) t.Dispose();
            }
        }
        finally
        {
            h.Dispose();
            bias.Dispose();
            cos.Dispose();
            sin.Dispose();
            modIndex.Dispose();
            tEmb.Dispose();
        }
    }

    /// <summary>Refines the text rows, projects video and audio, and gathers all three into the packed stream [S, hidden].
    /// Source row <c>k</c> of the concatenation is text <c>k</c>, then video, then audio; the last source row is zero for padding.</summary>
    private Tensor BuildDuoStream(IBackend backend, DuoLayout layout, IReadOnlyList<DuoGroup> refinerGroups,
        DuoAttentionMode mode, Tensor textStates, Tensor videoRows, Tensor? audioRows, int hidden,
        int textCount, int videoCount, int audioCount)
    {
        Tensor refined = RefineDuoText(backend, textStates, refinerGroups, mode);
        Tensor selected = textCount > 0 ? Gather(backend, refined, layout.TextSourceRows) : refined;
        Tensor videoEmbed = ProjectIntoStream(backend, videoRows, "video_patch_proj", hidden);
        Tensor? audioEmbed = audioCount > 0 ? ProjectIntoStream(backend, audioRows!, "audio_patch_proj", hidden) : null;
        Tensor zero = new Tensor(new TensorShape(1, hidden), DType.F32);
        zero.AsSpan<float>().Clear();
        try
        {
            List<Tensor> pieces = new List<Tensor>(4);
            if (textCount > 0) pieces.Add(selected);
            if (videoCount > 0) pieces.Add(videoEmbed);
            if (audioEmbed is not null) pieces.Add(audioEmbed);
            pieces.Add(zero);

            int sourceRows = textCount + videoCount + audioCount + 1;
            Tensor source = new Tensor(new TensorShape(sourceRows, hidden), DType.F32);
            backend.Concat(source, CollectionsMarshal.AsSpan(pieces), 0);
            try
            {
                int zeroRow = sourceRows - 1;
                int[] packed = new int[layout.Length];
                Array.Fill(packed, zeroRow);
                for (int k = 0; k < textCount; k++) packed[layout.TextRows[k]] = k;
                for (int k = 0; k < videoCount; k++) packed[layout.VideoRows[k]] = textCount + k;
                for (int k = 0; k < audioCount; k++) packed[layout.AudioRows[k]] = textCount + videoCount + k;
                return Gather(backend, source, packed);
            }
            finally
            {
                source.Dispose();
            }
        }
        finally
        {
            zero.Dispose();
            videoEmbed.Dispose();
            audioEmbed?.Dispose();
            if (textCount > 0) selected.Dispose();
            refined.Dispose();
        }
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

        Tensor? bias = _config.TokenRefinerNumLayers > 0
            ? HostTensor(new TensorShape(rows, rows),
                DuoAttentionBias.Build(DuoMask.Build(refinerGroups, refinerGroups, mode, refiner: true), rows, rows))
            : null;
        try
        {
            for (int i = 0; i < _config.TokenRefinerNumLayers; i++)
            {
                string p = $"token_refiner.blocks.{i}";
                Tensor normed = Norm(backend, x, $"{p}.norm1.weight", _config.NormEps, BodyDType);
                Tensor attn = Attention(backend, normed, $"{p}.attn", cos: null, sin: null, bias);
                normed.Dispose();
                x = Residual(backend, x, attn);

                Tensor normed2 = Norm(backend, x, $"{p}.norm2.weight", _config.NormEps, BodyDType);
                Tensor mlp = Mlp(backend, normed2, $"{p}.mlp");
                normed2.Dispose();
                x = Residual(backend, x, mlp);
            }
        }
        finally
        {
            bias?.Dispose();
        }

        Tensor final = Norm(backend, x, "token_refiner.final_norm.weight", _config.FinalNormEps, BodyDType);
        x.Dispose();
        return final;
    }

    /// <summary>Per-row AdaLN index <c>timeIds * 3 + max(tag, 0)</c>, the row layout the block tables are built for.</summary>
    private static Tensor BuildDuoModulationIndex(DuoLayout layout, int timestepCount)
    {
        Tensor index = new Tensor(new TensorShape(layout.Length), DType.I32);
        int* p = (int*)index.DataPointer;
        for (int r = 0; r < layout.Length; r++)
        {
            int time = layout.TimeIds[r];
            if (time < 0 || time >= timestepCount)
            {
                index.Dispose();
                throw new ArgumentOutOfRangeException(nameof(layout), time, $"Row {r} names timestep {time}; only {timestepCount} exist.");
            }
            p[r] = time * 3 + Math.Max(layout.Tags[r], 0);
        }
        return index;
    }

    /// <summary>Per-row time index for norm_out, which has one table row per timestep and no modality.</summary>
    private static Tensor BuildDuoTimeIndex(DuoLayout layout, int timestepCount)
    {
        Tensor index = new Tensor(new TensorShape(layout.Length), DType.I32);
        int* p = (int*)index.DataPointer;
        for (int r = 0; r < layout.Length; r++)
        {
            int time = layout.TimeIds[r];
            if (time < 0 || time >= timestepCount)
            {
                index.Dispose();
                throw new ArgumentOutOfRangeException(nameof(layout), time, $"Row {r} names timestep {time}; only {timestepCount} exist.");
            }
            p[r] = time;
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

    /// <summary>Output head: a bias-carrying linear from the stream to one modality's patch width.</summary>
    private Tensor LinearDuoHead(IBackend backend, Tensor x, string prefix)
    {
        Tensor weight = Require($"{prefix}.weight");
        Tensor outT = new Tensor(new TensorShape(x.Shape[0], weight.Shape[0]), DType.F32);
        backend.Linear(outT, x, weight, Optional($"{prefix}.bias"));
        x.Dispose();
        return outT;
    }

    /// <summary>A host-written F32 tensor with the given values (row-major, length must equal the shape's element count).</summary>
    private static Tensor HostTensor(TensorShape shape, float[] values)
    {
        Tensor t = new Tensor(shape, DType.F32);
        values.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }
}
