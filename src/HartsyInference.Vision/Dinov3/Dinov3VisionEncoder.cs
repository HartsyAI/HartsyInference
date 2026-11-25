using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Dinov3;

/// <summary>DINOv3 ViT backbone with rotary patch positions. Produces <c>[B, 1 + storage + patches, hidden]</c> in the
/// reference order (CLS, storage tokens, patch tokens) after the final LayerNorm applied to every token. Patch tokens
/// get RoPE inside attention; CLS and storage tokens are not rotated and carry no position embedding.
/// <para>Load a checkpoint with <see cref="LoadWeights"/> from the HF DINOv3 layout (<c>embeddings.*</c>, <c>layer.N.*</c>,
/// <c>norm.*</c>), using a preset from <see cref="Dinov2Preset.FromDinov3Weights"/>. Weights are borrowed; the loader owns
/// them. Input is ImageNet-normalized pixels from <see cref="Dinov2ImagePreprocessor"/>.</para></summary>
public sealed unsafe class Dinov3VisionEncoder
{
    private const int HeadDim = 64;

    private readonly Dinov2Preset _preset;
    private readonly Dinov3Layer[] _layers;

    private Tensor? _patchW, _patchB;   // [hidden,3,p,p], [hidden]
    private Tensor? _clsToken;          // [1,1,hidden]
    private Tensor? _storageTokens;     // [1,storage,hidden]
    private Tensor? _finalLnW, _finalLnB;

    public Dinov3VisionEncoder(Dinov2Preset preset)
    {
        ArgumentNullException.ThrowIfNull(preset);
        if (!preset.UsesRotaryPositionEmbedding)
            throw new ArgumentException("Dinov3VisionEncoder takes a DINOv3 (RoPE) preset; use Dinov2VisionEncoder for DINOv2.", nameof(preset));
        if (preset.HiddenSize % preset.NumHeads != 0 || preset.HiddenSize / preset.NumHeads != HeadDim)
            throw new ArgumentException($"DINOv3 heads are {HeadDim} wide; got hidden {preset.HiddenSize} over {preset.NumHeads} heads.", nameof(preset));
        _preset = preset;
        _layers = new Dinov3Layer[preset.NumLayers];
        for (int i = 0; i < preset.NumLayers; i++)
            _layers[i] = new Dinov3Layer(preset.HiddenSize, preset.NumHeads, preset.IntermediateSize, preset.LayerNormEps, preset.GatedMlp, preset.AttentionQkvBias);
    }

    public Dinov2Preset Preset => _preset;
    public int HiddenSize => _preset.HiddenSize;

    /// <summary>Tokens per image at <see cref="Dinov2Preset.ImageSize"/>: CLS, storage tokens and patches.</summary>
    public int SequenceLength => 1 + _preset.NumRegisterTokens + _preset.NumPatches;

    /// <summary>Loads the HF-layout DINOv3 tensors under <paramref name="prefix"/>; shapes are checked against the preset.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        string p = prefix.Length == 0 ? "" : prefix + ".";
        int hidden = _preset.HiddenSize, storage = _preset.NumRegisterTokens;
        _patchW = F32(Require(weights, $"{p}embeddings.patch_embeddings.weight", hidden, 3, _preset.PatchSize, _preset.PatchSize));
        _patchB = F32(Require(weights, $"{p}embeddings.patch_embeddings.bias", hidden));
        _clsToken = F32(Require(weights, $"{p}embeddings.cls_token", 1, 1, hidden));
        _storageTokens = F32(Require(weights, $"{p}embeddings.register_tokens", 1, storage, hidden));
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(weights, $"{p}layer.{i}");
        if (_preset.HasFinalNorm)
        {
            _finalLnW = F32(Require(weights, $"{p}norm.weight", hidden));
            _finalLnB = F32(Require(weights, $"{p}norm.bias", hidden));
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] head = [_patchW, _patchB, _clsToken, _storageTokens, _finalLnW, _finalLnB];
        foreach (Tensor? t in head) if (t is not null) yield return t;
        foreach (Dinov3Layer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes pixels <c>[B,3,H,W]</c> (H and W multiples of the patch size) into <c>[B, S, hidden]</c> with the final
    /// LayerNorm applied to every token, or the last block's output as it is when the preset has no final norm
    /// (<see cref="Dinov2Preset.HasFinalNorm"/>).</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        Tensor h = EncodeBlocks(backend, pixelValues, _layers.Length - 1);
        if (!_preset.HasFinalNorm) return h;
        Tensor normed = new(h.Shape, DType.F32);
        backend.LayerNorm(normed, h, _finalLnW!, _finalLnB!, _preset.LayerNormEps);
        h.Dispose();
        return normed;
    }

    /// <summary>The token sequence after blocks <c>0..lastBlock</c>, before the final norm; <paramref name="lastBlock"/> of -1
    /// returns the embeddings alone. For parity taps; the caller owns the result.</summary>
    public Tensor EncodeBlocks(IBackend backend, Tensor pixelValues, int lastBlock)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (lastBlock < -1 || lastBlock >= _layers.Length)
            throw new ArgumentOutOfRangeException(nameof(lastBlock), lastBlock, $"Last block must be -1 or below {_layers.Length}.");
        return RunBlocks(backend, pixelValues, lastBlock, tapBlocks: default, taps: null);
    }

    /// <summary>The token sequences after each block in <paramref name="blockIndices"/>, from one forward pass and before any final norm: the raw block
    /// outputs a multi-scale head (DPT-style) reads. The blocks must be distinct and in range; the result is in the order given and the caller owns every
    /// tensor. Blocks past the last one asked for are not run.</summary>
    public Tensor[] EncodeIntermediates(IBackend backend, Tensor pixelValues, ReadOnlySpan<int> blockIndices) =>
        EncodeIntermediates(backend, pixelValues, blockIndices, prefixResetBlocks: default);

    /// <summary><see cref="EncodeIntermediates(IBackend, Tensor, ReadOnlySpan{int})"/>, where the blocks in <paramref name="prefixResetBlocks"/> keep the CLS
    /// and storage tokens they were given, so only the patch tokens move on: a lone frame through Depth Pro 2's cross-frame blocks.</summary>
    public Tensor[] EncodeIntermediates(IBackend backend, Tensor pixelValues, ReadOnlySpan<int> blockIndices, ReadOnlySpan<int> prefixResetBlocks)
    {
        ArgumentNullException.ThrowIfNull(backend);
        int last = CheckTaps(blockIndices, prefixResetBlocks, nameof(prefixResetBlocks));
        return CollectTaps(backend, pixelValues, last, blockIndices, prefixResetBlocks, crossFrameBlocks: default, ring: null);
    }

    /// <summary>
    /// <see cref="EncodeIntermediates(IBackend, Tensor, ReadOnlySpan{int})"/> over the frames of one clip or the views of one scene,
    /// <c>[T, 3, H, W]</c>: the blocks in <paramref name="crossFrameBlocks"/> run every frame as one sequence (<see cref="CrossFrameTokens"/>),
    /// each frame keeping its prefix tokens, and the others run each frame on its own. With a <paramref name="ring"/> the input is one frame
    /// (<c>T = 1</c>) whose patches in the cross-frame blocks attend to the frames the ring holds as well (<see cref="CrossFrameKvRing"/>).
    /// Taps are <c>[T, prefix + patches, hidden]</c>; the caller owns them.
    /// </summary>
    public Tensor[] EncodeFrameIntermediates(IBackend backend, Tensor frames, ReadOnlySpan<int> blockIndices, ReadOnlySpan<int> crossFrameBlocks,
        CrossFrameKvRing? ring = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(frames);
        if (ring is not null && frames.Shape.Rank == 4 && frames.Shape[0] != 1)
            throw new ArgumentException($"A cached frame runs alone; got {frames.Shape[0]} frames.", nameof(frames));
        int last = CheckTaps(blockIndices, crossFrameBlocks, nameof(crossFrameBlocks));
        return CollectTaps(backend, frames, last, blockIndices, prefixResetBlocks: default, crossFrameBlocks, ring);
    }

    // The highest tapped block, after checking the taps are distinct and every block named is in range.
    private int CheckTaps(ReadOnlySpan<int> blockIndices, ReadOnlySpan<int> otherBlocks, string otherName)
    {
        if (blockIndices.IsEmpty) throw new ArgumentException("Name at least one block to tap.", nameof(blockIndices));
        int last = -1;
        for (int i = 0; i < blockIndices.Length; i++)
        {
            int block = blockIndices[i];
            if (block < 0 || block >= _layers.Length)
                throw new ArgumentOutOfRangeException(nameof(blockIndices), block, $"Tapped blocks must be in 0..{_layers.Length - 1}.");
            for (int j = 0; j < i; j++)
                if (blockIndices[j] == block) throw new ArgumentException($"Block {block} is tapped twice.", nameof(blockIndices));
            last = Math.Max(last, block);
        }
        foreach (int block in otherBlocks)
            if (block < 0 || block >= _layers.Length)
                throw new ArgumentOutOfRangeException(otherName, block, $"Blocks must be in 0..{_layers.Length - 1}.");
        return last;
    }

    private Tensor[] CollectTaps(IBackend backend, Tensor pixelValues, int last, ReadOnlySpan<int> blockIndices, ReadOnlySpan<int> prefixResetBlocks,
        ReadOnlySpan<int> crossFrameBlocks, CrossFrameKvRing? ring)
    {
        Tensor?[] taps = new Tensor?[_layers.Length];
        try
        {
            // The last block run is always tapped, so every kept tensor is in taps.
            Tensor final = RunBlocks(backend, pixelValues, last, blockIndices, taps, prefixResetBlocks, crossFrameBlocks, ring);
            System.Diagnostics.Debug.Assert(ReferenceEquals(final, taps[last]), "the last block run must be a tapped block");
            Tensor[] result = new Tensor[blockIndices.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = taps[blockIndices[i]]!;
                taps[blockIndices[i]] = null;   // ownership moves to the caller
            }
            return result;
        }
        finally
        {
            foreach (Tensor? t in taps) t?.Dispose();
        }
    }

    // Runs blocks 0..lastBlock. A block listed in tapBlocks has its output stored in taps[block] and is not disposed when the next block starts, so the
    // caller can keep it. Without taps the last block's output is returned for the caller to dispose; with taps it is the last tapped tensor, owned by taps.
    private Tensor RunBlocks(IBackend backend, Tensor pixelValues, int lastBlock, ReadOnlySpan<int> tapBlocks, Tensor?[]? taps,
        ReadOnlySpan<int> prefixResetBlocks = default, ReadOnlySpan<int> crossFrameBlocks = default, CrossFrameKvRing? ring = null)
    {
        int p = _preset.PatchSize;
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[1] != 3 || pixelValues.Shape[2] % p != 0 || pixelValues.Shape[3] % p != 0
            || pixelValues.Shape[2] == 0 || pixelValues.Shape[3] == 0)
            throw new ArgumentException($"pixelValues must be [B,3,H,W] with H and W multiples of {p}; got {pixelValues.Shape}.", nameof(pixelValues));

        int batch = (int)pixelValues.Shape[0];
        int hidden = _preset.HiddenSize;
        int gridH = (int)pixelValues.Shape[2] / p, gridW = (int)pixelValues.Shape[3] / p;
        int numPatches = gridH * gridW;
        int prefix = 1 + _preset.NumRegisterTokens;
        int seqLen = prefix + numPatches;

        Tensor seq = EmbedTokens(backend, pixelValues, batch, hidden, gridH, gridW, prefix, seqLen);
        (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(batch, prefix, gridH, gridW, HeadDim, _preset.RoundRopePeriodsToBf16, _preset.RopeNormalizesByLongSide);
        CrossFrameRope? frameRope = crossFrameBlocks.IsEmpty ? null : new CrossFrameRope(this, ring is null ? batch : 0, prefix, gridH, gridW);
        try
        {
            Tensor h = seq;
            bool hIsTap = false;
            for (int i = 0; i <= lastBlock; i++)
            {
                Tensor next = !crossFrameBlocks.Contains(i) ? _layers[i].Forward(backend, h, cos, sin)
                    : ring is null ? JointBlock(backend, _layers[i], h, prefix, frameRope!)
                    : CachedBlock(backend, i, h, prefix, frameRope!, ring);
                if (prefixResetBlocks.Contains(i)) CopyPrefix(next, h, batch, seqLen, prefix * hidden);
                if (!hIsTap) h.Dispose();
                h = next;
                hIsTap = taps is not null && tapBlocks.Contains(i);
                if (hIsTap) taps![i] = h;
            }
            return h;   // with taps, the last block is always tapped: the tensor is in taps and the caller must not dispose it
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
            frameRope?.Dispose();
        }
    }

    // One block over every frame as one sequence; each frame keeps the prefix it went in with.
    private static Tensor JointBlock(IBackend backend, Dinov3Layer layer, Tensor frames, int prefix, CrossFrameRope rope)
    {
        using Tensor joint = CrossFrameTokens.Join(frames, prefix);
        using Tensor output = layer.Forward(backend, joint, rope.Cos, rope.Sin);
        return CrossFrameTokens.Split(output, frames, prefix);
    }

    // One block over one frame whose patches attend to [pooled-prefix keys, the ring's patch keys]; the prefix passes through unchanged.
    private Tensor CachedBlock(IBackend backend, int block, Tensor frame, int prefix, CrossFrameRope rope, CrossFrameKvRing ring)
    {
        int hidden = _preset.HiddenSize, seqLen = (int)frame.Shape[1], patches = seqLen - prefix;
        Dinov3Layer layer = _layers[block];
        // The frame, its prefix and patch rows, the ring and the result are only ever read and written by backend ops, so on a GPU backend
        // the block moves no tokens through the host.
        long prefixWidth = (long)prefix * hidden, patchWidth = (long)patches * hidden;
        using Tensor prefixRows = new(new TensorShape(1, prefix, hidden), DType.F32);
        using Tensor patchRows = new(new TensorShape(1, patches, hidden), DType.F32);
        backend.CopyStrided(prefixRows, frame, StridedCopySpec.Gather(Flat(0, prefixWidth)));
        backend.CopyStrided(patchRows, frame, StridedCopySpec.Gather(Flat(prefixWidth, patchWidth)));

        int live;
        Tensor? qMh = null, keys = null, values = null;
        try
        {
            (Tensor q, Tensor k, Tensor v) = layer.ProjectAttention(backend, patchRows, rope.Cos, rope.Sin);
            using (q)
            using (k)
            using (v)
            {
                using Tensor pooled = ring.UpdatePrefix(backend, block, prefixRows);
                (Tensor pq, Tensor pk, Tensor pv) = layer.ProjectAttention(backend, pooled, null, null);
                pq.Dispose();
                using (pk)
                using (pv)
                    live = ring.Update(backend, block, k, v, pk, pv);
                qMh = new(new TensorShape(1, _preset.NumHeads, patches, HeadDim), DType.F32);
                backend.Permute0213(qMh, q, patches, _preset.NumHeads, HeadDim);
            }

            // Exact-length copies of the live rows take the dense attention entry, which is faster on the CPU than the cache-view entry.
            keys = CrossFrameKvRing.LiveCopy(backend, ring.Keys(block), live);
            values = CrossFrameKvRing.LiveCopy(backend, ring.Values(block), live);
        }
        catch
        {
            qMh?.Dispose();
            keys?.Dispose();
            values?.Dispose();
            throw;
        }

        AttentionLayer attention = layer.Attention;
        Tensor attended = attention.AttendHeadMajorOperands(backend, qMh, keys, values, default);
        Tensor flat = attention.MergeHeads(backend, attended, new TensorShape(1, patches, hidden));
        using Tensor patchOut = layer.FinishAttention(backend, patchRows, flat);

        Tensor next = new(frame.Shape, DType.F32);
        backend.CopyStrided(next, prefixRows, StridedCopySpec.Scatter(Flat(0, prefixWidth)));
        backend.CopyStrided(next, patchOut, StridedCopySpec.Scatter(Flat(prefixWidth, patchWidth)));
        return next;
    }

    // `count` contiguous elements of a tensor starting at element `offset`, as a view.
    private static StridedView Flat(long offset, long count) => new([count], [1], offset);

    // The RoPE tables of the cross-frame blocks: the joint sequence's (prefix rows, then the patch rows once per frame), or with zero
    // frames the patch rows alone, which a cached frame's patches take.
    private sealed class CrossFrameRope : IDisposable
    {
        public CrossFrameRope(Dinov3VisionEncoder encoder, int frames, int prefix, int gridH, int gridW)
        {
            Dinov2Preset preset = encoder._preset;
            (Tensor cos, Tensor sin) = Dinov3RopeTables.Build(1, 0, gridH, gridW, HeadDim, preset.RoundRopePeriodsToBf16,
                preset.RopeNormalizesByLongSide);
            if (frames == 0)
            {
                (Cos, Sin) = (cos, sin);
                return;
            }

            using (cos)
            using (sin)
            {
                Cos = CrossFrameTokens.TileRope(cos, frames, prefix, 1f);
                Sin = CrossFrameTokens.TileRope(sin, frames, prefix, 0f);
            }
        }

        public Tensor Cos { get; }

        public Tensor Sin { get; }

        public void Dispose()
        {
            Cos.Dispose();
            Sin.Dispose();
        }
    }

    // The first prefixWidth values of every batch item's sequence, from source to target.
    private static void CopyPrefix(Tensor target, Tensor source, int batch, int seqLen, int prefixWidth)
    {
        int hidden = (int)source.Shape[2];
        ReadOnlySpan<float> src = source.AsReadOnlySpan<float>();
        Span<float> dst = target.AsSpan<float>();
        for (int b = 0; b < batch; b++)
        {
            int start = b * seqLen * hidden;
            src.Slice(start, prefixWidth).CopyTo(dst.Slice(start, prefixWidth));
        }
    }

    // Patch projection, then [CLS, storage, patches] per batch item. No position embedding is added: RoPE does that in attention.
    private Tensor EmbedTokens(IBackend backend, Tensor pixelValues, int batch, int hidden, int gridH, int gridW, int prefix, int seqLen)
    {
        int p = _preset.PatchSize, numPatches = gridH * gridW, storage = _preset.NumRegisterTokens;
        Tensor patch = new(new TensorShape(batch, hidden, gridH, gridW), DType.F32);
        backend.Conv2D(patch, pixelValues, _patchW!, _patchB!, p, p, 0, 0);
        Tensor flat = patch.Reshape(new TensorShape(batch, hidden, numPatches));
        Tensor patches = new(new TensorShape(batch, numPatches, hidden), DType.F32);
        backend.Transpose2D(patches, flat, hidden, numPatches);
        patch.Dispose();

        Tensor seq = new(new TensorShape(batch, seqLen, hidden), DType.F32);
        float* dp = (float*)seq.DataPointer;
        float* cls = (float*)_clsToken!.DataPointer;
        float* reg = (float*)_storageTokens!.DataPointer;
        float* pat = (float*)patches.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long rowBase = (long)b * seqLen * hidden;
            for (int c = 0; c < hidden; c++) dp[rowBase + c] = cls[c];
            for (int r = 0; r < storage; r++)
                for (int c = 0; c < hidden; c++) dp[rowBase + (long)(1 + r) * hidden + c] = reg[r * hidden + c];
            long patchBase = (long)b * numPatches * hidden;
            for (long i = 0; i < (long)numPatches * hidden; i++)
                dp[rowBase + (long)prefix * hidden + i] = pat[patchBase + i];
        }
        patches.Dispose();
        return seq;
    }

    /// <summary>The named weight, checked against its exact shape.</summary>
    internal static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] expected)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new ArgumentException($"DINOv3 checkpoint is missing '{key}'.", nameof(weights));
        bool match = t.Shape.Rank == expected.Length;
        for (int d = 0; match && d < expected.Length; d++)
            match = t.Shape[d] == expected[d];
        if (!match)
            throw new ArgumentException($"'{key}' must have shape [{string.Join(", ", expected)}]; got {t.Shape}.", nameof(weights));
        return t;
    }

    internal static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
