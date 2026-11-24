using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
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
    public Tensor[] EncodeIntermediates(IBackend backend, Tensor pixelValues, ReadOnlySpan<int> blockIndices)
    {
        ArgumentNullException.ThrowIfNull(backend);
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

        Tensor?[] taps = new Tensor?[_layers.Length];
        try
        {
            _ = RunBlocks(backend, pixelValues, last, blockIndices, taps);   // every kept tensor is in taps; the last block is tapped
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
    private Tensor RunBlocks(IBackend backend, Tensor pixelValues, int lastBlock, ReadOnlySpan<int> tapBlocks, Tensor?[]? taps)
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
        try
        {
            Tensor h = seq;
            bool hIsTap = false;
            for (int i = 0; i <= lastBlock; i++)
            {
                Tensor next = _layers[i].Forward(backend, h, cos, sin);
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
