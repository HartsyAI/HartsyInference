using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>MiniMax-H3's conditioning tower: Qwen3-VL-32B truncated to its first 50 layers, producing the
/// <b>unnormalized</b> last hidden state <c>[textLen, 5120]</c> that <c>MiniMaxH3Pipeline.Generate</c> consumes,
/// together with the per-token modality tag runs from <see cref="MiniMaxH3TextEncoding"/>.
///
/// <para>The checkpoint (<c>qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors</c>) ships no <c>model.norm</c> and no
/// <c>lm_head</c> — the truncation IS the tap, so the last block's output is returned as-is. The prompt is not
/// chat-templated; <see cref="MiniMaxH3TextEncoding.Build(Qwen2Tokenizer, string, IReadOnlyList{MiniMaxH3TextEncoding.Condition})"/>
/// owns the presentation and tagging.</para>
///
/// <para>Every dimension is derived from the checkpoint header rather than a preset, because the 32B tower differs
/// from the 8B presets already in the repo (LM hidden 5120 vs 4096, 64/8 GQA heads, MLP 25600). Quantized linears
/// are bound as <see cref="Nvfp4Linear"/> and dequantized one BF16 slice at a time inside the forward: eager F32 for
/// the whole tower is ~100 GB and even 16-bit it is ~49 GB, so no dequantized weight can stay resident. Every layer
/// borrows the same <see cref="Nvfp4Linear.CreateDequantScratch">scratch buffer</see> (sized by the widest linear,
/// ~262 MB), held only for the duration of an encode. Plain BF16 releases of the same checkpoint load through the
/// identical path with the dense branch.</para>
///
/// <para>Positions use Qwen3-VL's interleaved <b>M-RoPE</b>; for a text-only prompt the three axes collapse to
/// <c>(s, s, s)</c> and it is ordinary 1D RoPE. Reference images are encoded by <see cref="Qwen3VlVisionEncoder"/>
/// and spliced at the <c>&lt;|image_pad|&gt;</c> positions with deepstack injection, matching
/// <see cref="Qwen3VlMultimodalEncoder"/>.</para></summary>
public sealed unsafe class MiniMaxH3TextEncoder : IDisposable
{
    /// <summary>Reference pixel budget from <c>comfy/text_encoders/minimax.py</c>.</summary>
    private const int MinPixels = 3136;

    private const int MaxPixels = 12845056;

    /// <summary>Qwen3-VL image normalization used by H3 — plain <c>[0.5, 0.5, 0.5]</c>, not CLIP's statistics.</summary>
    private static readonly float[] NormalizationMean = [0.5f, 0.5f, 0.5f];

    private static readonly float[] NormalizationStd = [0.5f, 0.5f, 0.5f];

    private readonly float _ropeTheta;
    private readonly int[] _mropeSection;
    private readonly float _rmsNormEps;
    private readonly int _imageTokenId;

    private readonly List<LanguageBlock> _blocks = [];
    private Tensor? _embedWeight;
    private Tensor? _embedScale;
    private Qwen3VlVisionEncoder? _vision;
    private Qwen3VlVisionConfig? _visionConfig;

    private int _hiddenSize;
    private int _vocabSize;
    private int _headDim;
    private int _queryHeads;
    private int _kvHeads;
    private int _intermediateSize;
    private long _dequantScratchElements;
    private int _disposed;

    /// <summary>Creates the encoder. The defaults are Qwen3-VL-32B-Instruct's <c>config.json</c> values, which are
    /// not recoverable from the weight header.</summary>
    /// <param name="mropeSection">Per-axis M-RoPE sections (T, H, W); Qwen3-VL uses <c>[24, 20, 20]</c>.</param>
    public MiniMaxH3TextEncoder(float ropeTheta = 5_000_000f, int[]? mropeSection = null, float rmsNormEps = 1e-6f,
        int imageTokenId = Qwen2Tokenizer.ImagePadId)
    {
        _ropeTheta = ropeTheta;
        _mropeSection = mropeSection ?? [24, 20, 20];
        _rmsNormEps = rmsNormEps;
        _imageTokenId = imageTokenId;
    }

    /// <summary>The encoded prompt: conditioning states plus the tag runs the DiT modulates by.</summary>
    public sealed record Result
    {
        /// <summary>Unnormalized last hidden state, <c>[textLen, hiddenSize]</c>, F32. Caller disposes.</summary>
        public required Tensor HiddenStates { get; init; }

        /// <summary>Contiguous per-token modality runs, in the tuple shape <c>MiniMaxH3Pipeline.Generate</c> takes.</summary>
        public required IReadOnlyList<(int Start, int Stop, int Tag)> TagRuns { get; init; }
    }

    /// <summary>Language hidden size (5120 for the 32B tower).</summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>Number of loaded transformer layers (50 — the checkpoint is truncated there).</summary>
    public int NumLayers => _blocks.Count;

    /// <summary>Vision-tower configuration derived from the header, or null when the checkpoint has no <c>visual.*</c>.</summary>
    public Qwen3VlVisionConfig? VisionConfig => _visionConfig;

    /// <summary>Binds the checkpoint under its HuggingFace keys (<c>model.embed_tokens.*</c>,
    /// <c>model.layers.{i}.*</c>, optional <c>visual.*</c>). Nothing is materialized beyond the norms, the embedding
    /// table, and the vision tower; quantized linears stay packed.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ThrowIfDisposed();

        _embedWeight = weights["model.embed_tokens.weight"];
        weights.TryGetValue("model.embed_tokens.weight_scale", out _embedScale);
        if (_embedWeight.Shape.Rank != 2)
            throw new InvalidOperationException($"model.embed_tokens.weight must be rank-2; got {_embedWeight.Shape}.");
        _vocabSize = (int)_embedWeight.Shape[0];
        _hiddenSize = (int)_embedWeight.Shape[1];
        if (_embedWeight.DType == DType.I8 && _embedScale is null)
            throw new InvalidOperationException("int8 model.embed_tokens.weight is missing its 'weight_scale' companion.");
        if (_embedScale is not null && _embedScale.ElementCount != _vocabSize)
            throw new InvalidOperationException(
                $"model.embed_tokens.weight_scale must have one scale per row ({_vocabSize}); got {_embedScale.Shape}.");
        if (_embedWeight.DType != DType.I8)
            _embedWeight = _embedWeight.DType == DType.F32 ? _embedWeight : _embedWeight.CastTo(DType.F32);

        while (weights.ContainsKey($"model.layers.{_blocks.Count}.input_layernorm.weight"))
        {
            LanguageBlock block = new LanguageBlock();
            block.LoadWeights(weights, $"model.layers.{_blocks.Count}");
            _blocks.Add(block);
        }
        if (_blocks.Count == 0)
            throw new InvalidOperationException("No 'model.layers.*' found in the MiniMax-H3 text-encoder checkpoint.");

        long scratchElements = 0;
        foreach (LanguageBlock b in _blocks)
            scratchElements = Math.Max(scratchElements, b.DequantScratchElements);
        _dequantScratchElements = scratchElements;

        LanguageBlock first = _blocks[0];
        _headDim = (int)first.QueryNorm.ElementCount;
        _queryHeads = first.Query.OutFeatures / _headDim;
        _kvHeads = first.Key.OutFeatures / _headDim;
        _intermediateSize = first.Gate.OutFeatures;
        if (_queryHeads * _headDim != first.Query.OutFeatures || _kvHeads * _headDim != first.Key.OutFeatures)
            throw new InvalidOperationException(
                $"Attention projections are not a multiple of head_dim {_headDim} (q={first.Query.OutFeatures}, k={first.Key.OutFeatures}).");
        if (_queryHeads % _kvHeads != 0)
        {
            throw new InvalidOperationException(
                $"Query heads ({_queryHeads}) must be a multiple of KV heads ({_kvHeads}) for grouped-query attention.");
        }

        LoadVisionTower(weights);
        Logs.Verbose($"MiniMaxH3TextEncoder loaded: {_blocks.Count} layers, hidden={_hiddenSize}, " +
            $"heads={_queryHeads}/{_kvHeads}x{_headDim}, mlp={_intermediateSize}, vision={(_vision is not null ? "yes" : "no")}");
    }

    /// <summary>Merged vision tokens an image of this size expands to. Callers must reserve exactly this many in the
    /// presentation before the tower runs, so the block size has to be predictable from the dimensions alone.</summary>
    public int VisionTokenCount(int height, int width)
    {
        if (_visionConfig is null)
        {
            throw new InvalidOperationException("The checkpoint has no 'visual.*' tower, so it encodes no images.");
        }
        Qwen3VlImageProcessor processor = new Qwen3VlImageProcessor(_visionConfig, MinPixels, MaxPixels,
            imageMean: NormalizationMean, imageStd: NormalizationStd);
        (int dstH, int dstW) = processor.SmartResize(height, width);
        return MiniMaxH3TextEncoding.MergedTokenCount(dstH / _visionConfig.PatchSize, dstW / _visionConfig.PatchSize,
            _visionConfig.SpatialMergeSize);
    }

    /// <summary>Tokenizes <paramref name="prompt"/> with H3's presentation rules and encodes it.</summary>
    public Result Encode(IBackend backend, Qwen2Tokenizer tokenizer, string prompt,
        IReadOnlyList<MiniMaxH3TextEncoding.Condition>? conditions = null, IReadOnlyList<Tensor>? referenceImages = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return Encode(backend, MiniMaxH3TextEncoding.Build(tokenizer, prompt, conditions), referenceImages);
    }

    /// <summary>Encodes an already-built presentation. <paramref name="referenceImages"/> carries one entry per spliced
    /// vision block in sequence order: an RGB <c>[3, H, W]</c> still, or a <c>[T, 3, H, W]</c> frame stack filling the
    /// temporal patch. Both are in <c>[0, 1]</c> and expand to the same token count at equal resolution.</summary>
    public Result Encode(IBackend backend, MiniMaxH3TextEncoding.Encoded presentation,
        IReadOnlyList<Tensor>? referenceImages = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(presentation);
        ThrowIfDisposed();
        if (_embedWeight is null)
            throw new InvalidOperationException("MiniMaxH3TextEncoder.LoadWeights must run before Encode.");

        int[] tokenIds = presentation.TokenIds;
        int seqLen = tokenIds.Length;
        Tensor hidden = LookupEmbeddings(tokenIds);
        VisionResult[] visionOut = RunVisionTower(backend, presentation, referenceImages);
        // One buffer for every quantized linear in the tower, instead of a host allocation per projection per layer.
        Tensor? dequantScratch = _dequantScratchElements > 0 ? Nvfp4Linear.CreateDequantScratch(_dequantScratchElements)
            : null;
        try
        {
            bool[] visualMask = new bool[seqLen];
            for (int s = 0; s < seqLen; s++) visualMask[s] = tokenIds[s] == _imageTokenId;
            if (visionOut.Length > 0) SpliceVisionTokens(hidden, visionOut, visualMask, seqLen);

            (int[] posT, int[] posH, int[] posW) = BuildRopeIndex(tokenIds, visionOut);
            (float[] cos, float[] sin) = BuildInterleavedMropeTable(posT, posH, posW);

            for (int i = 0; i < _blocks.Count; i++)
            {
                Tensor next = _blocks[i].Forward(backend, hidden, cos, sin, seqLen, this, dequantScratch);
                hidden.Dispose();
                hidden = next;
                if (visionOut.Length > 0 && i < visionOut[0].Output.DeepstackFeatures.Length)
                    InjectDeepstack(hidden, visionOut, visualMask, seqLen, i);
            }

            Tensor states = new Tensor(new TensorShape(seqLen, _hiddenSize), DType.F32);
            long bytes = (long)seqLen * _hiddenSize * sizeof(float);
            Buffer.MemoryCopy((void*)hidden.DataPointer, (void*)states.DataPointer, bytes, bytes);
            return new Result { HiddenStates = states, TagRuns = ToTupleRuns(presentation.TagRuns) };
        }
        finally
        {
            dequantScratch?.Dispose();
            hidden.Dispose();
            foreach (VisionResult vr in visionOut)
            {
                vr.Output.MergedTokens.Dispose();
                foreach (Tensor d in vr.Output.DeepstackFeatures) d.Dispose();
            }
        }
    }

    /// <summary>Enumerates the resident weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        foreach (LanguageBlock block in _blocks)
            foreach (Tensor w in block.EnumerateWeights())
                yield return w;
        if (_vision is not null)
            foreach (Tensor w in _vision.EnumerateWeights())
                yield return w;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (LanguageBlock block in _blocks) block.Dispose();
        _blocks.Clear();
        _vision?.Dispose();
        _vision = null;
        _embedWeight = null;
        _embedScale = null;
    }

    private void LoadVisionTower(IReadOnlyDictionary<string, Tensor> weights)
    {
        if (!weights.TryGetValue("visual.merger.linear_fc2.weight", out Tensor? mergerOut)) return;

        int depth = 0;
        while (weights.ContainsKey($"visual.blocks.{depth}.norm1.weight")) depth++;
        int deepstack = 0;
        while (weights.ContainsKey($"visual.deepstack_merger_list.{deepstack}.norm.weight")) deepstack++;
        Tensor patchProj = weights["visual.patch_embed.proj.weight"];
        Qwen3VlVisionConfig baseConfig = Qwen3VlVisionConfig.Qwen3Vl8B;
        if (deepstack != baseConfig.DeepstackVisualIndexes.Length)
            throw new InvalidOperationException(
                $"Vision tower has {deepstack} deepstack mergers; the known Qwen3-VL tap layout has " +
                $"{baseConfig.DeepstackVisualIndexes.Length}.");

        _visionConfig = baseConfig with
        {
            Depth = depth,
            HiddenSize = (int)weights["visual.patch_embed.proj.bias"].ElementCount,
            IntermediateSize = (int)weights["visual.blocks.0.mlp.linear_fc1.bias"].ElementCount,
            InChannels = (int)patchProj.Shape[1],
            TemporalPatchSize = (int)patchProj.Shape[2],
            PatchSize = (int)patchProj.Shape[4],
            NumPositionEmbeddings = (int)weights["visual.pos_embed.weight"].Shape[0],
            OutHiddenSize = (int)mergerOut.Shape[0],
        };

        Dictionary<string, Tensor> visual = new Dictionary<string, Tensor>();
        foreach (KeyValuePair<string, Tensor> kvp in weights)
            if (kvp.Key.StartsWith("visual.", StringComparison.Ordinal))
                visual[kvp.Key["visual.".Length..]] = kvp.Value;

        _vision = new Qwen3VlVisionEncoder(_visionConfig);
        _vision.LoadWeights(visual);
    }

    private VisionResult[] RunVisionTower(IBackend backend,
        MiniMaxH3TextEncoding.Encoded presentation, IReadOnlyList<Tensor>? referenceImages)
    {
        int blockCount = presentation.VisionBlockTokenCounts.Count;
        int imageCount = referenceImages?.Count ?? 0;
        if (imageCount == 0 && blockCount == 0) return [];
        if (imageCount != blockCount)
            throw new ArgumentException(
                $"Presentation splices {blockCount} vision blocks but {imageCount} images were supplied.",
                nameof(referenceImages));
        if (_vision is null)
            throw new InvalidOperationException("The checkpoint has no 'visual.*' tower, so vision conditions cannot be encoded.");

        Qwen3VlImageProcessor processor = new Qwen3VlImageProcessor(_visionConfig!, MinPixels, MaxPixels,
            imageMean: NormalizationMean, imageStd: NormalizationStd);
        VisionResult[] outputs = new VisionResult[imageCount];
        for (int i = 0; i < imageCount; i++)
        {
            Tensor input = referenceImages![i];
            (Tensor pix, int gt, int gh, int gw) = input.Shape.Rank == 4 ? processor.Preprocess(FrameViews(input))
                : processor.Preprocess(input);
            outputs[i] = new VisionResult(_vision.Forward(backend, pix, gt, gh, gw), gh, gw);
            pix.Dispose();
            if (outputs[i].Output.NumMergedTokens != presentation.VisionBlockTokenCounts[i])
                throw new InvalidOperationException(
                    $"Vision block {i} expands to {outputs[i].Output.NumMergedTokens} tokens but the presentation " +
                    $"reserved {presentation.VisionBlockTokenCounts[i]}.");
        }
        return outputs;
    }

    /// <summary>Borrowed per-frame <c>[3, H, W]</c> views into a <c>[T, 3, H, W]</c> stack. The views own no memory and
    /// never outlive the caller's loop iteration, which is what keeps the stack itself rooted.</summary>
    private static Tensor[] FrameViews(Tensor frames)
    {
        if (frames.Shape[1] != 3)
            throw new ArgumentException($"Expected a frame stack [T, 3, H, W], got {frames.Shape}.", nameof(frames));
        if (frames.DType != DType.F32)
            throw new ArgumentException($"Frame stacks must be F32, got {frames.DType}.", nameof(frames));
        int count = (int)frames.Shape[0];
        TensorShape frameShape = new TensorShape(3, frames.Shape[2], frames.Shape[3]);
        long stride = frameShape.ElementCount;
        float* start = (float*)frames.DataPointer;
        Tensor[] views = new Tensor[count];
        for (int i = 0; i < count; i++)
            views[i] = new Tensor(start + i * stride, frameShape, frames.DType, frames.Device);
        return views;
    }

    /// <summary>Looks up <c>[1, seqLen, hidden]</c> F32 embeddings, dequantizing only the rows actually referenced
    /// when the table is int8 (the full table is 3.1 GB at F32).</summary>
    private Tensor LookupEmbeddings(int[] tokenIds)
    {
        int seqLen = tokenIds.Length;
        Tensor output = new Tensor(new TensorShape(1, seqLen, _hiddenSize), DType.F32);
        float* dst = (float*)output.DataPointer;
        int hidden = _hiddenSize;

        for (int s = 0; s < seqLen; s++)
        {
            int id = tokenIds[s];
            if ((uint)id >= (uint)_vocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} out of vocab ({_vocabSize}).");
            float* row = dst + (long)s * hidden;
            if (_embedWeight!.DType == DType.I8)
            {
                sbyte* src = (sbyte*)_embedWeight.DataPointer + (long)id * hidden;
                float scale = ((float*)_embedScale!.DataPointer)[id];
                for (int h = 0; h < hidden; h++) row[h] = src[h] * scale;
            }
            else
            {
                float* src = (float*)_embedWeight.DataPointer + (long)id * hidden;
                Buffer.MemoryCopy(src, row, hidden * sizeof(float), hidden * sizeof(float));
            }
        }
        return output;
    }

    /// <summary>Runs Qwen3-VL's causal grouped-query attention without expanding KV heads or materializing a score mask.</summary>
    internal static void CausalGqaAttention(IBackend backend, Tensor output, Tensor query, Tensor key, Tensor value,
        int queryHeads, int kvHeads, int seqLen, int headDim)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
        {
            throw new ArgumentException(
                $"Qwen3-VL query heads ({queryHeads}) must be a positive multiple of KV heads ({kvHeads}).");
        }
        backend.FlashAttention(output, query, key, value, seqLen, queryHeads / kvHeads, causal: true,
            qOffset: 0, scale: 1.0f / MathF.Sqrt(headDim));
    }

    private static void SpliceVisionTokens(Tensor embeds, VisionResult[] visionOut, bool[] visualMask, int seqLen)
    {
        int hidden = (int)embeds.Shape[2];
        float* ep = (float*)embeds.DataPointer;
        int img = 0, within = 0;
        for (int s = 0; s < seqLen; s++)
        {
            if (!visualMask[s]) continue;
            while (img < visionOut.Length && within >= visionOut[img].Output.NumMergedTokens) { img++; within = 0; }
            if (img >= visionOut.Length)
                throw new InvalidOperationException("More image placeholders than merged vision tokens.");
            float* src = (float*)visionOut[img].Output.MergedTokens.DataPointer + (long)within * hidden;
            Buffer.MemoryCopy(src, ep + (long)s * hidden, (long)hidden * sizeof(float), (long)hidden * sizeof(float));
            within++;
        }
    }

    private void InjectDeepstack(Tensor hidden, VisionResult[] visionOut, bool[] visualMask, int seqLen, int layer)
    {
        float* hp = (float*)hidden.DataPointer;
        int img = 0, within = 0;
        for (int s = 0; s < seqLen; s++)
        {
            if (!visualMask[s]) continue;
            while (img < visionOut.Length && within >= visionOut[img].Output.NumMergedTokens) { img++; within = 0; }
            float* src = (float*)visionOut[img].Output.DeepstackFeatures[layer].DataPointer + (long)within * _hiddenSize;
            float* dst = hp + (long)s * _hiddenSize;
            for (int d = 0; d < _hiddenSize; d++) dst[d] += src[d];
            within++;
        }
    }

    /// <summary>HF <c>get_rope_index</c>: text runs advance all three axes together, an image run takes
    /// <c>(base, base + row, base + col)</c> and the base then advances by <c>max(rows, cols)</c>.</summary>
    private (int[] T, int[] H, int[] W) BuildRopeIndex(int[] tokenIds, VisionResult[] visionOut)
    {
        int seqLen = tokenIds.Length;
        int[] pt = new int[seqLen], ph = new int[seqLen], pw = new int[seqLen];
        int merge = _visionConfig?.SpatialMergeSize ?? 2;
        int current = 0, imgIdx = 0, s = 0;
        while (s < seqLen)
        {
            if (tokenIds[s] != _imageTokenId)
            {
                pt[s] = ph[s] = pw[s] = current;
                current++;
                s++;
                continue;
            }
            VisionResult vr = visionOut[imgIdx];
            int llmH = vr.GridH / merge;
            int llmW = vr.GridW / merge;
            int count = vr.Output.NumMergedTokens;
            for (int i = 0; i < count; i++)
            {
                pt[s + i] = current;
                ph[s + i] = current + (i / llmW) % llmH;
                pw[s + i] = current + i % llmW;
            }
            current += Math.Max(llmH, llmW);
            s += count;
            imgIdx++;
        }
        return (pt, ph, pw);
    }

    /// <summary>Interleaved-M-RoPE split-half table sized <c>seqLen · headDim/2</c>: frequency <c>j</c> reads axis
    /// <c>j % 3</c> inside the interleaved region and the temporal axis beyond it.</summary>
    private (float[] Cos, float[] Sin) BuildInterleavedMropeTable(int[] pt, int[] ph, int[] pw)
    {
        int seqLen = pt.Length;
        int half = _headDim / 2;
        double[] invFreq = new double[half];
        for (int j = 0; j < half; j++)
            invFreq[j] = 1.0 / Math.Pow(_ropeTheta, (double)(2 * j) / _headDim);
        int interleavedLen = Math.Min(half, _mropeSection[1] * 3);

        float[] cos = new float[(long)seqLen * half];
        float[] sin = new float[(long)seqLen * half];
        for (int s = 0; s < seqLen; s++)
        {
            long b = (long)s * half;
            for (int j = 0; j < half; j++)
            {
                int axis = j < interleavedLen ? j % 3 : 0;
                int pos = axis == 0 ? pt[s] : axis == 1 ? ph[s] : pw[s];
                double angle = pos * invFreq[j];
                cos[b + j] = (float)Math.Cos(angle);
                sin[b + j] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static IReadOnlyList<(int Start, int Stop, int Tag)> ToTupleRuns(
        IReadOnlyList<MiniMaxH3TextEncoding.TagRun> runs)
    {
        (int, int, int)[] result = new (int, int, int)[runs.Count];
        for (int i = 0; i < runs.Count; i++) result[i] = (runs[i].Start, runs[i].Stop, runs[i].Tag);
        return result;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>One vision block's tower output plus the patch grid its M-RoPE positions need.</summary>
    private readonly record struct VisionResult(Qwen3VlVisionEncoder.VisionOutput Output, int GridH, int GridW);

    /// <summary>One decoder layer: RMSNorm → GQA attention with per-head Q/K RMSNorm and M-RoPE → residual →
    /// RMSNorm → SwiGLU → residual. Mirrors the Llama/Qwen block in <see cref="LlamaStyleEncoder"/>; it is a
    /// separate implementation only because every projection here may be nvfp4-packed and is therefore
    /// dequantized inside the forward rather than held dense.</summary>
    private sealed class LanguageBlock : IDisposable
    {
        private Tensor? _inputNorm;
        private Tensor? _postAttnNorm;
        private Tensor? _queryNorm;
        private Tensor? _keyNorm;
        private Nvfp4Linear? _query;
        private Nvfp4Linear? _key;
        private Nvfp4Linear? _value;
        private Nvfp4Linear? _out;
        private Nvfp4Linear? _gate;
        private Nvfp4Linear? _up;
        private Nvfp4Linear? _down;

        public Nvfp4Linear Query => _query!;

        public Nvfp4Linear Key => _key!;

        public Nvfp4Linear Gate => _gate!;

        public Tensor QueryNorm => _queryNorm!;

        /// <summary>Widest dequant slice any of this block's linears needs.</summary>
        public long DequantScratchElements
        {
            get
            {
                long max = 0;
                foreach (Nvfp4Linear linear in Linears())
                    max = Math.Max(max, linear.DequantScratchElements);
                return max;
            }
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _inputNorm = TensorCasts.EnsureF32(weights[$"{prefix}.input_layernorm.weight"]);
            _postAttnNorm = TensorCasts.EnsureF32(weights[$"{prefix}.post_attention_layernorm.weight"]);
            _queryNorm = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.q_norm.weight"]);
            _keyNorm = TensorCasts.EnsureF32(weights[$"{prefix}.self_attn.k_norm.weight"]);
            _query = Nvfp4Linear.Load(weights, $"{prefix}.self_attn.q_proj");
            _key = Nvfp4Linear.Load(weights, $"{prefix}.self_attn.k_proj");
            _value = Nvfp4Linear.Load(weights, $"{prefix}.self_attn.v_proj");
            _out = Nvfp4Linear.Load(weights, $"{prefix}.self_attn.o_proj");
            _gate = Nvfp4Linear.Load(weights, $"{prefix}.mlp.gate_proj");
            _up = Nvfp4Linear.Load(weights, $"{prefix}.mlp.up_proj");
            _down = Nvfp4Linear.Load(weights, $"{prefix}.mlp.down_proj");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_inputNorm is not null) yield return _inputNorm;
            if (_postAttnNorm is not null) yield return _postAttnNorm;
            if (_queryNorm is not null) yield return _queryNorm;
            if (_keyNorm is not null) yield return _keyNorm;
            foreach (Nvfp4Linear linear in Linears())
                foreach (Tensor t in linear.EnumerateWeights())
                    yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, float[] ropeCos, float[] ropeSin,
            int seqLen, MiniMaxH3TextEncoder owner, Tensor? dequantScratch)
        {
            int hiddenSize = owner._hiddenSize;
            int queryHeads = owner._queryHeads;
            int kvHeads = owner._kvHeads;
            int headDim = owner._headDim;
            TensorShape hiddenShape = new TensorShape(1, seqLen, hiddenSize);
            TensorShape queryShape = new TensorShape(1, seqLen, queryHeads * headDim);
            TensorShape kvShape = new TensorShape(1, seqLen, kvHeads * headDim);
            TensorShape queryHeadShape = new TensorShape(1, queryHeads, seqLen, headDim);
            TensorShape kvHeadShape = new TensorShape(1, kvHeads, seqLen, headDim);

            Tensor preAttn = new Tensor(hiddenShape, DType.F32);
            backend.RmsNorm(preAttn, hidden, _inputNorm!, owner._rmsNormEps);

            Tensor qFlat = new Tensor(queryShape, DType.F32);
            Tensor kFlat = new Tensor(kvShape, DType.F32);
            Tensor vFlat = new Tensor(kvShape, DType.F32);
            _query!.Forward(backend, qFlat, preAttn, dequantScratch);
            _key!.Forward(backend, kFlat, preAttn, dequantScratch);
            _value!.Forward(backend, vFlat, preAttn, dequantScratch);
            preAttn.Dispose();

            Tensor qMh = new Tensor(queryHeadShape, DType.F32);
            Tensor kMh = new Tensor(kvHeadShape, DType.F32);
            Tensor vMh = new Tensor(kvHeadShape, DType.F32);
            FlatToHeads(qMh, qFlat, seqLen, queryHeads, headDim);
            FlatToHeads(kMh, kFlat, seqLen, kvHeads, headDim);
            FlatToHeads(vMh, vFlat, seqLen, kvHeads, headDim);
            qFlat.Dispose();
            kFlat.Dispose();
            vFlat.Dispose();

            Tensor qNormed = new Tensor(queryHeadShape, DType.F32);
            Tensor kNormed = new Tensor(kvHeadShape, DType.F32);
            backend.RmsNorm(qNormed, qMh, _queryNorm!, owner._rmsNormEps);
            backend.RmsNorm(kNormed, kMh, _keyNorm!, owner._rmsNormEps);
            qMh.Dispose();
            kMh.Dispose();
            qMh = qNormed;
            kMh = kNormed;

            ApplyRopeSplitHalf(qMh, ropeCos, ropeSin, queryHeads, seqLen, headDim);
            ApplyRopeSplitHalf(kMh, ropeCos, ropeSin, kvHeads, seqLen, headDim);

            Tensor attnOut = new Tensor(queryHeadShape, DType.F32);
            CausalGqaAttention(backend, attnOut, qMh, kMh, vMh, queryHeads, kvHeads, seqLen, headDim);
            qMh.Dispose();
            kMh.Dispose();
            vMh.Dispose();

            Tensor attnFlat = new Tensor(queryShape, DType.F32);
            HeadsToFlat(attnFlat, attnOut, seqLen, queryHeads, headDim);
            attnOut.Dispose();

            Tensor attnProj = new Tensor(hiddenShape, DType.F32);
            _out!.Forward(backend, attnProj, attnFlat, dequantScratch);
            attnFlat.Dispose();

            Tensor afterAttn = new Tensor(hiddenShape, DType.F32);
            backend.Add(afterAttn, hidden, attnProj);
            attnProj.Dispose();

            Tensor preMlp = new Tensor(hiddenShape, DType.F32);
            backend.RmsNorm(preMlp, afterAttn, _postAttnNorm!, owner._rmsNormEps);

            TensorShape mlpShape = new TensorShape(1, seqLen, owner._intermediateSize);
            Tensor gate = new Tensor(mlpShape, DType.F32);
            Tensor up = new Tensor(mlpShape, DType.F32);
            _gate!.Forward(backend, gate, preMlp, dequantScratch);
            _up!.Forward(backend, up, preMlp, dequantScratch);
            preMlp.Dispose();

            Tensor activated = new Tensor(mlpShape, DType.F32);
            backend.Silu(activated, gate);
            gate.Dispose();
            Tensor gated = new Tensor(mlpShape, DType.F32);
            backend.Mul(gated, activated, up);
            activated.Dispose();
            up.Dispose();

            Tensor mlpOut = new Tensor(hiddenShape, DType.F32);
            _down!.Forward(backend, mlpOut, gated, dequantScratch);
            gated.Dispose();

            Tensor result = new Tensor(hiddenShape, DType.F32);
            backend.Add(result, afterAttn, mlpOut);
            afterAttn.Dispose();
            mlpOut.Dispose();
            return result;
        }

        public void Dispose()
        {
            foreach (Nvfp4Linear linear in Linears()) linear.Dispose();
            _query = _key = _value = _out = _gate = _up = _down = null;
        }

        private IEnumerable<Nvfp4Linear> Linears()
        {
            if (_query is not null) yield return _query;
            if (_key is not null) yield return _key;
            if (_value is not null) yield return _value;
            if (_out is not null) yield return _out;
            if (_gate is not null) yield return _gate;
            if (_up is not null) yield return _up;
            if (_down is not null) yield return _down;
        }

        private static void FlatToHeads(Tensor output, Tensor input, int seqLen, int heads, int headDim)
        {
            float* src = (float*)input.DataPointer;
            float* dst = (float*)output.DataPointer;
            long bytes = (long)headDim * sizeof(float);
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < heads; h++)
                    Buffer.MemoryCopy(src + ((long)s * heads + h) * headDim,
                        dst + ((long)h * seqLen + s) * headDim, bytes, bytes);
        }

        private static void HeadsToFlat(Tensor output, Tensor input, int seqLen, int heads, int headDim)
        {
            float* src = (float*)input.DataPointer;
            float* dst = (float*)output.DataPointer;
            long bytes = (long)headDim * sizeof(float);
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < heads; h++)
                    Buffer.MemoryCopy(src + ((long)h * seqLen + s) * headDim,
                        dst + ((long)s * heads + h) * headDim, bytes, bytes);
        }

        private static void ApplyRopeSplitHalf(Tensor x, float[] cos, float[] sin, int heads, int seqLen, int headDim)
        {
            int half = headDim / 2;
            float* p = (float*)x.DataPointer;
            for (int h = 0; h < heads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    long off = ((long)h * seqLen + s) * headDim;
                    int posOff = s * half;
                    for (int i = 0; i < half; i++)
                    {
                        float c = cos[posOff + i];
                        float sn = sin[posOff + i];
                        float a = p[off + i];
                        float b = p[off + i + half];
                        p[off + i] = a * c - b * sn;
                        p[off + i + half] = b * c + a * sn;
                    }
                }
            }
        }
    }
}
