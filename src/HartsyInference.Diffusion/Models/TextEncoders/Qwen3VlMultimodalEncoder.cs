using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Qwen3-VL multimodal instruction encoder used by Boogu-Image editing. Runs the <see cref="Qwen3VlVisionEncoder"/>
/// on the reference image(s), splices the merged vision tokens into the language-model input embeds at the
/// <c>image_token_id</c> placeholder positions, builds the interleaved <b>M-RoPE</b> 3D position ids (HF
/// <c>get_rope_index</c>), and runs the Qwen3-VL language tower (via <see cref="LlamaStyleEncoder.EncodeEmbedsMrope"/>)
/// with <b>deepstack</b> injection. Returns the final hidden state <c>[1, seqLen, hidden]</c> — the Boogu instruction
/// features.
///
/// <para>For text-only prompts (no image) the 3D positions collapse to <c>(s, s, s)</c>, so this is exactly the
/// standard 1D-RoPE text encode. Structural; numeric parity pending (docs/Research/BOOGU_IMAGE.md §10).</para></summary>
public sealed unsafe class Qwen3VlMultimodalEncoder
{
    private readonly LlamaStyleEncoder _language;
    private readonly Qwen3VlVisionEncoder _vision;
    private readonly Qwen3VlImageProcessor _imageProcessor;
    private readonly int _imageTokenId;
    private readonly int _spatialMerge;
    private readonly int _textHeadDim;
    private readonly double _ropeTheta;
    private readonly int[] _mropeSection;
    private readonly int _numDeepstackLayers;

    /// <summary>Creates the encoder over a shared language tower, vision tower, and image processor.</summary>
    /// <param name="imageTokenId">The placeholder token id replaced by merged vision tokens (151655 for Qwen3-VL).</param>
    /// <param name="textHeadDim">LM head dim (128).</param>
    /// <param name="ropeTheta">LM RoPE base (5e6).</param>
    /// <param name="mropeSection">M-RoPE per-axis sections (T,H,W) = [24,20,20].</param>
    public Qwen3VlMultimodalEncoder(LlamaStyleEncoder language, Qwen3VlVisionEncoder vision,
        Qwen3VlImageProcessor imageProcessor, Qwen3VlVisionConfig visionConfig, int imageTokenId = 151655,
        int textHeadDim = 128, double ropeTheta = 5_000_000.0, int[]? mropeSection = null)
    {
        _language = language;
        _vision = vision;
        _imageProcessor = imageProcessor;
        _imageTokenId = imageTokenId;
        _spatialMerge = visionConfig.SpatialMergeSize;
        _textHeadDim = textHeadDim;
        _ropeTheta = ropeTheta;
        _mropeSection = mropeSection ?? [24, 20, 20];
        _numDeepstackLayers = visionConfig.DeepstackVisualIndexes.Length;
    }

    /// <summary>Encodes a templated instruction. <paramref name="tokenIds"/> must already contain
    /// <c>image_token_id</c> repeated once per merged vision token (between the vision start/end markers) for each image
    /// in <paramref name="referenceImages"/>; pass an empty list for text-only. Returns the LM final hidden state.</summary>
    public Tensor Encode(IBackend backend, int[] tokenIds, IReadOnlyList<Tensor> referenceImages)
    {
        int seqLen = tokenIds.Length;

        // 1. Vision forward per image; collect grids + merged tokens + deepstack features.
        int imageCount = referenceImages.Count;
        Qwen3VlVisionEncoder.VisionOutput[] visionOut = new Qwen3VlVisionEncoder.VisionOutput[imageCount];
        (int t, int h, int w)[] grids = new (int, int, int)[imageCount];
        int totalImageTokens = 0;
        for (int i = 0; i < imageCount; i++)
        {
            (Tensor pix, int gt, int gh, int gw) = _imageProcessor.Preprocess(referenceImages[i]);
            grids[i] = (gt, gh, gw);
            visionOut[i] = _vision.Forward(backend, pix, gt, gh, gw);
            pix.Dispose();
            totalImageTokens += visionOut[i].NumMergedTokens;
        }

        // 2. Validate placeholder count.
        bool[] visualMask = new bool[seqLen];
        int placeholderCount = 0;
        for (int s = 0; s < seqLen; s++)
            if (tokenIds[s] == _imageTokenId) { visualMask[s] = true; placeholderCount++; }
        if (placeholderCount != totalImageTokens)
            throw new InvalidOperationException(
                $"image placeholder count ({placeholderCount}) != merged vision tokens ({totalImageTokens}).");

        // 3. Input embeds with vision tokens spliced in.
        Tensor embeds = _language.LookupEmbeddings(tokenIds);
        SpliceVisionTokens(embeds, visionOut, visualMask, seqLen);

        // 4. 3D M-RoPE position ids → interleaved cos/sin table.
        (int[] posT, int[] posH, int[] posW) = BuildRopeIndex(tokenIds, grids);
        (float[] cos, float[] sin) = BuildInterleavedMropeTable(posT, posH, posW);

        // 5. Deepstack features per first-N layers (concatenated across images, in sequence order).
        IReadOnlyList<Tensor>? deepstack = imageCount > 0 ? ConcatDeepstack(visionOut) : null;

        Tensor result = _language.EncodeEmbedsMrope(backend, embeds, cos, sin, deepstack, imageCount > 0 ? visualMask : null);
        embeds.Dispose();

        foreach (Qwen3VlVisionEncoder.VisionOutput vo in visionOut)
        {
            vo.MergedTokens.Dispose();
            foreach (Tensor d in vo.DeepstackFeatures) d.Dispose();
        }
        if (deepstack is not null) foreach (Tensor d in deepstack) d.Dispose();
        return result;
    }

    private static void SpliceVisionTokens(Tensor embeds, Qwen3VlVisionEncoder.VisionOutput[] visionOut,
        bool[] visualMask, int seqLen)
    {
        int H = (int)embeds.Shape[2];
        float* ep = (float*)embeds.DataPointer;
        int img = 0, within = 0;
        for (int s = 0; s < seqLen; s++)
        {
            if (!visualMask[s]) continue;
            while (img < visionOut.Length && within >= visionOut[img].NumMergedTokens) { img++; within = 0; }
            float* src = (float*)visionOut[img].MergedTokens.DataPointer + (long)within * H;
            Buffer.MemoryCopy(src, ep + (long)s * H, (long)H * sizeof(float), (long)H * sizeof(float));
            within++;
        }
    }

    /// <summary>Concatenates each deepstack layer's features across all images into one tensor per layer, in image
    /// (and within-image raster) order — matching the visual-token order in the sequence.</summary>
    private Tensor[] ConcatDeepstack(Qwen3VlVisionEncoder.VisionOutput[] visionOut)
    {
        int outDim = (int)visionOut[0].MergedTokens.Shape[1];
        int total = 0;
        foreach (Qwen3VlVisionEncoder.VisionOutput vo in visionOut) total += vo.NumMergedTokens;
        Tensor[] result = new Tensor[_numDeepstackLayers];
        for (int layer = 0; layer < _numDeepstackLayers; layer++)
        {
            Tensor cat = new Tensor(new TensorShape(total, outDim), DType.F32);
            float* dp = (float*)cat.DataPointer;
            int offset = 0;
            foreach (Qwen3VlVisionEncoder.VisionOutput vo in visionOut)
            {
                long bytes = (long)vo.NumMergedTokens * outDim * sizeof(float);
                Buffer.MemoryCopy((void*)vo.DeepstackFeatures[layer].DataPointer, dp + (long)offset * outDim, bytes, bytes);
                offset += vo.NumMergedTokens;
            }
            result[layer] = cat;
        }
        return result;
    }

    /// <summary>HF <c>get_rope_index</c> for a single (unpadded) sequence: text runs advance all three axes together;
    /// each image's tokens get <c>(t = base, h = base + row, w = base + col)</c> and the base then advances by
    /// <c>max(llm_h, llm_w)</c>.</summary>
    private (int[] t, int[] h, int[] w) BuildRopeIndex(int[] tokenIds, (int t, int h, int w)[] grids)
    {
        int seqLen = tokenIds.Length;
        int[] pt = new int[seqLen], ph = new int[seqLen], pw = new int[seqLen];
        int current = 0;
        int imgIdx = 0;
        int s = 0;
        while (s < seqLen)
        {
            if (tokenIds[s] != _imageTokenId)
            {
                pt[s] = ph[s] = pw[s] = current;
                current++;
                s++;
                continue;
            }
            // image run.
            (int gt, int gh, int gw) = grids[imgIdx];
            int llmH = gh / _spatialMerge;
            int llmW = gw / _spatialMerge;
            int llmT = gt;
            int count = llmT * llmH * llmW;
            for (int i = 0; i < count; i++)
            {
                int row = (i / llmW) % llmH;
                int col = i % llmW;
                pt[s + i] = current;
                ph[s + i] = current + row;
                pw[s + i] = current + col;
            }
            current += Math.Max(llmH, llmW);
            s += count;
            imgIdx++;
        }
        return (pt, ph, pw);
    }

    /// <summary>Builds the interleaved-M-RoPE split-half <c>(cos, sin)</c> table sized <c>seqLen · headDim/2</c>:
    /// freq dim <c>j</c> uses axis <c>j%3</c> for <c>j &lt; sum(section)·... </c> (interleaved T,H,W) and the temporal
    /// axis beyond the interleaved region, with <c>inv_freq[j] = theta^(-2j/headDim)</c>.</summary>
    private (float[] cos, float[] sin) BuildInterleavedMropeTable(int[] pt, int[] ph, int[] pw)
    {
        int seqLen = pt.Length;
        int half = _textHeadDim / 2; // 64
        double[] invFreq = new double[half];
        for (int j = 0; j < half; j++)
            invFreq[j] = 1.0 / Math.Pow(_ropeTheta, (double)(2 * j) / _textHeadDim);

        // Interleaved region length: H and W each occupy mrope_section[1|2] interleaved triples.
        int interleavedLen = Math.Min(half, _mropeSection[1] * 3); // 60 for [24,20,20]

        float[] cos = new float[(long)seqLen * half];
        float[] sin = new float[(long)seqLen * half];
        for (int s = 0; s < seqLen; s++)
        {
            long b = (long)s * half;
            for (int j = 0; j < half; j++)
            {
                int axis = (j < interleavedLen) ? (j % 3) : 0; // 0=T,1=H,2=W; tail = T
                int pos = axis == 0 ? pt[s] : axis == 1 ? ph[s] : pw[s];
                double angle = pos * invFreq[j];
                cos[b + j] = (float)Math.Cos(angle);
                sin[b + j] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }
}
