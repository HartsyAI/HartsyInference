using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Qwen2.5-VL multimodal instruction encoder used by Qwen-Image-Edit. Runs the
/// <see cref="Qwen25VlVisionEncoder"/> on the reference image(s), splices the merged vision tokens into the
/// language-model input embeds at the <c>image_token_id</c> placeholder positions, builds the <b>sectioned</b>
/// M-RoPE 3D position table (Qwen2.5-VL <c>mrope_section = [16, 24, 24]</c>: contiguous frequency chunks per
/// axis, unlike Qwen3-VL's interleaved layout — see <see cref="Qwen3VlMultimodalEncoder"/>), and runs the
/// Qwen2.5-VL language tower via <see cref="LlamaStyleEncoder.EncodeEmbedsMrope"/> (no deepstack — that is a
/// Qwen3-VL feature). Returns the final hidden state <c>[1, seqLen, hidden]</c>.
///
/// <para>For text-only prompts the 3D positions collapse to <c>(s, s, s)</c> and every section reads the same
/// position — exactly the standard 1D-RoPE text encode, so one entry point serves both.</para></summary>
public sealed unsafe class Qwen25VlMultimodalEncoder
{
    /// <summary>Qwen2.5-VL <c>&lt;|image_pad|&gt;</c> placeholder token id.</summary>
    public const int ImageTokenId = 151655;
    /// <summary>Qwen2.5-VL <c>&lt;|vision_start|&gt;</c> token id.</summary>
    public const int VisionStartId = 151652;
    /// <summary>Qwen2.5-VL <c>&lt;|vision_end|&gt;</c> token id.</summary>
    public const int VisionEndId = 151653;

    private readonly LlamaStyleEncoder _language;
    private readonly Qwen25VlVisionEncoder _vision;
    private readonly Qwen25VlImageProcessor _imageProcessor;
    private readonly int _spatialMerge;
    private readonly int _textHeadDim;
    private readonly double _ropeTheta;
    private readonly int[] _mropeSection;

    /// <summary>Creates the encoder over a shared language tower, vision tower, and image processor.</summary>
    /// <param name="textHeadDim">LM head dim (128 for Qwen2.5-VL-7B).</param>
    /// <param name="ropeTheta">LM RoPE base (1e6 for Qwen2.5-VL).</param>
    /// <param name="mropeSection">M-RoPE per-axis frequency-chunk sizes (T, H, W) = [16, 24, 24].</param>
    public Qwen25VlMultimodalEncoder(LlamaStyleEncoder language, Qwen25VlVisionEncoder vision,
        Qwen25VlImageProcessor imageProcessor, int spatialMergeSize = 2,
        int textHeadDim = 128, double ropeTheta = 1_000_000.0, int[]? mropeSection = null)
    {
        _language = language;
        _vision = vision;
        _imageProcessor = imageProcessor;
        _spatialMerge = spatialMergeSize;
        _textHeadDim = textHeadDim;
        _ropeTheta = ropeTheta;
        _mropeSection = mropeSection ?? [16, 24, 24];
    }

    /// <summary>Vision-tower weights, for pipeline-phase PreloadWeights/FreeWeights staging. The language tower is staged separately by the owner (it is shared with the text-only path); without an explicit free the tower's weights auto-promote into the resident cache on the second encode (cond + uncond) and permanently shrink the DiT budget.</summary>
    public IEnumerable<Tensor> EnumerateVisionWeights() => _vision.EnumerateWeights();

    /// <summary>Number of merged vision tokens (= <c>&lt;|image_pad|&gt;</c> placeholders the template must
    /// contain) that <paramref name="image"/> will produce, without running the tower. Used by callers to
    /// build the templated token sequence before encoding.</summary>
    public int CountImageTokens(Tensor image)
    {
        (int gridH, int gridW) = _imageProcessor.ComputeGrid(image);
        return (gridH / _spatialMerge) * (gridW / _spatialMerge);
    }

    /// <summary>Encodes a templated instruction. <paramref name="tokenIds"/> must already contain
    /// <see cref="ImageTokenId"/> repeated once per merged vision token (between the vision start/end markers)
    /// for each image in <paramref name="referenceImages"/> — use <see cref="CountImageTokens"/> to size the
    /// runs; pass an empty list for text-only. Returns the LM final hidden state <c>[1, seqLen, hidden]</c>.</summary>
    public Tensor Encode(IBackend backend, int[] tokenIds, IReadOnlyList<Tensor> referenceImages)
    {
        int seqLen = tokenIds.Length;

        // 1. Vision forward per image; collect grids + merged tokens.
        int imageCount = referenceImages.Count;
        Tensor[] mergedTokens = new Tensor[imageCount];
        (int h, int w)[] grids = new (int, int)[imageCount];
        int totalImageTokens = 0;
        for (int i = 0; i < imageCount; i++)
        {
            (Tensor pix, int gh, int gw) = _imageProcessor.Preprocess(referenceImages[i]);
            grids[i] = (gh, gw);
            mergedTokens[i] = _vision.Forward(backend, pix, gh, gw);
            pix.Dispose();
            totalImageTokens += (int)mergedTokens[i].Shape[1];
        }

        // 2. Validate placeholder count.
        bool[] visualMask = new bool[seqLen];
        int placeholderCount = 0;
        for (int s = 0; s < seqLen; s++)
            if (tokenIds[s] == ImageTokenId) { visualMask[s] = true; placeholderCount++; }
        if (placeholderCount != totalImageTokens)
            throw new InvalidOperationException(
                $"image placeholder count ({placeholderCount}) != merged vision tokens ({totalImageTokens}).");

        // 3. Input embeds with vision tokens spliced in.
        Tensor embeds = _language.LookupEmbeddings(tokenIds);
        SpliceVisionTokens(embeds, mergedTokens, visualMask, seqLen);

        // 4. 3D M-RoPE position ids → sectioned cos/sin table.
        (int[] posT, int[] posH, int[] posW) = BuildRopeIndex(tokenIds, grids);
        (float[] cos, float[] sin) = BuildSectionedMropeTable(posT, posH, posW);

        Tensor result = _language.EncodeEmbedsMrope(backend, embeds, cos, sin);
        embeds.Dispose();
        foreach (Tensor t in mergedTokens) t.Dispose();
        return result;
    }

    private static void SpliceVisionTokens(Tensor embeds, Tensor[] mergedTokens, bool[] visualMask, int seqLen)
    {
        int H = (int)embeds.Shape[2];
        float* ep = (float*)embeds.DataPointer;
        int img = 0, within = 0;
        for (int s = 0; s < seqLen; s++)
        {
            if (!visualMask[s]) continue;
            while (img < mergedTokens.Length && within >= (int)mergedTokens[img].Shape[1]) { img++; within = 0; }
            float* src = (float*)mergedTokens[img].DataPointer + (long)within * H;
            Buffer.MemoryCopy(src, ep + (long)s * H, (long)H * sizeof(float), (long)H * sizeof(float));
            within++;
        }
    }

    /// <summary>HF <c>get_rope_index</c> for a single (unpadded) sequence: text runs advance all three axes
    /// together; each image's tokens get <c>(t = base, h = base + row, w = base + col)</c> over the MERGED
    /// grid and the base then advances by <c>max(llm_h, llm_w)</c>. Identical semantics to the ComfyUI
    /// <c>qwen2vl_mrope_position_ids</c> running-offset formulation.</summary>
    private (int[] t, int[] h, int[] w) BuildRopeIndex(int[] tokenIds, (int h, int w)[] grids)
    {
        int seqLen = tokenIds.Length;
        int[] pt = new int[seqLen], ph = new int[seqLen], pw = new int[seqLen];
        int current = 0;
        int imgIdx = 0;
        int s = 0;
        while (s < seqLen)
        {
            if (tokenIds[s] != ImageTokenId)
            {
                pt[s] = ph[s] = pw[s] = current;
                current++;
                s++;
                continue;
            }
            (int gh, int gw) = grids[imgIdx];
            int llmH = gh / _spatialMerge;
            int llmW = gw / _spatialMerge;
            int count = llmH * llmW;
            for (int i = 0; i < count; i++)
            {
                pt[s + i] = current;
                ph[s + i] = current + i / llmW;
                pw[s + i] = current + i % llmW;
            }
            current += Math.Max(llmH, llmW);
            s += count;
            imgIdx++;
        }
        return (pt, ph, pw);
    }

    /// <summary>Builds the sectioned-M-RoPE split-half <c>(cos, sin)</c> table sized <c>seqLen · headDim/2</c>:
    /// frequency-pair index <c>j</c> reads the T axis for <c>j &lt; 16</c>, H for <c>16 ≤ j &lt; 40</c>, W for
    /// <c>40 ≤ j &lt; 64</c> (Qwen2.5-VL <c>mrope_section</c> chunks), with
    /// <c>inv_freq[j] = theta^(-2j/headDim)</c>.</summary>
    private (float[] cos, float[] sin) BuildSectionedMropeTable(int[] pt, int[] ph, int[] pw)
    {
        int seqLen = pt.Length;
        int half = _textHeadDim / 2;   // 64
        double[] invFreq = new double[half];
        for (int j = 0; j < half; j++)
            invFreq[j] = 1.0 / Math.Pow(_ropeTheta, (double)(2 * j) / _textHeadDim);

        int hStart = _mropeSection[0];                       // 16
        int wStart = _mropeSection[0] + _mropeSection[1];    // 40

        float[] cos = new float[(long)seqLen * half];
        float[] sin = new float[(long)seqLen * half];
        for (int s = 0; s < seqLen; s++)
        {
            long b = (long)s * half;
            for (int j = 0; j < half; j++)
            {
                int pos = j < hStart ? pt[s] : j < wStart ? ph[s] : pw[s];
                double angle = pos * invFreq[j];
                cos[b + j] = (float)Math.Cos(angle);
                sin[b + j] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }
}
