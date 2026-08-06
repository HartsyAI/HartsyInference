using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Preprocesses RGB images for the Qwen3-VL vision tower, mirroring the HF <c>Qwen2VLImageProcessor</c> /
/// <c>Qwen3VLProcessor</c> path: smart-resize the image so both sides are multiples of <c>patch · merge</c> and the
/// pixel count sits in <c>[minPixels, maxPixels]</c>, normalize with the CLIP mean/std, duplicate the still image
/// across the temporal patch, and flatten patches into <b>merge-block order</b> (consecutive groups of
/// <c>spatial_merge²</c> patches form a 2×2 spatial block — the layout the merger and the vision RoPE/pos-embed assume).
/// <para>Output: <c>pixelValues [numPatches, inChannels·temporal·patch²]</c> (= <c>[N, 1536]</c>) and the grid
/// <c>(t, h, w)</c> in <i>patch</i> units (t = 1 for a still image).</para></summary>
public sealed unsafe class Qwen3VlImageProcessor
{
    // OpenAI CLIP normalization (Qwen-VL default image_mean / image_std).
    private static readonly float[] ClipMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] ClipStd = [0.26862954f, 0.26130258f, 0.27577711f];

    private readonly Qwen3VlVisionConfig _config;
    private readonly int _minPixels;
    private readonly int _maxPixels;
    private readonly int _maxSideLength;
    private readonly float[] _mean;
    private readonly float[] _std;

    /// <summary>Creates an image processor. <paramref name="minPixels"/>/<paramref name="maxPixels"/> bound the resized
    /// pixel count (defaults pick a ~256–1024 token budget at patch 16 / merge 2); <paramref name="maxSideLength"/>
    /// optionally caps the longer side (0 = uncapped). Boogu-Image's reference pipeline feeds the VLM at most
    /// 384·384 pixels with side ≤ 768 (its training-data budget) — callers replicating it pass those here.</summary>
    /// <param name="imageMean">Per-channel normalization mean; null keeps CLIP's. MiniMax-H3 / upstream Qwen3-VL use <c>[0.5, 0.5, 0.5]</c>.</param>
    /// <param name="imageStd">Per-channel normalization std; null keeps CLIP's.</param>
    public Qwen3VlImageProcessor(Qwen3VlVisionConfig config, int minPixels = 0, int maxPixels = 0, int maxSideLength = 0,
        IReadOnlyList<float>? imageMean = null, IReadOnlyList<float>? imageStd = null)
    {
        _config = config;
        int factor = config.PatchSize * config.SpatialMergeSize;
        _minPixels = minPixels > 0 ? minPixels : factor * factor * 4;
        _maxPixels = maxPixels > 0 ? maxPixels : factor * factor * 1024;
        _maxSideLength = maxSideLength;
        _mean = ToChannelTriple(imageMean, ClipMean, nameof(imageMean));
        _std = ToChannelTriple(imageStd, ClipStd, nameof(imageStd));
    }

    private static float[] ToChannelTriple(IReadOnlyList<float>? values, float[] fallback, string name)
    {
        if (values is null) return fallback;
        if (values.Count != 3) throw new ArgumentException($"Expected 3 channel values, got {values.Count}.", name);
        return [values[0], values[1], values[2]];
    }

    /// <summary>Smart-resizes (h, w) to multiples of <c>patch·merge</c> within the pixel budget (and the optional
    /// side-length cap), preserving aspect.</summary>
    public (int height, int width) SmartResize(int height, int width)
    {
        int factor = _config.PatchSize * _config.SpatialMergeSize;
        if (height < factor) height = factor;
        if (width < factor) width = factor;

        if (_maxSideLength > 0 && Math.Max(height, width) > _maxSideLength)
        {
            double beta = (double)Math.Max(height, width) / _maxSideLength;
            height = Math.Max(factor, (int)Math.Floor(height / beta));
            width = Math.Max(factor, (int)Math.Floor(width / beta));
        }

        int hBar = Math.Max(factor, (int)Math.Round((double)height / factor) * factor);
        int wBar = Math.Max(factor, (int)Math.Round((double)width / factor) * factor);

        if ((long)hBar * wBar > _maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / _maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < _minPixels)
        {
            double beta = Math.Sqrt((double)_minPixels / (height * width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }
        return (hBar, wBar);
    }

    /// <summary>Preprocesses one RGB image tensor <c>[3, H, W]</c> (values in <c>[0, 1]</c>).</summary>
    /// <returns><c>pixelValues [numPatches, patchEmbedInDim]</c> and grid <c>(t, h, w)</c> in patch units.</returns>
    public (Tensor pixelValues, int gridT, int gridH, int gridW) Preprocess(Tensor rgb)
    {
        if (rgb.Shape.Rank != 3 || rgb.Shape[0] != 3)
            throw new ArgumentException($"Expected RGB [3, H, W], got {rgb.Shape}.", nameof(rgb));
        int srcH = (int)rgb.Shape[1];
        int srcW = (int)rgb.Shape[2];
        (int dstH, int dstW) = SmartResize(srcH, srcW);

        // Bilinear resize + normalize into a [3, dstH, dstW] buffer.
        float[] img = ResizeAndNormalize(rgb, srcH, srcW, dstH, dstW, _mean, _std);

        int patch = _config.PatchSize;
        int merge = _config.SpatialMergeSize;
        int temporal = _config.TemporalPatchSize;
        int channels = _config.InChannels;
        int gridH = dstH / patch;
        int gridW = dstW / patch;
        int hBlocks = gridH / merge;
        int wBlocks = gridW / merge;
        int numPatches = gridH * gridW;
        int featDim = channels * temporal * patch * patch;

        Tensor pixelValues = new Tensor(new TensorShape(numPatches, featDim), DType.F32);
        float* dst = (float*)pixelValues.DataPointer;

        int token = 0;
        for (int hb = 0; hb < hBlocks; hb++)
            for (int wb = 0; wb < wBlocks; wb++)
                for (int mh = 0; mh < merge; mh++)
                    for (int mw = 0; mw < merge; mw++)
                    {
                        int row0 = (hb * merge + mh) * patch;
                        int col0 = (wb * merge + mw) * patch;
                        float* tokenDst = dst + (long)token * featDim;
                        int idx = 0;
                        for (int c = 0; c < channels; c++)
                            for (int tp = 0; tp < temporal; tp++) // still image: identical across temporal frames
                                for (int ph = 0; ph < patch; ph++)
                                    for (int pw = 0; pw < patch; pw++)
                                        tokenDst[idx++] = img[(long)c * dstH * dstW + (row0 + ph) * dstW + (col0 + pw)];
                        token++;
                    }

        return (pixelValues, 1, gridH, gridW);
    }

    /// <summary>Preprocesses one temporal patch of RGB frames (<c>[3, H, W]</c> each, values in <c>[0, 1]</c>) as a
    /// single vision block. The frames fill the temporal patch instead of a still image repeating itself, so the grid
    /// is still <c>t = 1</c> and the block costs exactly what a still image of the same resolution costs.</summary>
    /// <param name="frames">Exactly <see cref="Qwen3VlVisionConfig.TemporalPatchSize"/> same-sized F32 frames.</param>
    /// <returns><c>pixelValues [numPatches, patchEmbedInDim]</c> and grid <c>(t, h, w)</c> in patch units.</returns>
    public (Tensor pixelValues, int gridT, int gridH, int gridW) Preprocess(IReadOnlyList<Tensor> frames)
    {
        ArgumentNullException.ThrowIfNull(frames);
        int temporal = _config.TemporalPatchSize;
        if (frames.Count != temporal)
            throw new ArgumentException($"A vision block takes exactly {temporal} frames, got {frames.Count}.", nameof(frames));

        Tensor first = frames[0];
        if (first.Shape.Rank != 3 || first.Shape[0] != 3)
            throw new ArgumentException($"Expected RGB [3, H, W], got {first.Shape}.", nameof(frames));
        for (int i = 0; i < frames.Count; i++)
        {
            // One SmartResize covers the whole block, so a mismatched frame would read out of bounds instead of failing.
            if (frames[i].Shape != first.Shape)
                throw new ArgumentException($"Frame {i} is {frames[i].Shape}, expected {first.Shape}.", nameof(frames));
            if (frames[i].DType != DType.F32)
                throw new ArgumentException($"Frame {i} is {frames[i].DType}, expected F32.", nameof(frames));
        }

        int srcH = (int)first.Shape[1];
        int srcW = (int)first.Shape[2];
        (int dstH, int dstW) = SmartResize(srcH, srcW);

        float[][] imgs = new float[temporal][];
        for (int i = 0; i < temporal; i++)
            imgs[i] = ResizeAndNormalize(frames[i], srcH, srcW, dstH, dstW, _mean, _std);

        int patch = _config.PatchSize;
        int merge = _config.SpatialMergeSize;
        int channels = _config.InChannels;
        int gridH = dstH / patch;
        int gridW = dstW / patch;
        int hBlocks = gridH / merge;
        int wBlocks = gridW / merge;
        int numPatches = gridH * gridW;
        int featDim = channels * temporal * patch * patch;

        Tensor pixelValues = new Tensor(new TensorShape(numPatches, featDim), DType.F32);
        float* dst = (float*)pixelValues.DataPointer;

        int token = 0;
        for (int hb = 0; hb < hBlocks; hb++)
            for (int wb = 0; wb < wBlocks; wb++)
                for (int mh = 0; mh < merge; mh++)
                    for (int mw = 0; mw < merge; mw++)
                    {
                        int row0 = (hb * merge + mh) * patch;
                        int col0 = (wb * merge + mw) * patch;
                        float* tokenDst = dst + (long)token * featDim;
                        int idx = 0;
                        for (int c = 0; c < channels; c++)
                            for (int tp = 0; tp < temporal; tp++)
                                for (int ph = 0; ph < patch; ph++)
                                    for (int pw = 0; pw < patch; pw++)
                                        tokenDst[idx++] = imgs[tp][(long)c * dstH * dstW + (row0 + ph) * dstW + (col0 + pw)];
                        token++;
                    }

        return (pixelValues, 1, gridH, gridW);
    }

    private static float[] ResizeAndNormalize(Tensor rgb, int srcH, int srcW, int dstH, int dstW,
        float[] meanPerChannel, float[] stdPerChannel)
    {
        float* src = (float*)rgb.DataPointer;
        float[] outImg = new float[3 * dstH * dstW];
        // Align-corners=false bilinear (PIL/torchvision default scaling).
        double scaleH = (double)srcH / dstH;
        double scaleW = (double)srcW / dstW;
        for (int c = 0; c < 3; c++)
        {
            float mean = meanPerChannel[c], std = stdPerChannel[c];
            long srcPlane = (long)c * srcH * srcW;
            long dstPlane = (long)c * dstH * dstW;
            for (int y = 0; y < dstH; y++)
            {
                double sy = (y + 0.5) * scaleH - 0.5;
                int y0 = (int)Math.Floor(sy);
                double fy = sy - y0;
                int y0c = Math.Clamp(y0, 0, srcH - 1);
                int y1c = Math.Clamp(y0 + 1, 0, srcH - 1);
                for (int x = 0; x < dstW; x++)
                {
                    double sx = (x + 0.5) * scaleW - 0.5;
                    int x0 = (int)Math.Floor(sx);
                    double fx = sx - x0;
                    int x0c = Math.Clamp(x0, 0, srcW - 1);
                    int x1c = Math.Clamp(x0 + 1, 0, srcW - 1);

                    float v00 = src[srcPlane + (long)y0c * srcW + x0c];
                    float v01 = src[srcPlane + (long)y0c * srcW + x1c];
                    float v10 = src[srcPlane + (long)y1c * srcW + x0c];
                    float v11 = src[srcPlane + (long)y1c * srcW + x1c];
                    double top = v00 + (v01 - v00) * fx;
                    double bot = v10 + (v11 - v10) * fx;
                    float val = (float)(top + (bot - top) * fy);
                    outImg[dstPlane + (long)y * dstW + x] = (val - mean) / std;
                }
            }
        }
        return outImg;
    }
}
