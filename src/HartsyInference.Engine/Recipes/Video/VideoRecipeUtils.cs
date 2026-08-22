using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Shared plumbing every lifted video recipe needs: the SwarmUI-side helpers (<c>RgbToImage</c>, <c>VideoParamResolver</c>, <c>VaePrecisionHelper</c>, <c>ControlVideoDecoder</c>, <c>AudioDecoder</c>, <c>WanModelVariants.PeekKeys</c>) rewritten against the Engine's native <see cref="VideoRequest"/> types.</summary>
internal static class VideoRecipeUtils
{
    /// <summary>Snaps a dimension to the nearest multiple of the model's VAE spatial compression (never below one multiple).</summary>
    internal static int SnapToMultiple(int value, int multiple)
    {
        int snapped = (int)Math.Round(value / (double)multiple) * multiple;
        return Math.Max(multiple, snapped);
    }

    /// <summary>Pixel-size multiple a patchified latent DiT needs: an odd latent grid makes <c>Unpatchify</c> emit one fewer row/column than the scheduler's step expects.</summary>
    internal static int PatchAlignedMultiple(int vaeSpatialCompression, (int T, int H, int W) patchSize) =>
        vaeSpatialCompression * Math.Max(patchSize.H, patchSize.W);

    /// <summary>Snaps the request's width/height to <paramref name="multiple"/>, logging when the geometry moved.</summary>
    internal static (int Width, int Height) ResolveResolution(VideoRequest request, int multiple)
    {
        int requestedWidth = request.Width ?? 704;
        int requestedHeight = request.Height ?? 480;
        int width = SnapToMultiple(requestedWidth, multiple);
        int height = SnapToMultiple(requestedHeight, multiple);
        if (width != requestedWidth || height != requestedHeight)
        {
            Logs.Info($"[VideoRecipe] Resolution {requestedWidth}x{requestedHeight} rounded to {width}x{height} (model requires multiples of {multiple}).");
        }
        return (width, height);
    }

    /// <summary>Rounds the requested frame count onto the model's <c>step·n + 1</c> grid (the VAE temporal compression constraint).</summary>
    internal static int ResolveFrames(VideoRequest request, int modelDefault, int step)
    {
        int requested = request.Frames ?? modelDefault;
        int snapped = Math.Max(1, 1 + (int)Math.Round((requested - 1) / (double)step) * step);
        if (snapped != requested)
        {
            Logs.Info($"[VideoRecipe] Video frame count {requested} rounded to {snapped} (model requires {step}n+1 frames).");
        }
        return snapped;
    }

    /// <summary>Image-to-video geometry: fits the init image's aspect into the model's standard pixel budget ("Image Aspect, Model Res" — the SwarmUI default), or takes the image's own size when the request asks for "Image". Everything snaps to <paramref name="multiple"/>.</summary>
    internal static (int Width, int Height) ResolveI2VResolution(VideoRequest request, int imageWidth, int imageHeight, int multiple)
    {
        string mode = request.VideoResolution ?? "Image Aspect, Model Res";
        int requestedWidth = request.Width ?? 704;
        int requestedHeight = request.Height ?? 480;
        if (string.Equals(mode, "Image", StringComparison.OrdinalIgnoreCase))
        {
            return (SnapToMultiple(imageWidth, multiple), SnapToMultiple(imageHeight, multiple));
        }
        if (string.Equals(mode, "Model Preferred", StringComparison.OrdinalIgnoreCase))
        {
            return (SnapToMultiple(requestedWidth, multiple), SnapToMultiple(requestedHeight, multiple));
        }
        long budget = (long)SnapToMultiple(requestedWidth, multiple) * SnapToMultiple(requestedHeight, multiple);
        double aspect = imageHeight <= 0 ? 1.0 : imageWidth / (double)imageHeight;
        int fitH = (int)Math.Round(Math.Sqrt(budget / aspect));
        int fitW = (int)Math.Round(fitH * aspect);
        return (SnapToMultiple(fitW, multiple), SnapToMultiple(fitH, multiple));
    }

    /// <summary>Frame-edited result for a silent family; <paramref name="audio"/> attaches a soundtrack meant to be heard, <paramref name="fps"/> pins a pipeline-determined playback rate (e.g. a decoded driving clip's).</summary>
    internal static VideoGenerationResult ToResult(byte[][] frames, int width, int height, VideoRequest request,
        AudioBuffer? audio = null, int? fps = null) =>
        new VideoGenerationResult { Frames = ToVideoFrames(frames, width, height, request), Audio = audio, Fps = fps };

    /// <summary>Applies the request's trim + boomerang frame edits and wraps the raw interleaved-RGB frames as the Engine's <see cref="VideoFrame"/> contract (mirrors the extension's <c>VideoOutputEncoder.ApplyFrameEdits</c>).</summary>
    internal static IReadOnlyList<VideoFrame> ToVideoFrames(byte[][] frames, int width, int height, VideoRequest request)
    {
        byte[][] edited = frames;
        int trimStart = Math.Max(0, request.TrimVideoStartFrames);
        int trimEnd = Math.Max(0, request.TrimVideoEndFrames);
        if (trimStart > 0 || trimEnd > 0)
        {
            int keep = edited.Length - trimStart - trimEnd;
            if (keep < 1)
            {
                throw new InvalidOperationException(
                    $"Trim start/end frames ({trimStart}/{trimEnd}) would remove all {edited.Length} generated frames.");
            }
            edited = edited[trimStart..(trimStart + keep)];
        }
        if (request.VideoBoomerang && edited.Length > 2)
        {
            byte[][] looped = new byte[edited.Length * 2 - 2][];
            edited.CopyTo(looped, 0);
            for (int i = 1; i < edited.Length - 1; i++)
            {
                looped[edited.Length + i - 1] = edited[edited.Length - 1 - i];
            }
            edited = looped;
        }
        List<VideoFrame> result = new List<VideoFrame>(edited.Length);
        for (int i = 0; i < edited.Length; i++)
        {
            result.Add(new VideoFrame { Rgb = edited[i], Width = width, Height = height, Index = i });
        }
        return result;
    }

    /// <summary>Aspect-preserving resize onto a black <paramref name="width"/>×<paramref name="height"/> canvas — Wan2.2's <c>padding_resize</c>. The stretching <see cref="ResizeRgb24"/> squashes a portrait reference into a square job, so the identity conditioning sees a distorted face.</summary>
    internal static byte[] LetterboxRgb24(ImageData image, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(image);
        float scale = MathF.Min(width / (float)image.Width, height / (float)image.Height);
        int innerW = Math.Max(1, (int)MathF.Round(image.Width * scale));
        int innerH = Math.Max(1, (int)MathF.Round(image.Height * scale));
        if (innerW == width && innerH == height)
        {
            return ResizeRgb24(image, width, height);
        }
        byte[] inner = ResizeRgb24(image, innerW, innerH);
        byte[] canvas = new byte[(long)width * height * 3];   // zero = black pad
        int offX = (width - innerW) / 2, offY = (height - innerH) / 2;
        for (int y = 0; y < innerH; y++)
        {
            Array.Copy(inner, (long)y * innerW * 3, canvas, ((long)(y + offY) * width + offX) * 3, (long)innerW * 3);
        }
        return canvas;
    }

    /// <summary>Colour-matches every frame of a continuation chunk to the static reference image's Lab stats in place, so drift cannot compound through the carried-frame chain. Chunk 0 is never touched — single-chunk generations stay byte-identical — and <paramref name="strength"/> &lt;= 0 is a no-op.</summary>
    internal static void CorrectContinuationChunk(byte[][] frames, int width, int height, int chunkIndex,
        in VideoColorMatch.LabStats referenceStats, float strength)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (chunkIndex <= 0 || strength <= 0f)
        {
            return;
        }
        foreach (byte[] frame in frames)
        {
            VideoColorMatch.MatchToReference(frame, width, height, referenceStats, strength);
        }
    }

    /// <summary>Bilinear-resamples an <see cref="ImageData"/> to interleaved HWC RGB24 at the target size (the extension's <c>RgbToImage.ToHwcRgbResized</c>).</summary>
    internal static byte[] ResizeRgb24(ImageData image, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), $"Target size must be positive; got {width}x{height}.");
        }
        byte[] src = image.Rgb;
        int srcW = image.Width, srcH = image.Height;
        if (src.Length < (long)srcW * srcH * 3)
        {
            throw new InvalidOperationException($"Init image has {src.Length} bytes, expected {(long)srcW * srcH * 3} for {srcW}x{srcH} RGB24.");
        }
        byte[] dst = new byte[(long)width * height * 3];
        double xRatio = srcW / (double)width;
        double yRatio = srcH / (double)height;
        for (int y = 0; y < height; y++)
        {
            double sy = Math.Min(srcH - 1.0, Math.Max(0.0, (y + 0.5) * yRatio - 0.5));
            int y0 = (int)sy;
            int y1 = Math.Min(srcH - 1, y0 + 1);
            double fy = sy - y0;
            for (int x = 0; x < width; x++)
            {
                double sx = Math.Min(srcW - 1.0, Math.Max(0.0, (x + 0.5) * xRatio - 0.5));
                int x0 = (int)sx;
                int x1 = Math.Min(srcW - 1, x0 + 1);
                double fx = sx - x0;
                for (int c = 0; c < 3; c++)
                {
                    double p00 = src[(y0 * srcW + x0) * 3 + c];
                    double p01 = src[(y0 * srcW + x1) * 3 + c];
                    double p10 = src[(y1 * srcW + x0) * 3 + c];
                    double p11 = src[(y1 * srcW + x1) * 3 + c];
                    double top = p00 + (p01 - p00) * fx;
                    double bottom = p10 + (p11 - p10) * fx;
                    dst[(y * width + x) * 3 + c] = (byte)Math.Clamp(Math.Round(top + (bottom - top) * fy), 0, 255);
                }
            }
        }
        return dst;
    }

    /// <summary>Tiles one interleaved-RGB24 still into the <c>[1, 3, T, H, W]</c> clip tensor in [-1, 1] the Wan control/driving entry points take (the still branch of the extension's <c>ControlVideoDecoder</c>).</summary>
    internal static Tensor TileRgbToClip(byte[] rgb24, int width, int height, int numFrames)
    {
        byte[][] frames = new byte[numFrames][];
        for (int f = 0; f < numFrames; f++)
        {
            frames[f] = rgb24;
        }
        return PackRgbFramesToClip(frames, width, height);
    }

    /// <summary>Packs interleaved-RGB24 frames into the <c>[1, 3, T, H, W]</c> clip tensor in [-1, 1] (the video branch of the extension's <c>ControlVideoDecoder.DecodeControlClip</c>).</summary>
    internal static unsafe Tensor PackRgbFramesToClip(IReadOnlyList<byte[]> frames, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(frames);
        int numFrames = frames.Count;
        if (numFrames < 1)
        {
            throw new ArgumentException("At least one frame is required.", nameof(frames));
        }
        long perFrame = (long)height * width;
        Tensor clip = new Tensor(new TensorShape([1L, 3, numFrames, height, width]), DType.F32);
        float* p = (float*)clip.DataPointer;
        for (int f = 0; f < numFrames; f++)
        {
            byte[] src = frames[f];
            if (src.Length != perFrame * 3)
            {
                clip.Dispose();
                throw new InvalidOperationException(
                    $"Clip frame {f} has {src.Length} bytes, expected {perFrame * 3} for {width}x{height}.");
            }
            for (long pix = 0; pix < perFrame; pix++)
            {
                for (int c = 0; c < 3; c++)
                {
                    p[((long)c * numFrames + f) * perFrame + pix] = src[pix * 3 + c] / 127.5f - 1f;
                }
            }
        }
        return clip;
    }

    /// <summary>Packs one interleaved-RGB24 still into the <c>[1, 3, 1, H, W]</c> reference tensor in [-1, 1].</summary>
    internal static Tensor RgbToReferenceTensor(byte[] rgb24, int width, int height) => TileRgbToClip(rgb24, width, height, 1);

    /// <summary>Strips a Comfy wrapper prefix (e.g. <c>text_encoders.t5xxl.transformer.</c>) from a standalone text-encoder safetensors file, passing unprefixed keys through.</summary>
    internal static Dictionary<string, Tensor> StripPrefix(IReadOnlyDictionary<string, Tensor> raw, string prefix)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = kv.Key.StartsWith(prefix, StringComparison.Ordinal) ? kv.Key[prefix.Length..] : kv.Key;
            result[key] = kv.Value;
        }
        return result;
    }

    /// <summary>Zeroes embedding rows past the real tokens (content + EOS; pad id 0). The Wan family cross-attends every context row with no text mask, and umT5 emits garbage at pad positions that otherwise drowns the prompt — the reference pipeline zero-pads instead. (LTX-Video does NOT use this: its reference truncates to the real tokens instead, via <c>CfgHelper.SliceBatchElementPrefix</c> — see <see cref="LtxVideoRecipePipeline"/>.)</summary>
    internal static unsafe void ZeroPaddedRows(Tensor embeds, int[] tokens, int dim)
    {
        int realLen = 0;
        while (realLen < tokens.Length && tokens[realLen] != 0)
        {
            realLen++;
        }
        int rows = (int)(embeds.Shape.ElementCount / dim);
        if (realLen >= rows)
        {
            return;
        }
        float* p = (float*)embeds.DataPointer;
        new Span<float>(p + (long)realLen * dim, (rows - realLen) * dim).Clear();
    }

    /// <summary>Fresh host-materialized F32 copy of a device-produced tensor (call after <c>backend.Sync()</c>) — the cross-generation cache form that survives per-step activation sweeps and re-faults to device on use.</summary>
    internal static unsafe Tensor HostCopy(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, DType.F32);
        long bytes = x.Shape.ElementCount * 4;
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Copies a <c>[1, seq, dim]</c> tensor to <c>[seq, dim]</c> (the Wan pipelines' image-embeds shape).</summary>
    internal static unsafe Tensor DropBatch(Tensor x)
    {
        int seq = (int)x.Shape[1], dim = (int)x.Shape[2];
        Tensor o = new Tensor(new TensorShape(seq, dim), DType.F32);
        long bytes = (long)seq * dim * 4;
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }

    /// <summary>Decodes an <see cref="AudioClip"/> to mono 16 kHz float samples (the extension's <c>AudioDecoder.DecodeMono16k</c>). Only RIFF/WAVE is decodable in-engine — the ffmpeg transcode the extension used for compressed containers is host-side, so anything else is refused rather than silently mis-read.</summary>
    internal static float[] DecodeMono16k(AudioClip clip)
    {
        ArgumentNullException.ThrowIfNull(clip);
        try
        {
            using MemoryStream stream = new MemoryStream(clip.Data, writable: false);
            WavFile.DecodedAudio decoded = WavFile.Read(stream);
            float[] mono = decoded.ToMono();
            return decoded.SampleRate == 16_000 ? mono : Resampler.Create(decoded.SampleRate, 16_000).Resample(mono);
        }
        catch (Exception ex)
        {
            Logs.Error($"[VideoRecipe] Failed to decode the driving audio clip (format hint '{clip.Format ?? "none"}'); the Engine decodes RIFF/WAVE PCM only.", ex);
            throw;
        }
    }

    /// <summary>Reads the tensor-name set from a safetensors header (8-byte little-endian length + JSON map) without loading any tensor data — the cheap variant sniff the extension's <c>WanModelVariants.PeekKeys</c> does. Returns an empty set on any read error.</summary>
    internal static IReadOnlySet<string> PeekSafeTensorKeys(string path)
    {
        HashSet<string> keys = new HashSet<string>(StringComparer.Ordinal);
        try
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                return keys;
            }
            using FileStream fs = File.OpenRead(path);
            Span<byte> lenBuf = stackalloc byte[8];
            fs.ReadExactly(lenBuf);
            long headerLen = BinaryPrimitives.ReadInt64LittleEndian(lenBuf);
            if (headerLen is <= 0 or > 64 * 1024 * 1024)
            {
                return keys;
            }
            byte[] json = new byte[headerLen];
            fs.ReadExactly(json, 0, (int)headerLen);
            using JsonDocument doc = JsonDocument.Parse(json);
            foreach (JsonProperty prop in doc.RootElement.EnumerateObject())
            {
                if (!string.Equals(prop.Name, "__metadata__", StringComparison.Ordinal))
                {
                    keys.Add(prop.Name);
                }
            }
            return keys;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[VideoRecipe] Safetensors header peek failed for '{path}': {ex.Message}");
            return keys;
        }
    }

    /// <summary>Reads the <c>__metadata__</c> string map from a safetensors header without loading tensor data. Wan-Animate-2 is key-for-key a Wan2.1 I2V-14B checkpoint, so <see cref="PeekSafeTensorKeys"/> cannot tell them apart and the metadata is the only signal. Returns an empty map on any read error.</summary>
    internal static IReadOnlyDictionary<string, string> PeekSafeTensorMetadata(string path)
    {
        Dictionary<string, string> metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        try
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                return metadata;
            }
            using FileStream fs = File.OpenRead(path);
            Span<byte> lenBuf = stackalloc byte[8];
            fs.ReadExactly(lenBuf);
            long headerLen = BinaryPrimitives.ReadInt64LittleEndian(lenBuf);
            if (headerLen is <= 0 or > 64 * 1024 * 1024)
            {
                return metadata;
            }
            byte[] json = new byte[headerLen];
            fs.ReadExactly(json, 0, (int)headerLen);
            using JsonDocument doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("__metadata__", out JsonElement meta) || meta.ValueKind != JsonValueKind.Object)
            {
                return metadata;
            }
            foreach (JsonProperty prop in meta.EnumerateObject())
            {
                if (prop.Value.ValueKind == JsonValueKind.String)
                {
                    metadata[prop.Name] = prop.Value.GetString()!;
                }
            }
            return metadata;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[VideoRecipe] Safetensors metadata peek failed for '{path}': {ex.Message}");
            return metadata;
        }
    }
}
