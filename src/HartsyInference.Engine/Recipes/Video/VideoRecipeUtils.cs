using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Shared plumbing every lifted video recipe needs: the SwarmUI-side helpers (<c>RgbToImage</c>,
/// <c>VideoParamResolver</c>, <c>VaePrecisionHelper</c>, <c>ControlVideoDecoder</c>, <c>AudioDecoder</c>,
/// <c>WanModelVariants.PeekKeys</c>) rewritten against the Engine's native <see cref="VideoRequest"/> types.</summary>
internal static class VideoRecipeUtils
{
    /// <summary>Snaps a dimension to the nearest multiple of the model's VAE spatial compression (never below one multiple).</summary>
    internal static int SnapToMultiple(int value, int multiple)
    {
        int snapped = (int)Math.Round(value / (double)multiple) * multiple;
        return Math.Max(multiple, snapped);
    }

    /// <summary>Snaps the request's width/height to <paramref name="multiple"/>, logging when the geometry moved.</summary>
    internal static (int Width, int Height) ResolveResolution(VideoRequest request, int multiple)
    {
        int width = SnapToMultiple(request.Width, multiple);
        int height = SnapToMultiple(request.Height, multiple);
        if (width != request.Width || height != request.Height)
        {
            Logs.Info($"[VideoRecipe] Resolution {request.Width}x{request.Height} rounded to {width}x{height} (model requires multiples of {multiple}).");
        }
        return (width, height);
    }

    /// <summary>Rounds the requested frame count onto the model's <c>step·n + 1</c> grid (the VAE temporal compression constraint).</summary>
    internal static int ResolveFrames(VideoRequest request, int modelDefault, int step)
    {
        int requested = request.Frames > 0 ? request.Frames : modelDefault;
        int snapped = Math.Max(1, 1 + (int)Math.Round((requested - 1) / (double)step) * step);
        if (snapped != requested)
        {
            Logs.Info($"[VideoRecipe] Video frame count {requested} rounded to {snapped} (model requires {step}n+1 frames).");
        }
        return snapped;
    }

    /// <summary>Image-to-video geometry: fits the init image's aspect into the model's standard pixel budget
    /// ("Image Aspect, Model Res" — the SwarmUI default), or takes the image's own size when the request asks for
    /// "Image". Everything snaps to <paramref name="multiple"/>.</summary>
    internal static (int Width, int Height) ResolveI2VResolution(VideoRequest request, int imageWidth, int imageHeight, int multiple)
    {
        string mode = request.VideoResolution ?? "Image Aspect, Model Res";
        if (string.Equals(mode, "Image", StringComparison.OrdinalIgnoreCase))
        {
            return (SnapToMultiple(imageWidth, multiple), SnapToMultiple(imageHeight, multiple));
        }
        if (string.Equals(mode, "Model Preferred", StringComparison.OrdinalIgnoreCase))
        {
            return (SnapToMultiple(request.Width, multiple), SnapToMultiple(request.Height, multiple));
        }
        long budget = (long)SnapToMultiple(request.Width, multiple) * SnapToMultiple(request.Height, multiple);
        double aspect = imageHeight <= 0 ? 1.0 : imageWidth / (double)imageHeight;
        int fitH = (int)Math.Round(Math.Sqrt(budget / aspect));
        int fitW = (int)Math.Round(fitH * aspect);
        return (SnapToMultiple(fitW, multiple), SnapToMultiple(fitH, multiple));
    }

    /// <summary>Applies the request's trim + boomerang frame edits and wraps the raw interleaved-RGB frames as the
    /// Engine's <see cref="VideoFrame"/> contract (mirrors the extension's <c>VideoOutputEncoder.ApplyFrameEdits</c>).</summary>
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

    /// <summary>Bilinear-resamples an <see cref="ImageData"/> to interleaved HWC RGB24 at the target size (the
    /// extension's <c>RgbToImage.ToHwcRgbResized</c>).</summary>
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

    /// <summary>Tiles one interleaved-RGB24 still into the <c>[1, 3, T, H, W]</c> clip tensor in [-1, 1] the Wan
    /// control/driving entry points take (the still branch of the extension's <c>ControlVideoDecoder</c>).</summary>
    internal static unsafe Tensor TileRgbToClip(byte[] rgb24, int width, int height, int numFrames)
    {
        Tensor clip = new Tensor(new TensorShape([1L, 3, numFrames, height, width]), DType.F32);
        float* p = (float*)clip.DataPointer;
        long perFrame = (long)height * width;
        for (int f = 0; f < numFrames; f++)
        {
            for (long pix = 0; pix < perFrame; pix++)
            {
                for (int c = 0; c < 3; c++)
                {
                    p[((long)c * numFrames + f) * perFrame + pix] = rgb24[pix * 3 + c] / 127.5f - 1f;
                }
            }
        }
        return clip;
    }

    /// <summary>Packs one interleaved-RGB24 still into the <c>[1, 3, 1, H, W]</c> reference tensor in [-1, 1].</summary>
    internal static Tensor RgbToReferenceTensor(byte[] rgb24, int width, int height) => TileRgbToClip(rgb24, width, height, 1);

    /// <summary>Casts a VAE weight dictionary to <paramref name="target"/>, leaving tensors already at that dtype
    /// untouched (the extension's <c>VaePrecisionHelper.CastVaeWeights</c>; the Wan VAE resnets overflow at F16).</summary>
    internal static Dictionary<string, Tensor> CastWeights(IReadOnlyDictionary<string, Tensor> weights, DType target)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            result[kv.Key] = kv.Value.DType == target ? kv.Value : kv.Value.CastTo(target);
        }
        return result;
    }

    /// <summary>Strips a Comfy wrapper prefix (e.g. <c>text_encoders.t5xxl.transformer.</c>) from a standalone
    /// text-encoder safetensors file, passing unprefixed keys through.</summary>
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

    /// <summary>Zeroes embedding rows past the real tokens (content + EOS; pad id 0). Wan/LTX cross-attend every
    /// context row with no text mask, and the encoders emit garbage at pad positions that otherwise drowns the
    /// prompt — the reference pipelines zero-pad instead.</summary>
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

    /// <summary>Fresh host-materialized F32 copy of a device-produced tensor (call after <c>backend.Sync()</c>) — the
    /// cross-generation cache form that survives per-step activation sweeps and re-faults to device on use.</summary>
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

    /// <summary>Decodes an <see cref="AudioClip"/> to mono 16 kHz float samples (the extension's
    /// <c>AudioDecoder.DecodeMono16k</c>). Only RIFF/WAVE is decodable in-engine — the ffmpeg transcode the extension
    /// used for compressed containers is host-side, so anything else is refused rather than silently mis-read.</summary>
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

    /// <summary>Reads the tensor-name set from a safetensors header (8-byte little-endian length + JSON map) without
    /// loading any tensor data — the cheap variant sniff the extension's <c>WanModelVariants.PeekKeys</c> does.
    /// Returns an empty set on any read error.</summary>
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
}
