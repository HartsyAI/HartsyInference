using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.Diffusion.Utilities;

/// <summary>"Tiny AutoEncoder for Stable Diffusion" preview decoder
/// (<see href="https://github.com/madebyollin/taesd"/>). Takes a noisy in-flight latent
/// <c>[1, C, H, W]</c> and produces an 8×-upscaled <c>[1, 3, H*8, W*8]</c> RGB tensor in
/// <c>[0, 1]</c>. ~30–80 ms per call on GPU — much higher fidelity than
/// <see cref="LatentPreview"/> at the cost of a ~10 MB safetensors file per architecture.
///
/// <para><b>Reference architecture</b> — a single <c>nn.Sequential</c>, exactly:
/// <code>
///   [0] Clamp                          # x ← tanh(x/3) * 3
///   [1] Conv2d(latent_C → 64, 3×3, padding=1)
///   [2] ReLU
///   [3..5] Block × 3                   # 3 ResBlocks at native latent resolution
///   [6] Upsample(2×, nearest)
///   [7] Conv2d(64 → 64, 3×3, padding=1, bias=False)
///   [8..10] Block × 3                  # 3 ResBlocks at 2× resolution
///   [11] Upsample(2×)
///   [12] Conv2d(64 → 64, bias=False)
///   [13..15] Block × 3                 # 3 ResBlocks at 4×
///   [16] Upsample(2×)
///   [17] Conv2d(64 → 64, bias=False)
///   [18] Block                         # final ResBlock at 8×
///   [19] Conv2d(64 → 3, 3×3, padding=1)
/// </code>
/// Each <c>Block</c> is <c>x → fuse(conv(x) + skip(x))</c> where <c>conv</c> is
/// <c>Conv→ReLU→Conv→ReLU→Conv</c>, <c>skip</c> is <c>Identity</c> (all blocks are
/// 64→64 so the 1×1 skip conv is omitted), and <c>fuse</c> is a final <c>ReLU</c>.
/// Final output is clamped to <c>[0, 1]</c>.</para>
///
/// <para><b>Latent channel count</b> varies by family — SD/SDXL = 4, SD3/Flux/etc = 16,
/// Flux.2 = 32. The safetensors weight files are family-specific (taesd /
/// taesdxl / taesd3 / taef1). Pass the matching <see cref="LatentArchitecture"/> at load
/// time; the loader validates the conv-in weight shape against the expected channel count.</para>
///
/// <para><b>Weight key naming</b> follows PyTorch's <c>nn.Sequential</c> state-dict convention,
/// i.e. flat numeric indices: <c>1.weight</c> / <c>1.bias</c> for conv_in, <c>3.conv.0.weight</c>
/// for the first conv inside Block-1, <c>19.weight</c> / <c>19.bias</c> for conv_out, etc.
/// This matches the format Comfy loads via <c>taesd_decoder.load_state_dict(...)</c> directly.</para></summary>
public sealed unsafe class TaesdDecoder : IDisposable
{
    /// <summary>Sequential indices where the 10 ResBlocks live in the decoder graph.</summary>
    private static readonly int[] BlockSeqIndices = [3, 4, 5, 8, 9, 10, 13, 14, 15, 18];

    /// <summary>Sequential indices of the post-upsample 3×3 convs (no bias).</summary>
    private static readonly int[] UpsampleConvSeqIndices = [7, 12, 17];

    private readonly LatentArchitecture _arch;
    private readonly int _latentChannels;

    // The safetensors loader keeps the file mmap'd. Disposing it invalidates every
    // borrowed Tensor below — keep it alive for the decoder's lifetime.
    private SafeTensorsLoader? _loader;

    // 1×1 conv_in: shape [64, latent_C, 3, 3] + bias [64]
    private Tensor? _convInW;
    private Tensor? _convInB;

    // 10 blocks × 3 convs each, all 3×3 → 30 weight tensors + 30 bias tensors.
    private Tensor[] _blockConv0W = [];
    private Tensor[] _blockConv0B = [];
    private Tensor[] _blockConv2W = [];
    private Tensor[] _blockConv2B = [];
    private Tensor[] _blockConv4W = [];
    private Tensor[] _blockConv4B = [];

    // 3 post-upsample convs (no biases — bias=False in the reference).
    private Tensor[] _upsampleConvW = [];

    // conv_out: shape [3, 64, 3, 3] + bias [3]
    private Tensor? _convOutW;
    private Tensor? _convOutB;

    private int _disposed;

    private TaesdDecoder(LatentArchitecture arch, int latentChannels)
    {
        _arch = arch;
        _latentChannels = latentChannels;
    }

    public int LatentChannels => _latentChannels;
    public LatentArchitecture Architecture => _arch;

    /// <summary>Latent channel count expected for each architecture family. SD/SDXL/AuraFlow
    /// use the 4-channel SD VAE, SD3/Flux/Chroma/FLite/ZImage use 16-channel VAEs, and
    /// Flux.2 uses a 32-channel VAE.</summary>
    public static int GetLatentChannels(LatentArchitecture arch) => arch switch
    {
        LatentArchitecture.Sd15 or LatentArchitecture.Sdxl or LatentArchitecture.AuraFlow => 4,
        LatentArchitecture.Sd3 or LatentArchitecture.Flux or LatentArchitecture.Chroma
            or LatentArchitecture.FLite or LatentArchitecture.ZImage or LatentArchitecture.Anima => 16,
        LatentArchitecture.Flux2 => 32,
        _ => throw new ArgumentException($"TAESD not defined for arch {arch}.", nameof(arch)),
    };

    /// <summary>Loads a TAESD decoder from a safetensors file. The file must contain the
    /// decoder-only state dict (taesd_decoder.safetensors, taesdxl_decoder.safetensors, etc.).
    /// Throws if the conv_in channel count doesn't match the architecture's expected latent
    /// channels (e.g. loading a 4-channel SD checkpoint for SD3 which expects 16).</summary>
    public static TaesdDecoder LoadFromSafetensors(LatentArchitecture arch, string path)
    {
        int expectedChannels = GetLatentChannels(arch);
        TaesdDecoder dec = new(arch, expectedChannels);
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(path);
            dec._loader = loader;

            dec._convInW = loader.GetTensor("1.weight");
            dec._convInB = loader.GetTensor("1.bias");

            // Sanity-check the conv_in weight: [64, latent_C, 3, 3]. Wrong arch ↔ wrong file
            // mismatch is the most common user error here — fail fast with a clear message.
            if (dec._convInW.Shape.Rank != 4 || dec._convInW.Shape[1] != expectedChannels)
            {
                throw new InvalidOperationException(
                    $"TAESD weight shape mismatch for arch {arch}: conv_in weight has shape " +
                    $"{dec._convInW.Shape}, expected [64, {expectedChannels}, 3, 3]. " +
                    $"Most likely a TAESD checkpoint for a different architecture was passed.");
            }

            dec._blockConv0W = new Tensor[BlockSeqIndices.Length];
            dec._blockConv0B = new Tensor[BlockSeqIndices.Length];
            dec._blockConv2W = new Tensor[BlockSeqIndices.Length];
            dec._blockConv2B = new Tensor[BlockSeqIndices.Length];
            dec._blockConv4W = new Tensor[BlockSeqIndices.Length];
            dec._blockConv4B = new Tensor[BlockSeqIndices.Length];
            for (int b = 0; b < BlockSeqIndices.Length; b++)
            {
                int i = BlockSeqIndices[b];
                dec._blockConv0W[b] = loader.GetTensor($"{i}.conv.0.weight");
                dec._blockConv0B[b] = loader.GetTensor($"{i}.conv.0.bias");
                dec._blockConv2W[b] = loader.GetTensor($"{i}.conv.2.weight");
                dec._blockConv2B[b] = loader.GetTensor($"{i}.conv.2.bias");
                dec._blockConv4W[b] = loader.GetTensor($"{i}.conv.4.weight");
                dec._blockConv4B[b] = loader.GetTensor($"{i}.conv.4.bias");
            }

            dec._upsampleConvW = new Tensor[UpsampleConvSeqIndices.Length];
            for (int u = 0; u < UpsampleConvSeqIndices.Length; u++)
            {
                dec._upsampleConvW[u] = loader.GetTensor($"{UpsampleConvSeqIndices[u]}.weight");
            }

            dec._convOutW = loader.GetTensor("19.weight");
            dec._convOutB = loader.GetTensor("19.bias");
        }
        catch
        {
            loader.Dispose();
            dec._loader = null;
            throw;
        }
        return dec;
    }

    /// <summary>Runs the tiny decoder over <paramref name="latent"/> and returns an RGB tensor
    /// <c>[1, 3, H*8, W*8]</c> in <c>[0, 1]</c>. The caller owns the result and must dispose it.</summary>
    public Tensor Forward(IBackend backend, Tensor latent)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 4 || latent.Shape[1] != _latentChannels)
        {
            throw new ArgumentException(
                $"Latent shape {latent.Shape} doesn't match expected [B, {_latentChannels}, H, W] for arch {_arch}.",
                nameof(latent));
        }

        int batch = (int)latent.Shape[0];
        int h = (int)latent.Shape[2];
        int w = (int)latent.Shape[3];
        DType dtype = DType.F32;

        // Sequential[0] = Clamp: x ← tanh(x/3) * 3. Done in-place on a fresh F32 copy of the
        // latent so we never mutate the caller's tensor.
        Tensor x = new(new TensorShape(batch, _latentChannels, h, w), dtype);
        CopyLatentToF32(x, latent);
        SoftClampInPlace(x);

        // Sequential[1] = conv_in (3×3 padding=1, latent_C → 64)
        Tensor hidden = new(new TensorShape(batch, 64, h, w), dtype);
        backend.Conv2D(hidden, x, _convInW!, _convInB, 1, 1, 1, 1);
        x.Dispose();
        // [2] ReLU
        ReluInPlace(hidden);

        // [3..5] three blocks at native res, [6] upsample, [7] post-upsample conv (no bias)
        for (int b = 0; b < 3; b++) hidden = ApplyBlock(backend, hidden, b);
        hidden = Upsample2xAndConv(backend, hidden, 0);

        // [8..10] three blocks at 2×, [11] upsample, [12] post-upsample conv
        for (int b = 3; b < 6; b++) hidden = ApplyBlock(backend, hidden, b);
        hidden = Upsample2xAndConv(backend, hidden, 1);

        // [13..15] three blocks at 4×, [16] upsample, [17] post-upsample conv
        for (int b = 6; b < 9; b++) hidden = ApplyBlock(backend, hidden, b);
        hidden = Upsample2xAndConv(backend, hidden, 2);

        // [18] one final block at 8×
        hidden = ApplyBlock(backend, hidden, 9);

        // [19] conv_out (3×3 padding=1, 64 → 3)
        int finalH = h * 8;
        int finalW = w * 8;
        Tensor rgb = new(new TensorShape(batch, 3, finalH, finalW), dtype);
        backend.Conv2D(rgb, hidden, _convOutW!, _convOutB, 1, 1, 1, 1);
        hidden.Dispose();

        // Decoder is trained against [0, 1] RGB targets — clamp residual overshoot for display.
        ClampInPlace(rgb, 0f, 1f);
        return rgb;
    }

    /// <summary>Runs one ResBlock at the current resolution. Returns a fresh tensor and
    /// disposes <paramref name="input"/>. All 10 blocks are 64→64 so no shape change.</summary>
    private Tensor ApplyBlock(IBackend backend, Tensor input, int blockIdx)
    {
        TensorShape shape = input.Shape;
        DType dtype = input.DType;

        // conv.0 → ReLU
        Tensor c0 = new(shape, dtype);
        backend.Conv2D(c0, input, _blockConv0W[blockIdx], _blockConv0B[blockIdx], 1, 1, 1, 1);
        ReluInPlace(c0);
        // conv.2 → ReLU
        Tensor c2 = new(shape, dtype);
        backend.Conv2D(c2, c0, _blockConv2W[blockIdx], _blockConv2B[blockIdx], 1, 1, 1, 1);
        c0.Dispose();
        ReluInPlace(c2);
        // conv.4 (no activation; the post-skip "fuse" ReLU comes next)
        Tensor c4 = new(shape, dtype);
        backend.Conv2D(c4, c2, _blockConv4W[blockIdx], _blockConv4B[blockIdx], 1, 1, 1, 1);
        c2.Dispose();

        // Residual: out = ReLU(conv(x) + skip(x)) — skip is identity since channels match.
        Tensor sum = new(shape, dtype);
        backend.Add(sum, c4, input);
        c4.Dispose();
        input.Dispose();
        ReluInPlace(sum);
        return sum;
    }

    /// <summary>Sequential pattern <c>Upsample(2×) → Conv(3×3, no bias)</c>. Returns a tensor
    /// at 2× spatial resolution; disposes <paramref name="input"/>.</summary>
    private Tensor Upsample2xAndConv(IBackend backend, Tensor input, int upIdx)
    {
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int inH = (int)input.Shape[2];
        int inW = (int)input.Shape[3];
        DType dtype = input.DType;

        Tensor upsampled = new(new TensorShape(batch, channels, inH * 2, inW * 2), dtype);
        backend.UpsampleNearest2D(upsampled, input, 2, 2);
        input.Dispose();

        Tensor convOut = new(upsampled.Shape, dtype);
        backend.Conv2D(convOut, upsampled, _upsampleConvW[upIdx], bias: null, 1, 1, 1, 1);
        upsampled.Dispose();
        return convOut;
    }

    /// <summary>Copies <paramref name="src"/> into <paramref name="dst"/>, casting from F16/BF16
    /// to F32 if needed. The forward pass runs in F32 — the source latent might be F16 when it
    /// comes from a half-precision pipeline.</summary>
    private static void CopyLatentToF32(Tensor dst, Tensor src)
    {
        if (src.DType == DType.F32)
        {
            long bytes = src.ElementCount * sizeof(float);
            Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
            return;
        }
        // Half / BF16 → F32 element-wise. Small tensor, so the scalar loop is fine.
        if (src.DType == DType.F16)
        {
            Half* sp = (Half*)src.DataPointer;
            float* dp = (float*)dst.DataPointer;
            long n = src.ElementCount;
            for (long i = 0; i < n; i++) dp[i] = (float)sp[i];
            return;
        }
        throw new NotSupportedException($"Unsupported latent dtype for TAESD preview: {src.DType}.");
    }

    /// <summary>The published TAESD's input <c>Clamp()</c> module: <c>x ← tanh(x/3) * 3</c>.
    /// Acts as a soft saturation: leaves <c>|x| &lt; 1</c> nearly unchanged, smoothly compresses
    /// larger magnitudes so an outlier noisy latent can't drive the decoder to nonsense.</summary>
    private static void SoftClampInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            p[i] = MathF.Tanh(p[i] * (1.0f / 3.0f)) * 3.0f;
        }
    }

    /// <summary>In-place ReLU: <c>x ← max(x, 0)</c>. Implemented as a scalar loop rather than
    /// going through the backend because (a) <see cref="IBackend"/> has no ReLU primitive (it
    /// has SiLU / GELU but TAESD uses ReLU), and (b) the tensors here are small enough that
    /// the device→host round-trip cost would dwarf the work itself for GPU backends.</summary>
    private static void ReluInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            if (p[i] < 0f) p[i] = 0f;
        }
    }

    /// <summary>In-place clamp to <c>[min, max]</c> — used after the final conv to pin RGB
    /// output into the displayable range, since the decoder has no terminal activation.</summary>
    private static void ClampInPlace(Tensor t, float min, float max)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (v < min) v = min; else if (v > max) v = max;
            p[i] = v;
        }
    }

    /// <summary>Reads the <c>[0, 1]</c>-valued RGB tensor that <see cref="Forward"/> returned
    /// and emits HWC bytes (the layout <c>RgbToImage.FromHwcRgb</c> consumes on the SwarmUI
    /// side). Saves a redundant clamp+scale pass vs <see cref="ImagePostProcessor.TensorToRgbBytes"/>
    /// which expects <c>[-1, 1]</c> input.</summary>
    public static byte[] ToHwcRgbBytes(Tensor rgb01, out int width, out int height)
    {
        height = (int)rgb01.Shape[2];
        width = (int)rgb01.Shape[3];
        byte[] result = new byte[height * width * 3];
        float* p = (float*)rgb01.DataPointer;
        int plane = height * width;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pix = y * width + x;
                int outOff = pix * 3;
                float r = p[0 * plane + pix];
                float g = p[1 * plane + pix];
                float b = p[2 * plane + pix];
                result[outOff + 0] = (byte)(MathF.Min(MathF.Max(r, 0f), 1f) * 255f + 0.5f);
                result[outOff + 1] = (byte)(MathF.Min(MathF.Max(g, 0f), 1f) * 255f + 0.5f);
                result[outOff + 2] = (byte)(MathF.Min(MathF.Max(b, 0f), 1f) * 255f + 0.5f);
            }
        }
        return result;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Weight tensors are borrowed from the mmap — they don't own memory, so dispose only
        // the loader. The borrowed Tensor objects can be left for GC; they have no IDisposable
        // resources beyond the data pointer which the loader is responsible for.
        _loader?.Dispose();
        _loader = null;
    }
}
