using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>AutoencoderKL VAE decoder. Converts latent representations [B, C, H/8, W/8] to RGB images [B, 3, H, W]. Supports SD1.5, SDXL, SD3, and Flux configurations.</summary>
public sealed class VaeDecoder
{
    private readonly VaeConfig _config;

    // The decoder's stage widths: the encoder's, unless the checkpoint's decoder is narrower (FLUX.2 small decoder).
    private readonly int[] _decoderChannels;

    // post_quant_conv (optional, SD1.5/SDXL only)
    private Tensor? _postQuantConvWeight;
    private Tensor? _postQuantConvBias;

    // conv_in: Conv2d(latent_channels → block_out_channels[-1], 3x3, padding=1)
    private Tensor? _convInWeight;
    private Tensor? _convInBias;

    // mid_block: ResNet → Attention → ResNet
    private readonly ResNetBlock2D _midResNet0;
    private readonly VaeAttention _midAttention;
    private readonly ResNetBlock2D _midResNet1;

    // up_blocks: 4x UpDecoderBlock2D, each with (layers_per_block + 1) ResNets + optional upsample
    private readonly ResNetBlock2D[][] _upBlockResNets;
    private readonly Tensor?[] _upsampleWeights;
    private readonly Tensor?[] _upsampleBiases;

    // conv_norm_out + conv_out
    private Tensor? _normOutWeight;
    private Tensor? _normOutBias;
    private Tensor? _convOutWeight;
    private Tensor? _convOutBias;

    /// <summary>The configuration this decoder was built with.</summary>
    public VaeConfig Config => _config;

    /// <summary>Creates a VAE decoder with the specified configuration.</summary>
    public VaeDecoder(VaeConfig config)
    {
        _config = config;
        _decoderChannels = config.DecoderBlockOutChannels ?? config.BlockOutChannels;

        int[] blockChannels = _decoderChannels;
        int midChannels = blockChannels[^1];

        // Mid-block: two ResNet blocks with attention between them (all at midChannels)
        _midResNet0 = new ResNetBlock2D(midChannels, midChannels, config.NormNumGroups, config.NormEps);
        _midAttention = new VaeAttention(midChannels, config.NormNumGroups, config.NormEps);
        _midResNet1 = new ResNetBlock2D(midChannels, midChannels, config.NormNumGroups, config.NormEps);

        // Up-blocks: reversed channel order [512, 512, 256, 128]
        int numBlocks = blockChannels.Length;
        int resnetsPerBlock = config.LayersPerBlock + 1;
        _upBlockResNets = new ResNetBlock2D[numBlocks][];
        _upsampleWeights = new Tensor?[numBlocks];
        _upsampleBiases = new Tensor?[numBlocks];

        // Up-blocks process in reversed channel order
        int[] reversedChannels = new int[numBlocks];
        for (int i = 0; i < numBlocks; i++)
        {
            reversedChannels[i] = blockChannels[numBlocks - 1 - i];
        }

        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            int outCh = reversedChannels[blockIdx];
            _upBlockResNets[blockIdx] = new ResNetBlock2D[resnetsPerBlock];

            for (int resIdx = 0; resIdx < resnetsPerBlock; resIdx++)
            {
                // First ResNet in first block takes midChannels as input
                // First ResNet in subsequent blocks takes previous block's output channels
                int inCh;
                if (resIdx == 0)
                {
                    inCh = blockIdx == 0 ? midChannels : reversedChannels[blockIdx - 1];
                }
                else
                {
                    inCh = outCh;
                }
                _upBlockResNets[blockIdx][resIdx] = new ResNetBlock2D(inCh, outCh, config.NormNumGroups, config.NormEps);
            }
        }
    }

    /// <summary>Loads all decoder weights from a dictionary of named tensors. Keys should match diffusers naming convention.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        // post_quant_conv (optional)
        if (_config.UsePostQuantConv)
        {
            _postQuantConvWeight = weights["post_quant_conv.weight"];
            _postQuantConvBias = weights["post_quant_conv.bias"];
        }

        // conv_in: a checkpoint whose decoder is built for other widths would otherwise load silently and decode garbage.
        Tensor convIn = weights["decoder.conv_in.weight"];
        if (convIn.Shape[0] != _decoderChannels[^1])
        {
            throw new InvalidDataException(
                $"VAE decoder conv_in has {convIn.Shape[0]} channels but the decoder is built for {_decoderChannels[^1]}; "
                + "the checkpoint's decoder widths do not match this VaeConfig.");
        }
        _convInWeight = convIn;
        _convInBias = weights["decoder.conv_in.bias"];

        // mid_block
        _midResNet0.LoadWeights(weights, "decoder.mid_block.resnets.0");
        _midAttention.LoadWeights(weights, "decoder.mid_block.attentions.0");
        _midResNet1.LoadWeights(weights, "decoder.mid_block.resnets.1");

        // up_blocks
        for (int blockIdx = 0; blockIdx < _upBlockResNets.Length; blockIdx++)
        {
            for (int resIdx = 0; resIdx < _upBlockResNets[blockIdx].Length; resIdx++)
            {
                _upBlockResNets[blockIdx][resIdx].LoadWeights(weights, $"decoder.up_blocks.{blockIdx}.resnets.{resIdx}");
            }

            // Upsample exists on all blocks except the last one
            if (blockIdx < _upBlockResNets.Length - 1)
            {
                _upsampleWeights[blockIdx] = weights[$"decoder.up_blocks.{blockIdx}.upsamplers.0.conv.weight"];
                _upsampleBiases[blockIdx] = weights[$"decoder.up_blocks.{blockIdx}.upsamplers.0.conv.bias"];
            }
        }

        // conv_norm_out + conv_out
        _normOutWeight = weights["decoder.conv_norm_out.weight"];
        _normOutBias = weights["decoder.conv_norm_out.bias"];
        _convOutWeight = weights["decoder.conv_out.weight"];
        _convOutBias = weights["decoder.conv_out.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this decoder and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_postQuantConvWeight is not null) yield return _postQuantConvWeight;
        if (_postQuantConvBias is not null) yield return _postQuantConvBias;
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor w in _midResNet0.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midAttention.EnumerateWeights()) yield return w;
        foreach (Tensor w in _midResNet1.EnumerateWeights()) yield return w;
        for (int blockIdx = 0; blockIdx < _upBlockResNets.Length; blockIdx++)
        {
            for (int resIdx = 0; resIdx < _upBlockResNets[blockIdx].Length; resIdx++)
            {
                foreach (Tensor w in _upBlockResNets[blockIdx][resIdx].EnumerateWeights()) yield return w;
            }
            if (_upsampleWeights[blockIdx] is not null) yield return _upsampleWeights[blockIdx]!;
            if (_upsampleBiases[blockIdx] is not null) yield return _upsampleBiases[blockIdx]!;
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
    }

    /// <summary>Decodes latent tensor [B, latentCh, H, W] to RGB image [B, 3, H*8, W*8]. Applies inverse scaling before decoding.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        // 1. Undo scaling: z = latent / scaling_factor + shift_factor
        Tensor z = UndoScaling(backend, latent);

        DType dtype = z.DType;

        // 2. Optional post_quant_conv (1x1 conv)
        if (_config.UsePostQuantConv)
        {
            Tensor quantOut = new Tensor(z.Shape, dtype);
            backend.Conv2D(quantOut, z, _postQuantConvWeight!, _postQuantConvBias, 1, 1, 0, 0);
            z.Dispose();
            z = quantOut;
        }

        // 3. conv_in: [B, latentCh, H, W] → [B, 512, H, W]
        int batch = (int)z.Shape[0];
        int h = (int)z.Shape[2];
        int w = (int)z.Shape[3];
        int midCh = _decoderChannels[^1];

        TensorShape convInOutShape = new TensorShape(batch, midCh, h, w);
        Tensor hidden = new Tensor(convInOutShape, dtype);
        backend.Conv2D(hidden, z, _convInWeight!, _convInBias, 1, 1, 1, 1);
        z.Dispose();

        // 4. Mid-block: ResNet → Attention → ResNet
        Tensor midOut0 = _midResNet0.Forward(backend, hidden);
        hidden.Dispose();

        Tensor attnOut = _midAttention.Forward(backend, midOut0);
        midOut0.Dispose();

        Tensor midOut1 = _midResNet1.Forward(backend, attnOut);
        attnOut.Dispose();

        hidden = midOut1;

        // 5. Up-blocks
        int[] reversedChannels = new int[_decoderChannels.Length];
        for (int i = 0; i < reversedChannels.Length; i++)
        {
            reversedChannels[i] = _decoderChannels[^(i + 1)];
        }

        for (int blockIdx = 0; blockIdx < _upBlockResNets.Length; blockIdx++)
        {
            // ResNet layers
            for (int resIdx = 0; resIdx < _upBlockResNets[blockIdx].Length; resIdx++)
            {
                Tensor resOut = _upBlockResNets[blockIdx][resIdx].Forward(backend, hidden);
                hidden.Dispose();
                hidden = resOut;
            }

            // Upsample (all blocks except last): nearest-neighbor 2x → Conv2d(3x3)
            if (blockIdx < _upBlockResNets.Length - 1)
            {
                int curH = (int)hidden.Shape[2];
                int curW = (int)hidden.Shape[3];
                int curCh = (int)hidden.Shape[1];

                TensorShape upShape = new TensorShape(batch, curCh, curH * 2, curW * 2);
                Tensor upsampled = new Tensor(upShape, dtype);
                backend.UpsampleNearest2D(upsampled, hidden, 2, 2);
                hidden.Dispose();

                Tensor convUp = new Tensor(upShape, dtype);
                backend.Conv2D(convUp, upsampled, _upsampleWeights[blockIdx]!, _upsampleBiases[blockIdx], 1, 1, 1, 1);
                upsampled.Dispose();

                hidden = convUp;
            }
        }

        // 6. Fused GroupNorm+SiLU → conv_out
        int finalH = (int)hidden.Shape[2];
        int finalW = (int)hidden.Shape[3];
        int finalCh = _decoderChannels[0];

        TensorShape finalShape = new TensorShape(batch, finalCh, finalH, finalW);
        Tensor siluOut = new Tensor(finalShape, dtype);
        backend.GroupNormSilu(siluOut, hidden, _normOutWeight!, _normOutBias!, _config.NormNumGroups, _config.NormEps);
        hidden.Dispose();

        TensorShape rgbShape = new TensorShape(batch, 3, finalH, finalW);
        Tensor convOut = new Tensor(rgbShape, dtype);
        backend.Conv2D(convOut, siluOut, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        siluOut.Dispose();

        // Cast final output to F32 for RGB post-processing. TensorToRgbBytes reads
        // DataPointer as float*, so any non-F32 dtype produces garbage if returned as-is.
        // EnsureF32 returns convOut unchanged when already F32, otherwise allocates F32 and
        // disposes the source — covers both F16 and BF16 paths.
        return Utilities.DtypeCastHelper.EnsureF32(backend, convOut);
    }

    /// <summary>Spatial scale factor applied by the decoder (latent → RGB pixel ratio).
    /// AutoencoderKL is always 8×, but exposed here so callers don't hardcode it.</summary>
    public const int SpatialScale = 8;

    /// <summary>Kill switch (vram.vaeFullres=false) for the full-resolution direct-decode attempt in
    /// <see cref="DecodeTiled"/>, forcing the always-tiled behavior.</summary>
    private static bool FullResEnabled => EngineKnobs.VaeFullres.Value;

    /// <summary>Smallest latent (batch × height × width) whose full-resolution attempt failed this session; latents
    /// at least that large tile without trying, smaller ones still attempt.</summary>
    private long _fullResFailedArea = long.MaxValue;

    /// <summary>Test seam: an exception returned here is thrown inside the full-resolution attempt.</summary>
    internal Func<Exception?>? FullResFailureForTests { get; set; }

    /// <summary>Largest single im2col workspace the full-res attempt may require before it falls back to tiling
    /// without trying. 10 GB covers 1024² output for an F32 [128,256,512,512] AutoencoderKL (peak 9.66 GB) while
    /// rejecting 2048²+ (≥4× that).</summary>
    private const long FullResWorkspaceCapBytes = 10L << 30;

    /// <summary>Decodes the latent in spatial tiles and blends overlapping RGB regions,
    /// rather than running the full convolution stack on the entire latent at once.
    /// Solves the catastrophic im2col workspace blow-up (e.g. ~9.7 GB for a 256ch 3×3
    /// conv at 1024² output): each tile's worst-case workspace scales with the tile area
    /// instead of the full image area, so a 64-latent / 512-RGB tile fits comfortably in
    /// any modern GPU regardless of final resolution.
    ///
    /// <para>Tile/overlap defaults match ComfyUI's tiled decode: 64 latent (=512 RGB) tile,
    /// 8 latent (=64 RGB) overlap. Overlap regions are blended with a tent-function
    /// weight (linear ramp from 0 at tile edge to 1 at <c>overlapLatent</c> px in),
    /// then divided by the per-pixel weight sum so the result is a proper weighted
    /// average regardless of how many tiles overlap any given pixel.</para>
    /// </summary>
    /// <param name="backend">Compute backend.</param>
    /// <param name="latent">Full latent <c>[B, latentCh, H, W]</c>. Any dtype accepted; the
    /// per-tile decode internally runs at F32 for numerical safety (F16 GroupNorm/softmax
    /// precision in the VAE has been observed to produce all-black output on some
    /// checkpoints — needs investigation before F16 can be re-enabled in this path).</param>
    /// <param name="tileLatentSize">Latent edge size of each tile, before overlap is
    /// subtracted. RGB tile size = <c>tileLatentSize × 8</c>. Default 64.</param>
    /// <param name="overlapLatent">Number of latent pixels overlapping between adjacent
    /// tiles. RGB overlap = <c>overlapLatent × 8</c>. Default 8.</param>
    public unsafe Tensor DecodeTiled(IBackend backend, Tensor latent, int tileLatentSize = 64, int overlapLatent = 8)
    {
        if (tileLatentSize <= overlapLatent || overlapLatent < 0)
        {
            throw new ArgumentException(
                $"tileLatentSize ({tileLatentSize}) must be > overlapLatent ({overlapLatent}) and overlap must be non-negative.");
        }

        int batch = (int)latent.Shape[0];
        int latentCh = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];

        // Fast path: latent already fits in one tile. Match the latent dtype to the VAE weight
        // dtype (callers now pass an F32 latent; Decode's per-op dispatch needs the input to match
        // the weights — F32 latent on BF16 weights would read garbage). No-op when already equal.
        DType weightDtype = EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32;
        if (latentH <= tileLatentSize && latentW <= tileLatentSize)
        {
            Tensor single = Utilities.DtypeCastHelper.EnsureDtype(backend, latent, weightDtype, disposeSourceOnCast: false);
            Tensor decoded = Decode(backend, single);
            if (!ReferenceEquals(single, latent)) single.Dispose();
            return decoded;
        }

        // Full-resolution direct decode: tiling exists only to bound Conv2D's im2col workspace, but it costs
        // real time — at 1024² (128 latent, 64 tile, 8 overlap) it is 9 overlapping tiles = ~2.25× redundant
        // GPU decode plus per-tile D2H drains and a multi-megapixel host blend/normalize loop (measured ~1.5 s
        // vs ~0.3 s for a comparable full-res VAE). When the estimated worst-case im2col buffer is within
        // budget, decode the whole latent in one pass (the same code path the single-tile case runs). OOM falls
        // back to tiles for this latent size and larger for the rest of the session — the allocator already syncs
        // + trims the pool before giving up, so a genuine failure means the VRAM truly isn't there.
        int gemmElemSize = Math.Max(2, weightDtype.SizeInBytes);   // fp8 weights GEMM at 16-bit
        long fullResWorkspace = EstimateDecodeWorkspaceBytes(latentH, latentW, gemmElemSize);
        // A banding backend caps every conv's im2col at its workspace cap — the naive per-conv estimate above
        // then overstates the real peak (e.g. Flux.2's 512-ch conv at 1024²: 9.2 GB naive vs a 2 GB band).
        if (backend.Capabilities.BandsIm2Col && backend.Capabilities.Im2ColWorkspaceCapBytes > 0)
            fullResWorkspace = Math.Min(fullResWorkspace, backend.Capabilities.Im2ColWorkspaceCapBytes);
        // Post-trim headroom the attempt needs beyond the single worst im2col: neighboring layer activations
        // plus a second in-flight im2col whose async free hasn't recycled yet. Below this, skip to tiled for
        // THIS generation only — a resident-DiT host (vram.keepModels) may free VRAM later, and a skipped
        // attempt costs nothing, unlike an OOM-failed one (~0.5 s + tiled at that size for the session).
        const long FullResHeadroomBytes = 3L << 29;   // 1.5 GB
        long latentArea = (long)batch * latentH * latentW;
        if (FullResEnabled && latentArea < _fullResFailedArea && fullResWorkspace <= FullResWorkspaceCapBytes)
        {
            // Pre-trim when the pool's reservations would OOM the big im2col: a planned trim here is cheap, vs. the
            // allocator's OOM-retry mid-decode which syncs a deep conv queue (measured ~+470 ms/gen).
            if (backend.FreeMemoryBytes() < fullResWorkspace + FullResHeadroomBytes)
                backend.TrimMemoryPool();
            long postTrimFree = backend.FreeMemoryBytes();
            if (postTrimFree < fullResWorkspace + FullResHeadroomBytes)
            {
                Logs.Verbose($"[vae-phase] full-res decode skipped this gen: post-trim free {postTrimFree >> 20} MB " +
                    $"< im2col est {fullResWorkspace >> 20} MB + headroom; using tiles");
            }
            else
            {
                System.Diagnostics.Stopwatch fullSw = System.Diagnostics.Stopwatch.StartNew();
                Tensor? single = null;
                try
                {
                    single = Utilities.DtypeCastHelper.EnsureDtype(backend, latent, weightDtype, disposeSourceOnCast: false);
                    if (FullResFailureForTests?.Invoke() is Exception injected)
                        throw injected;
                    Tensor decoded = Decode(backend, single);
                    if (!ReferenceEquals(single, latent)) single.Dispose();
                    Logs.Verbose($"[vae-phase] full-res decode {latentH}x{latentW} latent in {fullSw.ElapsedMilliseconds}ms " +
                        $"(im2col est {fullResWorkspace >> 20} MB)");
                    return decoded;
                }
                catch (Exception ex)
                {
                    // Almost certainly VRAM exhaustion on the im2col workspace (despite the headroom check —
                    // e.g. fragmentation). Clean up whatever the partial decode left on-device and fall back
                    // to the tiled path (and stay tiled at this size this session).
                    _fullResFailedArea = Math.Min(_fullResFailedArea, latentArea);
                    Logs.Warning($"[vae-phase] full-res decode failed after {fullSw.ElapsedMilliseconds}ms — " +
                        $"falling back to tiled for {latentH}x{latentW} latents and larger this session: {ex.Message}");
                    if (single is not null && !ReferenceEquals(single, latent)) single.Dispose();
                    // FreeActivations drops device copies without a sync-back and the tiles slice the latent on
                    // the host, so a device-resident latent would otherwise decode as zeros (a flat gray image).
                    backend.OffloadActivation(latent);
                    backend.FreeActivations();
                }
            }
        }

        return VaeTileAccumulator.Decode(backend, latent, tileLatentSize, overlapLatent, weightDtype, SpatialScale,
            t => Decode(backend, t),
            (h, w) => EstimateDecodeWorkspaceBytes(h, w, gemmElemSize));
    }

    /// <summary>Walks the decoder's conv plan (mid → up-blocks with 2× upsamples → conv_out) and returns the
    /// largest single im2col workspace (bytes) <see cref="Decode"/> will ask Conv2D for at this latent size.
    /// Every conv here is 3×3, so workspace = inCh · 9 · outH · outW · elemSize; the peak is the first conv at
    /// full output resolution that still carries the previous block's channel count.</summary>
    private long EstimateDecodeWorkspaceBytes(int latentH, int latentW, int elemSize)
    {
        int[] ch = _decoderChannels;
        int numBlocks = ch.Length;
        long h = latentH, w = latentW;
        long prevCh = ch[^1];
        long worstElems = prevCh * 9 * h * w;   // conv_in + mid-block at latent resolution
        for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++)
        {
            long outCh = ch[numBlocks - 1 - blockIdx];
            worstElems = Math.Max(worstElems, Math.Max(prevCh, outCh) * 9 * h * w);
            if (blockIdx < numBlocks - 1)
            {
                h *= 2;
                w *= 2;
                worstElems = Math.Max(worstElems, outCh * 9 * h * w);   // upsampler conv at the doubled resolution
            }
            prevCh = outCh;
        }
        worstElems = Math.Max(worstElems, prevCh * 9 * h * w);   // conv_out at full resolution
        return worstElems * elemSize;
    }

    /// <summary>Undoes the latent scaling: z = latent / scaling_factor + shift_factor.</summary>
    private Tensor UndoScaling(IBackend backend, Tensor latent)
    {
        DType dtype = latent.DType;
        float invScale = 1.0f / _config.ScalingFactor;
        Tensor scaled = new Tensor(latent.Shape, dtype);
        backend.Scale(scaled, latent, invScale);

        if (_config.ShiftFactor.HasValue)
        {
            // Add shift_factor to every element
            Tensor shiftTensor = new Tensor(latent.Shape, dtype);
            backend.Fill(shiftTensor, _config.ShiftFactor.Value);

            Tensor shifted = new Tensor(latent.Shape, dtype);
            backend.Add(shifted, scaled, shiftTensor);
            scaled.Dispose();
            shiftTensor.Dispose();

            return shifted;
        }

        return scaled;
    }
}
