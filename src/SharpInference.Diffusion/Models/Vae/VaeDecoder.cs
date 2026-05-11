using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>AutoencoderKL VAE decoder. Converts latent representations [B, C, H/8, W/8] to RGB images [B, 3, H, W]. Supports SD1.5, SDXL, SD3, and Flux configurations.</summary>
public sealed class VaeDecoder
{
    private readonly VaeConfig _config;

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

        int[] blockChannels = config.BlockOutChannels;
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

        // conv_in
        _convInWeight = weights["decoder.conv_in.weight"];
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
        int midCh = _config.BlockOutChannels[^1];

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
        int[] reversedChannels = new int[_config.BlockOutChannels.Length];
        for (int i = 0; i < reversedChannels.Length; i++)
        {
            reversedChannels[i] = _config.BlockOutChannels[^(i + 1)];
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
        int finalCh = _config.BlockOutChannels[0];

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
        // Both F16 and BF16 paths must cast back here.
        if (dtype == DType.F16 || dtype == DType.BF16)
        {
            Tensor outputF32 = new Tensor(rgbShape, DType.F32);
            backend.CastToF32(outputF32, convOut);
            convOut.Dispose();
            return outputF32;
        }

        return convOut;
    }

    /// <summary>Spatial scale factor applied by the decoder (latent → RGB pixel ratio).
    /// AutoencoderKL is always 8×, but exposed here so callers don't hardcode it.</summary>
    public const int SpatialScale = 8;

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

        // Fast path: latent already fits in one tile, just decode it directly at its
        // native dtype (no cast).
        if (latentH <= tileLatentSize && latentW <= tileLatentSize)
        {
            return Decode(backend, latent);
        }

        int rgbH = latentH * SpatialScale;
        int rgbW = latentW * SpatialScale;
        int rgbTileSize = tileLatentSize * SpatialScale;
        int rgbOverlap = overlapLatent * SpatialScale;
        int strideLatent = tileLatentSize - overlapLatent;

        int nTilesH = Math.Max(1, (int)Math.Ceiling((double)(latentH - overlapLatent) / strideLatent));
        int nTilesW = Math.Max(1, (int)Math.Ceiling((double)(latentW - overlapLatent) / strideLatent));

        Logs.Info($"VAE tiled decode: {latentH}x{latentW} latent → {rgbH}x{rgbW} RGB, " +
            $"{nTilesH}x{nTilesW} tiles ({tileLatentSize} latent / {rgbTileSize} RGB, overlap {overlapLatent}/{rgbOverlap})");

        // Output accumulator (F32) and per-pixel weight accumulator (F32).
        // Both are CPU tensors — the per-tile decode runs on GPU and we accumulate on
        // CPU because each tile's contribution writes into different overlapping regions
        // and a GPU-side scatter would need extra kernels we don't have yet. The
        // accumulator buffers are tiny vs the savings (12 MB for 1024² RGB output).
        TensorShape outShape = new TensorShape(batch, 3, rgbH, rgbW);
        Tensor accumOutput = new Tensor(outShape, DType.F32);
        float* accumPtr = (float*)accumOutput.DataPointer;
        for (long i = 0; i < accumOutput.ElementCount; i++) accumPtr[i] = 0f;

        // Single weight plane shared across the 3 channels.
        long weightCount = (long)rgbH * rgbW;
        float[] weightAccum = new float[weightCount];

        // Pre-read the source latent as a CPU float pointer so we can slice quickly.
        if (latent.DType != DType.F32)
        {
            // Decode contracts on F32-or-F16 CPU latents. A non-F32 latent gets converted
            // up so we don't have to teach the slice loop multiple dtypes.
            Tensor latentF32 = new Tensor(latent.Shape, DType.F32);
            backend.CastToF32(latentF32, latent);
            latent = latentF32;
        }
        float* latentPtr = (float*)latent.DataPointer;

        // The VAE weights are loaded at a specific dtype (BF16 on Ampere+ for SDXL,
        // F32 elsewhere). After we slice an F32 tile out of the latent, we must cast
        // the slice to match the weight dtype before calling Decode â otherwise
        // Decode's per-op dispatch picks the F32 kernel which reads BF16 weights as
        // F32 garbage. This is the analog of the SdxlPipeline match-to-VAE cast, but
        // at per-tile granularity. Sample dtype from any weight tensor.
        DType vaeDtype = EnumerateWeights().FirstOrDefault()?.DType ?? DType.F32;

        for (int ty = 0; ty < nTilesH; ty++)
        {
            int latentY = Math.Min(ty * strideLatent, latentH - tileLatentSize);
            if (latentY < 0) latentY = 0;
            int actualTileH = Math.Min(tileLatentSize, latentH - latentY);

            for (int tx = 0; tx < nTilesW; tx++)
            {
                int latentX = Math.Min(tx * strideLatent, latentW - tileLatentSize);
                if (latentX < 0) latentX = 0;
                int actualTileW = Math.Min(tileLatentSize, latentW - latentX);

                // 1. Slice the latent into a contiguous tile [B, C, actualTileH, actualTileW].
                TensorShape sliceShape = new TensorShape(batch, latentCh, actualTileH, actualTileW);
                Tensor latentSlice = new Tensor(sliceShape, DType.F32);
                float* slicePtr = (float*)latentSlice.DataPointer;
                for (int b = 0; b < batch; b++)
                {
                    for (int c = 0; c < latentCh; c++)
                    {
                        for (int yy = 0; yy < actualTileH; yy++)
                        {
                            long srcRow = ((long)b * latentCh + c) * latentH * latentW + (long)(latentY + yy) * latentW + latentX;
                            long dstRow = ((long)b * latentCh + c) * actualTileH * actualTileW + (long)yy * actualTileW;
                            for (int xx = 0; xx < actualTileW; xx++)
                            {
                                slicePtr[dstRow + xx] = latentPtr[srcRow + xx];
                            }
                        }
                    }
                }

                // 2. Decode the tile. We need to feed Decode at the same dtype its
                // weights are loaded in â otherwise the per-op dispatch picks the wrong
                // kernel and reads the weights at the wrong width. Cast the F32 slice to
                // vaeDtype if they differ.
                Tensor tileInput = latentSlice;
                if (vaeDtype != DType.F32)
                {
                    tileInput = new Tensor(sliceShape, vaeDtype);
                    if (vaeDtype == DType.F16) backend.CastToF16(tileInput, latentSlice);
                    else if (vaeDtype == DType.BF16) backend.CastToBf16(tileInput, latentSlice);
                    else throw new InvalidOperationException($"Unsupported VAE dtype: {vaeDtype}");
                    latentSlice.Dispose();
                }
                Tensor rgbTile = Decode(backend, tileInput);
                tileInput.Dispose();

                int rgbTileH = actualTileH * SpatialScale;
                int rgbTileW = actualTileW * SpatialScale;
                int rgbY = latentY * SpatialScale;
                int rgbX = latentX * SpatialScale;
                bool firstTileX = (tx == 0);
                bool lastTileX = (tx == nTilesW - 1);
                bool firstTileY = (ty == 0);
                bool lastTileY = (ty == nTilesH - 1);

                // 3. Blend weight + tile contribution into the accumulator.
                float* tilePtr = (float*)rgbTile.DataPointer;
                for (int yy = 0; yy < rgbTileH; yy++)
                {
                    int outY = rgbY + yy;
                    float wY = ComputeBlendWeight(yy, rgbTileH, rgbOverlap, firstTileY, lastTileY);
                    for (int xx = 0; xx < rgbTileW; xx++)
                    {
                        int outX = rgbX + xx;
                        float wX = ComputeBlendWeight(xx, rgbTileW, rgbOverlap, firstTileX, lastTileX);
                        float w = wY * wX;
                        long outPixIdx = (long)outY * rgbW + outX;
                        weightAccum[outPixIdx] += w;
                        for (int c = 0; c < 3; c++)
                        {
                            long outIdx = ((long)c * rgbH + outY) * rgbW + outX;
                            long tileIdx = ((long)c * rgbTileH + yy) * rgbTileW + xx;
                            accumPtr[outIdx] += tilePtr[tileIdx] * w;
                        }
                    }
                }
                rgbTile.Dispose();
            }
        }

        // 4. Normalize by the weight accumulator. Pixels at the very edges of the
        // image have weightSum=1 (single tile contribution); interior pixels are
        // weighted averages of multiple tiles.
        for (int c = 0; c < 3; c++)
        {
            for (long i = 0; i < weightCount; i++)
            {
                long outIdx = (long)c * weightCount + i;
                float w = weightAccum[i];
                if (w > 1e-8f) accumPtr[outIdx] /= w;
            }
        }

        return accumOutput;
    }

    /// <summary>Tent function: ramps linearly from 0 at the tile edge to 1 at
    /// <paramref name="overlap"/> pixels in, then stays at 1 through the interior, then
    /// ramps back to 0 at the opposite edge. Edges that touch the image border get full
    /// weight (no ramp) since there's no neighbor to blend with — without this, image
    /// edges would be darkened by division.</summary>
    private static float ComputeBlendWeight(int idx, int tileSize, int overlap, bool isFirstTile, bool isLastTile)
    {
        if (overlap <= 0) return 1f;
        float left = isFirstTile ? 1f : Math.Min(1f, (idx + 1) / (float)overlap);
        float right = isLastTile ? 1f : Math.Min(1f, (tileSize - idx) / (float)overlap);
        return Math.Min(left, right);
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
