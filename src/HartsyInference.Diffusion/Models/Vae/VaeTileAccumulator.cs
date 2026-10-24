using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Overlapping-tile decode shared by the VAE decoders: slices the latent into tiles, runs <c>decodeTile</c> on
/// each, and blends the RGB results with tent weights into host F32 accumulators. Tile geometry and the tent weights are
/// the ones <see cref="VaeDecoder.DecodeTiled"/> has always used; the caller supplies the per-tile decode and its workspace estimate.</summary>
internal static class VaeTileAccumulator
{
    /// <summary>Decodes <paramref name="latent"/> ([B, 3-ch-compatible C, H, W]) tile by tile into [B, 3, H*spatialScale, W*spatialScale].</summary>
    /// <param name="decodeTile">Decodes one latent tile (already in <paramref name="weightDtype"/>) to RGB [B, 3, h*s, w*s].</param>
    /// <param name="tileWorkspaceEstimate">Worst-case im2col workspace bytes for a latent tile of (h, w), before the backend band cap.</param>
    public static unsafe Tensor Decode(
        IBackend backend, Tensor latent, int tileLatentSize, int overlapLatent, DType weightDtype, int spatialScale,
        Func<Tensor, Tensor> decodeTile, Func<int, int, long> tileWorkspaceEstimate)
    {
        int batch = (int)latent.Shape[0];
        int latentCh = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];

        int rgbH = latentH * spatialScale;
        int rgbW = latentW * spatialScale;
        int rgbTileSize = tileLatentSize * spatialScale;
        int rgbOverlap = overlapLatent * spatialScale;
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
        // Decode contracts on F32 CPU latents — a non-F32 latent gets converted up so we
        // don't have to teach the slice loop multiple dtypes. Don't dispose the source: the
        // caller still owns it, and replacing `latent` with a new F32 tensor here means we
        // own the new one (it goes out of scope at method end and gets GC'd via Tensor's
        // finalizer if we don't dispose — but it's small relative to the per-tile work).
        latent = Utilities.DtypeCastHelper.EnsureF32(backend, latent, disposeSourceOnCast: false);
        float* latentPtr = (float*)latent.DataPointer;

        // The VAE weights are loaded at a specific dtype (BF16 on Ampere+ for SDXL,
        // F32 elsewhere). After we slice an F32 tile out of the latent, we must cast
        // the slice to match the weight dtype before calling Decode â otherwise
        // Decode's per-op dispatch picks the F32 kernel which reads BF16 weights as
        // F32 garbage. This is the analog of the SdxlPipeline match-to-VAE cast, but
        // at per-tile granularity.
        DType vaeDtype = weightDtype;

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

                if (EngineKnobs.VaeStats.Value && ty == 0 && tx == 0)
                {
                    float dbgMin = float.MaxValue, dbgMax = float.MinValue;
                    double dbgSum = 0;
                    long dbgNan = 0;
                    for (long di = 0; di < latentSlice.ElementCount; di++)
                    {
                        float v = slicePtr[di];
                        if (float.IsNaN(v)) { dbgNan++; continue; }
                        if (v < dbgMin) dbgMin = v;
                        if (v > dbgMax) dbgMax = v;
                        dbgSum += v;
                    }
                    Logs.Info($"[TILEDBG] latentSlice(0,0) min={dbgMin} max={dbgMax} mean={dbgSum / latentSlice.ElementCount} nan={dbgNan} elems={latentSlice.ElementCount}");
                }

                // 2. Decode the tile. We need to feed Decode at the same dtype its
                // weights are loaded in — otherwise the per-op dispatch picks the wrong
                // kernel and reads the weights at the wrong width. Cast the F32 slice to
                // vaeDtype if they differ (no-op when already F32).
                Tensor tileInput = Utilities.DtypeCastHelper.EnsureDtype(backend, latentSlice, vaeDtype);
                if (EngineKnobs.VaeStats.Value && ty == 0 && tx == 0)
                {
                    Tensor tileInputF32 = tileInput.DType == DType.F32 ? tileInput : Utilities.DtypeCastHelper.EnsureDtype(backend, tileInput, DType.F32, disposeSourceOnCast: false);
                    float* dbgPtr2 = (float*)tileInputF32.DataPointer;
                    float dbgMin = float.MaxValue, dbgMax = float.MinValue;
                    double dbgSum = 0;
                    long dbgNan = 0;
                    for (long di = 0; di < tileInputF32.ElementCount; di++)
                    {
                        float v = dbgPtr2[di];
                        if (float.IsNaN(v)) { dbgNan++; continue; }
                        if (v < dbgMin) dbgMin = v;
                        if (v > dbgMax) dbgMax = v;
                        dbgSum += v;
                    }
                    Logs.Info($"[TILEDBG] tileInput(0,0) [{vaeDtype.Name}] min={dbgMin} max={dbgMax} mean={dbgSum / tileInputF32.ElementCount} nan={dbgNan} elems={tileInputF32.ElementCount}");
                    if (!ReferenceEquals(tileInputF32, tileInput)) tileInputF32.Dispose();
                }
                // Pre-flight: estimate THIS tile's im2col workspace and trim the pool proactively when it's
                // close to the edge, rather than letting the allocator discover it's short mid-decode and pay
                // the OOM-retry sync (~470ms, see the full-res gate above for the same tradeoff at latent scale).
                // Tile-scale headroom is much smaller than the full-res gate's 1.5GB — a single tile's own
                // workspace is the dominant term here, not neighboring-layer activation slop.
                long tileWorkspace = tileWorkspaceEstimate(actualTileH, actualTileW);
                if (backend.Capabilities.BandsIm2Col && backend.Capabilities.Im2ColWorkspaceCapBytes > 0)
                    tileWorkspace = Math.Min(tileWorkspace, backend.Capabilities.Im2ColWorkspaceCapBytes);
                const long TileHeadroomBytes = 1L << 27;   // 128 MB
                if (backend.FreeMemoryBytes() < tileWorkspace + TileHeadroomBytes)
                {
                    backend.TrimMemoryPool();
                }
                Tensor rgbTile = decodeTile(tileInput);
                tileInput.Dispose();
                if (EngineKnobs.VaeStats.Value && ty == 0 && tx == 0)
                {
                    float* dbgPtr = (float*)rgbTile.DataPointer;
                    float dbgMin = float.MaxValue, dbgMax = float.MinValue;
                    double dbgSum = 0;
                    for (long di = 0; di < rgbTile.ElementCount; di++)
                    {
                        float v = dbgPtr[di];
                        if (v < dbgMin) dbgMin = v;
                        if (v > dbgMax) dbgMax = v;
                        dbgSum += v;
                    }
                    Logs.Info($"[TILEDBG] tile(0,0) rgbTile min={dbgMin} max={dbgMax} mean={dbgSum / rgbTile.ElementCount} dtype={rgbTile.DType.Name} elems={rgbTile.ElementCount}");
                }

                int rgbTileH = actualTileH * spatialScale;
                int rgbTileW = actualTileW * spatialScale;
                int rgbY = latentY * spatialScale;
                int rgbX = latentX * spatialScale;
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

        if (EngineKnobs.VaeStats.Value)
        {
            float dbgMin = float.MaxValue, dbgMax = float.MinValue;
            double dbgSum = 0;
            long dbgNanCount = 0;
            for (long di = 0; di < accumOutput.ElementCount; di++)
            {
                float v = accumPtr[di];
                if (float.IsNaN(v)) { dbgNanCount++; continue; }
                if (v < dbgMin) dbgMin = v;
                if (v > dbgMax) dbgMax = v;
                dbgSum += v;
            }
            float wMin = float.MaxValue, wMax = float.MinValue;
            for (long di = 0; di < weightCount; di++)
            {
                if (weightAccum[di] < wMin) wMin = weightAccum[di];
                if (weightAccum[di] > wMax) wMax = weightAccum[di];
            }
            Logs.Info($"[TILEDBG] accumOutput min={dbgMin} max={dbgMax} mean={dbgSum / accumOutput.ElementCount} nanCount={dbgNanCount} weightMin={wMin} weightMax={wMax}");
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
}
