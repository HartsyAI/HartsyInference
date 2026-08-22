using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>Merges LLaVA-NeXT's per-tile projected embeddings into one flat sequence, spliced into the text decoder exactly like every other <see cref="IVlmImageEncoder"/>'s output.</summary>
/// <remarks>Direct port of HF's <c>LlavaNextModel.pack_image_features</c>/<c>get_anyres_image_grid_shape</c>/<c>unpad_image</c> (<c>modeling_llava_next.py</c>), read verbatim from the installed <c>transformers</c> source and cross-checked against <c>tests/python-reference/dump_llavanext_vision_ref.py</c> (which calls the REAL bound method as the oracle) rather than reconstructed from memory — this repo's own research flagged llama.cpp's own LLaVA-NeXT merge as buggy (base tile placed last, no unpad, an unused <c>image_newline</c> tensor; ggml-org/llama.cpp#8457), so llama.cpp cannot be the reference here.
/// <para>Order of operations (tile 0 = base/overview, tiles 1.. = the best-fit grid's patches in raster order): reshape the grid tiles into one <c>[hidden, gridH·patchGrid, gridW·patchGrid]</c> patch plane → crop (unpad) the dimension whose aspect ratio the padding distorted, back toward the original image's aspect ratio → append the learned <c>model.image_newline</c> vector as one more "column" at the end of every row → flatten row-major → prepend the base tile's embeddings.</para></remarks>
public static unsafe class LlavaNextFeatureMerger
{
    /// <summary><paramref name="tileEmbeds"/>[0] is the base/overview tile, [1..] are the best-fit grid's tiles in raster order; <paramref name="origH"/>/<paramref name="origW"/> are the ORIGINAL (pre-tiling) image dimensions, required for the unpad crop. Returns <c>[1, totalTokens, hidden]</c>, base tokens first.</summary>
    public static Tensor PackImageFeatures(Tensor[] tileEmbeds, int origH, int origW,
        (int h, int w)[] gridPinpoints, int tileSize, int patchGrid, int hidden, Tensor imageNewline)
    {
        int baseTokens = (int)tileEmbeds[0].Shape[1];
        float* newlinePtr = (float*)imageNewline.DataPointer;

        if (tileEmbeds.Length == 1)
        {
            // Degenerate anyres config where the best-fit resolution equals the tile size itself (no extra grid
            // patches) — HF's `image_feature.shape[0] > 1` false branch. Base tokens + one trailing newline row.
            Tensor single = new(new TensorShape(1, baseTokens + 1, hidden), DType.F32);
            float* sd = (float*)single.DataPointer;
            Buffer.MemoryCopy((void*)tileEmbeds[0].DataPointer, sd, (long)baseTokens * hidden * 4, (long)baseTokens * hidden * 4);
            Buffer.MemoryCopy(newlinePtr, sd + (long)baseTokens * hidden, (long)hidden * 4, (long)hidden * 4);
            return single;
        }

        (int bestH, int bestW) = LlavaNextImagePreprocessor.SelectBestResolution((origH, origW), gridPinpoints);
        int gh = bestH / tileSize, gw = bestW / tileSize;
        int fullH = gh * patchGrid, fullW = gw * patchGrid;

        // Reshape+permute the (gh*gw) grid tiles into one [hidden, fullH, fullW] patch plane:
        //   merged[d, ph*patchGrid+ih, pw*patchGrid+iw] = tileEmbeds[1 + ph*gw+pw][ih*patchGrid+iw][d]
        // (HF: image_feature.view(gh,gw,patchGrid,patchGrid,-1).permute(4,0,2,1,3).flatten(1,2).flatten(2,3))
        float[] merged = new float[(long)hidden * fullH * fullW];
        for (int ph = 0; ph < gh; ph++)
            for (int pw = 0; pw < gw; pw++)
            {
                float* tile = (float*)tileEmbeds[1 + ph * gw + pw].DataPointer;
                for (int ih = 0; ih < patchGrid; ih++)
                    for (int iw = 0; iw < patchGrid; iw++)
                    {
                        int row = ph * patchGrid + ih, col = pw * patchGrid + iw;
                        float* srcVec = tile + (long)(ih * patchGrid + iw) * hidden;
                        for (int d = 0; d < hidden; d++)
                            merged[(long)d * fullH * fullW + (long)row * fullW + col] = srcVec[d];
                    }
            }

        // unpad_image: crop whichever dimension the resize+pad distorted, back toward the original aspect ratio.
        double originalAspect = (double)origW / origH, currentAspect = (double)fullW / fullH;
        int cropRowStart, cropRowEnd, cropColStart, cropColEnd;
        if (originalAspect > currentAspect)
        {
            double scale = (double)fullW / origW;
            int newHeight = (int)Math.Round(origH * scale, 7, MidpointRounding.AwayFromZero);
            int padding = (fullH - newHeight) / 2;
            (cropRowStart, cropRowEnd, cropColStart, cropColEnd) = (padding, fullH - padding, 0, fullW);
        }
        else
        {
            double scale = (double)fullH / origH;
            int newWidth = (int)Math.Round(origW * scale, 7, MidpointRounding.AwayFromZero);
            int padding = (fullW - newWidth) / 2;
            (cropRowStart, cropRowEnd, cropColStart, cropColEnd) = (0, fullH, padding, fullW - padding);
        }
        int croppedH = cropRowEnd - cropRowStart, croppedW = cropColEnd - cropColStart;

        // Append image_newline as one more column at the end of every row, flatten row-major, prepend base.
        int patchTokens = croppedH * (croppedW + 1);
        Tensor result = new(new TensorShape(1, baseTokens + patchTokens, hidden), DType.F32);
        float* rd = (float*)result.DataPointer;
        Buffer.MemoryCopy((void*)tileEmbeds[0].DataPointer, rd, (long)baseTokens * hidden * 4, (long)baseTokens * hidden * 4);

        long outBase = (long)baseTokens * hidden;
        for (int row = 0; row < croppedH; row++)
        {
            int srcRow = cropRowStart + row;
            long rowOff = outBase + (long)row * (croppedW + 1) * hidden;
            for (int col = 0; col < croppedW; col++)
            {
                int srcCol = cropColStart + col;
                float* dstVec = rd + rowOff + (long)col * hidden;
                for (int d = 0; d < hidden; d++)
                    dstVec[d] = merged[(long)d * fullH * fullW + (long)srcRow * fullW + srcCol];
            }
            Buffer.MemoryCopy(newlinePtr, rd + rowOff + (long)croppedW * hidden, (long)hidden * 4, (long)hidden * 4);
        }
        return result;
    }
}
