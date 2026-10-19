using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Z-Image position ids. These encode which token sits at which coordinate, so they stay model-owned; the RoPE
/// tables built from them live in <see cref="ZImageRope"/> and <c>RopeTableCache</c>.</summary>
public static unsafe class ZImagePosEmbed
{
    /// <summary>Builds position IDs for Z-Image's main DiT in [image, caption] concat order. Mirrors diffusers' `patchify_and_embed`: caption uses pos_grid_size=(capPaddedLen,1,1) and pos_start=(1,0,0) — every caption slot (real + pad) gets frame index 1..capPaddedLen; image uses pos_start=(capPaddedLen+1, 0, 0) with grid (1, hPacked, wPacked) so image tokens share frame=capPaddedLen+1 and vary in (h,w). Image pad slots stay at (0,0,0). Returns shape [imgPaddedLen + capPaddedLen, 3] F32.</summary>
    /// <param name="capPaddedLen">Padded caption sequence length (real_cap + (-real_cap)%32). Used for BOTH the caption frame range and the image frame offset, matching diffusers' single _pad_with_ids call which uses the padded grid size.</param>
    /// <param name="hPacked">Image height in patch units (latent_h / patch_size).</param>
    /// <param name="wPacked">Image width in patch units (latent_w / patch_size).</param>
    /// <param name="imgPaddedLen">Padded image sequence length (real h*w padded up to multiple of 32 with x_pad_token).</param>
    public static Tensor BuildPositionIds(int capPaddedLen, int hPacked, int wPacked, int imgPaddedLen)
    {
        int imgRealLen = hPacked * wPacked;
        if (imgRealLen > imgPaddedLen)
            throw new ArgumentException($"imgRealLen={imgRealLen} cannot exceed imgPaddedLen={imgPaddedLen}", nameof(imgPaddedLen));

        int totalLen = imgPaddedLen + capPaddedLen;
        TensorShape shape = new TensorShape(totalLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);

        float* ptr = (float*)posIds.DataPointer;
        for (int i = 0; i < totalLen * 3; i++)
            ptr[i] = 0f;

        // [0 .. imgPaddedLen) — image tokens. Real tokens get (capPaddedLen+1, row, col); pad slots stay (0,0,0).
        float imgFrame = capPaddedLen + 1;
        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = row * wPacked + col;
                ptr[idx * 3 + 0] = imgFrame;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }

        // [imgPaddedLen .. imgPaddedLen+capPaddedLen) — caption tokens including pads. Frame runs 1..capPaddedLen, h=w=0.
        for (int t = 0; t < capPaddedLen; t++)
        {
            int idx = imgPaddedLen + t;
            ptr[idx * 3 + 0] = t + 1;
        }

        return posIds;
    }

    /// <summary>Builds caption-only position IDs for the context_refiner stack. Diffusers patchify_and_embed creates these with pos_grid_size=(capPaddedLen, 1, 1) and pos_start=(1,0,0), so EVERY caption slot (real + pad) gets a unique frame index 1..capPaddedLen with h=w=0. Returns shape [capPaddedLen, 3] F32.</summary>
    public static Tensor BuildCaptionPositionIds(int capPaddedLen)
    {
        TensorShape shape = new TensorShape(capPaddedLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* ptr = (float*)posIds.DataPointer;
        for (int t = 0; t < capPaddedLen; t++)
        {
            ptr[t * 3 + 0] = t + 1;
            ptr[t * 3 + 1] = 0f;
            ptr[t * 3 + 2] = 0f;
        }
        return posIds;
    }

    /// <summary>Builds image-only position IDs for the noise_refiner stack (operates on padded image tokens with no caption present). Real tokens: (1, row, col); pad: (0,0,0). The frame=1 base mirrors diffusers' use of pos_start=(1,0,0) for image-only refinement.</summary>
    public static Tensor BuildImagePositionIds(int hPacked, int wPacked, int imgPaddedLen)
    {
        int imgRealLen = hPacked * wPacked;
        TensorShape shape = new TensorShape(imgPaddedLen, 3);
        Tensor posIds = new Tensor(shape, DType.F32);

        float* ptr = (float*)posIds.DataPointer;
        for (int i = 0; i < imgPaddedLen * 3; i++)
            ptr[i] = 0f;

        for (int row = 0; row < hPacked; row++)
        {
            for (int col = 0; col < wPacked; col++)
            {
                int idx = row * wPacked + col;
                ptr[idx * 3 + 0] = 1f;
                ptr[idx * 3 + 1] = row;
                ptr[idx * 3 + 2] = col;
            }
        }
        return posIds;
    }
}
