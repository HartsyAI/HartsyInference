using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Cheap content signatures for reference-image tensors, used to key single-slot edit-reference latent caches (Qwen-Image-Edit, OmniGen2 edit).</summary>
public static unsafe class ImageSignature
{
    /// <summary>FNV-1a over 1024 strided F32 samples + the trailing two shape dims. Collision-safe enough for a single-slot cache whose only job is "same image as last generation?".</summary>
    public static long Compute(Tensor image)
    {
        float* p = (float*)image.DataPointer;
        long n = image.ElementCount;
        long stride = Math.Max(1, n / 1024);
        ulong h = 14695981039346656037UL;
        for (long i = 0; i < n; i += stride)
        {
            h ^= (uint)BitConverter.SingleToInt32Bits(p[i]);
            h *= 1099511628211UL;
        }
        h ^= (ulong)image.Shape[image.Shape.Rank - 2] << 32 ^ (ulong)image.Shape[image.Shape.Rank - 1];
        return (long)h;
    }

    /// <summary>Combined signature of an ordered reference list: <c>sig = sig·1000003 ^ Compute(image)</c> per image.</summary>
    public static long CombineList(IReadOnlyList<Tensor> images)
    {
        long sig = 0L;
        for (int i = 0; i < images.Count; i++)
            sig = sig * 1000003L ^ Compute(images[i]);
        return sig;
    }
}
