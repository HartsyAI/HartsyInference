using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>Device-side frame plumbing shared by the LongCat denoise loops. Latents are laid out <c>[channels, frames, hw]</c>; a device state keeps
/// the conditioning frames and the stepped frames apart, each as <c>[channels·frames, hw]</c>, so a step never reads the latents back. Every op here
/// is a device op: a copy is a scale by one, and rows move with SliceRows and Concat.</summary>
internal static class LongCatFrameOps
{
    /// <summary>A device tensor holding <paramref name="data"/>, of <paramref name="shape"/>.</summary>
    public static Tensor FromHost(float[] data, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        data.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    /// <summary>The model input of an uncached step: the conditioning and the stepped frames as one clip, reshaped. The clip is a temporary and is
    /// freed here; before this, the Reshape callers left it undisposed, one [channels·frames, hw] tensor per step.</summary>
    public static Tensor ModelInput(IBackend backend, Tensor stepped, Tensor? condState, int channels, int cond, int steppedFrames, int hw, TensorShape shape)
    {
        using Tensor clip = Clip(backend, stepped, condState, channels, cond, steppedFrames, hw);
        return Reshape(backend, clip, shape);
    }

    /// <summary>The same elements under another shape: a scale by one, so the copy stays on the device.</summary>
    public static Tensor Reshape(IBackend backend, Tensor source, TensorShape shape)
    {
        Tensor result = new(shape, DType.F32);
        try
        {
            backend.Scale(result, source, 1f);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    /// <summary><paramref name="count"/> rows of a 2-D tensor, from row <paramref name="offset"/>, as a new tensor.</summary>
    public static Tensor RowsOf(IBackend backend, Tensor source, int offset, int count, int width)
    {
        Tensor rows = new(new TensorShape(count, width), DType.F32);
        try
        {
            backend.SliceRows(rows, source, offset);
            return rows;
        }
        catch
        {
            rows.Dispose();
            throw;
        }
    }

    /// <summary>The rows <c>first…first+count</c> of every channel of a <c>[channels·frames, width]</c> clip, joined in channel order.</summary>
    public static Tensor Rows(IBackend backend, Tensor clip, int channels, int frames, int first, int count, int width)
    {
        List<Tensor> pieces = new(channels);
        try
        {
            for (int c = 0; c < channels; c++)
                pieces.Add(RowsOf(backend, clip, c * frames + first, count, width));
            Tensor joined = new(new TensorShape(channels * count, width), DType.F32);
            try
            {
                backend.Concat(joined, pieces.ToArray(), 0);
            }
            catch
            {
                joined.Dispose();
                throw;
            }
            return joined;
        }
        finally
        {
            foreach (Tensor piece in pieces)
                piece.Dispose();
        }
    }

    /// <summary>The whole clip <c>[channels·(cond + stepped), width]</c>: each channel's conditioning rows (when <paramref name="condState"/> is not
    /// null) then its stepped rows. The caller owns the result.</summary>
    public static Tensor Clip(IBackend backend, Tensor stepped, Tensor? condState, int channels, int cond, int steppedFrames, int width)
    {
        List<Tensor> pieces = new(2 * channels);
        try
        {
            for (int c = 0; c < channels; c++)
            {
                if (cond > 0)
                    pieces.Add(RowsOf(backend, condState!, c * cond, cond, width));
                pieces.Add(RowsOf(backend, stepped, c * steppedFrames, steppedFrames, width));
            }
            Tensor clip = new(new TensorShape(channels * (cond + steppedFrames), width), DType.F32);
            try
            {
                backend.Concat(clip, pieces.ToArray(), 0);
            }
            catch
            {
                clip.Dispose();
                throw;
            }
            return clip;
        }
        finally
        {
            foreach (Tensor piece in pieces)
                piece.Dispose();
        }
    }

    /// <summary>The stepped frames of a host latent <c>[channels, frames, hw]</c>, as <c>[channels·stepped, hw]</c>.</summary>
    public static float[] SteppedFrames(float[] latents, int channels, int frames, int hw, int cond)
    {
        int stepped = frames - cond;
        float[] result = new float[channels * stepped * hw];
        for (int c = 0; c < channels; c++)
            Array.Copy(latents, (c * frames + cond) * hw, result, c * stepped * hw, stepped * hw);
        return result;
    }

    /// <summary>The host latent <c>[channels, frames, hw]</c> from the conditioning frames (when given) and the stepped state.</summary>
    public static float[] JoinFrames(float[]? condLatents, float[] stepped, int channels, int frames, int hw, int cond)
    {
        int steppedFrames = frames - cond;
        float[] result = new float[channels * frames * hw];
        for (int c = 0; c < channels; c++)
        {
            if (cond > 0)
                Array.Copy(condLatents!, c * cond * hw, result, c * frames * hw, cond * hw);
            Array.Copy(stepped, c * steppedFrames * hw, result, (c * frames + cond) * hw, steppedFrames * hw);
        }
        return result;
    }
}
