using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Time, caption, plucker and RoPE inputs of the SANA-WM DiT, shared by the DiT and the real-weight block tests.
/// Every weight is F32 and borrowed; every returned tensor is owned by the caller's scratch list.</summary>
internal static class SanaWmConditioning
{
    private const float TimestepMaxPeriod = 10000f;
    private const float CaptionNormEpsilon = 1e-5f;
    private const double RopeTheta = 10000.0;
    private const int RopeMaxSequence = 1024;
    internal const int ModulationSlots = 6;

    /// <summary>Time embedding and block modulation: <c>t = Linear(SiLU(Linear(sin/cos features)))</c>, <c>t0 = Linear(SiLU(t))</c>.</summary>
    internal static (Tensor Time, Tensor BlockTime) Time(IBackend backend, SanaWmTimeWeights weights, int timestepWidth,
        ReadOnlySpan<float> timesteps, List<Tensor> scratch)
    {
        int channels = (int)weights.FirstBias.Shape[0];
        int rows = timesteps.Length;
        Tensor features = new(new TensorShape([rows, timestepWidth]), DType.F32, DeviceKind.Cpu);
        Span<float> destination = features.AsSpan<float>();
        int half = timestepWidth / 2;
        for (int row = 0; row < rows; row++)
        {
            float timestep = (long)timesteps[row];
            for (int i = 0; i < half; i++)
            {
                float frequency = MathF.Exp(-MathF.Log(TimestepMaxPeriod) * i / half);
                float angle = timestep * frequency;
                destination[row * timestepWidth + i] = MathF.Cos(angle);
                destination[row * timestepWidth + half + i] = MathF.Sin(angle);
            }
        }

        Tensor input = Upload(backend, scratch, features);
        Tensor hidden = NewTensor(backend, scratch, new TensorShape([rows, channels]));
        backend.Linear(hidden, input, weights.FirstWeight, weights.FirstBias);
        Tensor activated = NewTensor(backend, scratch, hidden.Shape);
        backend.Silu(activated, hidden);
        Tensor time = NewTensor(backend, scratch, hidden.Shape);
        backend.Linear(time, activated, weights.SecondWeight, weights.SecondBias);

        Tensor timeActivated = NewTensor(backend, scratch, time.Shape);
        backend.Silu(timeActivated, time);
        Tensor blockTime = NewTensor(backend, scratch, new TensorShape([rows, ModulationSlots * channels]));
        backend.Linear(blockTime, timeActivated, weights.BlockWeight, weights.BlockBias);
        return (time, blockTime);
    }

    /// <summary>Caption projection: fc1, GELU(tanh), fc2, then <c>attention_y_norm</c>. Returns <c>[B, L, C]</c>.</summary>
    internal static Tensor Caption(IBackend backend, SanaWmCaptionWeights weights, Tensor text, long batch, long textTokens,
        List<Tensor> scratch)
    {
        int channels = (int)weights.FirstBias.Shape[0];
        TensorShape textRows = new([batch * textTokens, text.Shape[3]]);
        TensorShape channelRows = new([batch * textTokens, channels]);
        Tensor hidden = NewTensor(backend, scratch, channelRows);
        backend.Linear(hidden, text.Reshape(textRows), weights.FirstWeight, weights.FirstBias);
        Tensor activated = NewTensor(backend, scratch, channelRows);
        backend.Gelu(activated, hidden);
        Tensor projected = NewTensor(backend, scratch, channelRows);
        backend.Linear(projected, activated, weights.SecondWeight, weights.SecondBias);
        Tensor normed = NewTensor(backend, scratch, channelRows);
        backend.RmsNorm(normed, projected, weights.Norm, CaptionNormEpsilon);
        return normed.Reshape(new TensorShape([batch, textTokens, channels]));
    }

    /// <summary>Additive cross-attention bias <c>[B, 1, 1, L]</c>: 0 for valid caption tokens, -10000 for padding.</summary>
    internal static Tensor MaskBias(IBackend backend, Tensor mask, long batch, long textTokens, List<Tensor> scratch)
    {
        using Tensor maskHost = SanaWmTensors.Readback(backend, mask);
        Tensor biasHost = new(new TensorShape([batch, 1, 1, textTokens]), DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> source = maskHost.AsReadOnlySpan<float>();
        Span<float> destination = biasHost.AsSpan<float>();
        for (int i = 0; i < destination.Length; i++)
            destination[i] = (1f - source[i]) * -10000f;
        return Upload(backend, scratch, biasHost);
    }

    /// <summary>Plucker embedding of <c>[B, P, T, H, W]</c> features as per-token rows <c>[B, N, C]</c>; <paramref name="weight"/>
    /// is the 1×1×1 kernel <c>[C, P]</c>.</summary>
    internal static Tensor Plucker(IBackend backend, Tensor weight, Tensor? bias, Tensor chunkPlucker, List<Tensor> scratch)
    {
        long batch = chunkPlucker.Shape[0];
        int tokens = checked((int)(chunkPlucker.Shape[2] * chunkPlucker.Shape[3] * chunkPlucker.Shape[4]));
        int channels = (int)weight.Shape[0];
        Tensor tokenInput = ChannelFirstToTokens(backend, scratch, chunkPlucker, (int)chunkPlucker.Shape[1], tokens);
        Tensor embedded = NewTensor(backend, scratch, new TensorShape([batch * tokens, channels]));
        backend.Linear(embedded, tokenInput, weight, bias);
        return embedded.Reshape(new TensorShape([batch, tokens, channels]));
    }

    /// <summary>Main 3D RoPE for a latent grid: interleaved-pair cos and sin <c>[N, headDim]</c> from <see cref="WanRope"/>,
    /// and the per-complex camera tables <c>[N, headDim/2]</c> taken from its even columns.</summary>
    internal static (Tensor Cos, Tensor Sin, Tensor CameraCos, Tensor CameraSin) Rope(IBackend backend, int headDim, int frames,
        int height, int width, List<Tensor> scratch)
    {
        WanRope rope = new(headDim, RopeTheta, RopeMaxSequence);
        (Tensor cosHost, Tensor sinHost) = rope.BuildCosSin(frames, height, width);
        Tensor cos = Upload(backend, scratch, cosHost);
        Tensor sin = Upload(backend, scratch, sinHost);
        int tokens = frames * height * width;
        int half = headDim / 2;
        Tensor cameraCos = NewTensor(backend, scratch, new TensorShape([tokens, half]));
        Tensor cameraSin = NewTensor(backend, scratch, new TensorShape([tokens, half]));
        ReadOnlySpan<long> dims = [tokens, half];
        ReadOnlySpan<long> strides = [headDim, 2];
        backend.CopyStrided(cameraCos, cos, StridedCopySpec.Gather(new StridedView(dims, strides, 0)));
        backend.CopyStrided(cameraSin, sin, StridedCopySpec.Gather(new StridedView(dims, strides, 0)));
        return (cos, sin, cameraCos, cameraSin);
    }

    /// <summary>Channel-first <c>[B, C, T, H, W]</c> to token rows <c>[B·N, C]</c>, N = T·H·W in frame-major order.</summary>
    internal static Tensor ChannelFirstToTokens(IBackend backend, List<Tensor> scratch, Tensor source, int channels, int tokens)
    {
        long batch = source.Shape[0];
        Tensor rows = NewTensor(backend, scratch, new TensorShape([batch * tokens, channels]));
        ReadOnlySpan<long> dims = [batch, tokens, channels];
        ReadOnlySpan<long> strides = [(long)channels * tokens, 1, tokens];
        backend.CopyStrided(rows, source, StridedCopySpec.Gather(new StridedView(dims, strides, 0)));
        return rows;
    }

    /// <summary>Token rows <c>[B·N, C]</c> to channel-first <c>[B, C, T, H, W]</c>. The caller owns the result.</summary>
    internal static Tensor TokensToChannelFirst(IBackend backend, Tensor tokens, long batch, int channels, int frames, int height,
        int width)
    {
        int count = checked(frames * height * width);
        Tensor output = new(new TensorShape([batch, channels, frames, height, width]), DType.F32, backend.Device);
        bool built = false;
        try
        {
            ReadOnlySpan<long> dims = [batch, channels, count];
            ReadOnlySpan<long> strides = [(long)count * channels, 1, channels];
            backend.CopyStrided(output, tokens, StridedCopySpec.Gather(new StridedView(dims, strides, 0)));
            built = true;
            return output;
        }
        finally
        {
            if (!built)
                output.Dispose();
        }
    }

    private static Tensor NewTensor(IBackend backend, List<Tensor> scratch, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32, backend.Device);
        scratch.Add(tensor);
        return tensor;
    }

    // Takes ownership of host: it is either returned as the result or disposed after the copy.
    private static Tensor Upload(IBackend backend, List<Tensor> scratch, Tensor host)
    {
        if (host.Device == backend.Device)
        {
            scratch.Add(host);
            return host;
        }

        Tensor device = SanaWmTensors.Upload(backend, host);
        host.Dispose();
        scratch.Add(device);
        return device;
    }
}
