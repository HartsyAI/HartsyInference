using HartsyInference.Audio.Dsp;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Port of the official <c>FlowMatching</c> euler sampler (<c>inference_mode='euler'</c>): integrates noise to
/// the latent over <c>numSteps</c> uniform steps of the guided velocity from <see cref="ControlFoleyNetwork.OdeWrapper"/>.</summary>
public static class ControlFoleySampler
{
    /// <summary>Standard-normal noise <c>[batch, seqLen, dim]</c> from the engine's <see cref="DeterministicRng"/>.</summary>
    public static unsafe Tensor CreateNoise(int batch, int seqLen, int dim, int seed)
    {
        Tensor noise = ControlFoleyOps.New(batch, seqLen, dim);
        uint state = DeterministicRng.Seed(seed);
        float* p = (float*)noise.DataPointer;
        for (long i = 0; i < noise.ElementCount; i++)
        {
            p[i] = DeterministicRng.NextGaussian(ref state);
        }

        return noise;
    }

    /// <summary><c>torch.linspace(0, 1 - minSigma, numSteps + 1)</c> in float32 (symmetric from both ends).</summary>
    public static float[] TimeGrid(int numSteps, float minSigma = 0f)
    {
        if (numSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numSteps));
        }

        int count = numSteps + 1;
        float end = 1f - minSigma, step = end / (count - 1);
        float[] grid = new float[count];
        for (int i = 0; i < count; i++)
        {
            grid[i] = i < count / 2 ? step * i : end - step * (count - i - 1);
        }

        return grid;
    }

    /// <summary>Runs the euler integration from <paramref name="noise"/> (left untouched) and returns the latent
    /// <c>[B, N, LatentDim]</c>, un-normalized when <paramref name="unnormalize"/> is set (what <c>generate()</c> hands
    /// to the VAE).</summary>
    /// <param name="onState">Optional tap receiving each step's input state and index.</param>
    public static unsafe Tensor Sample(IBackend backend, ControlFoleyNetwork net, ControlFoleyConditions cond,
        ControlFoleyConditions empty, Tensor noise, int numSteps, float cfgStrength, bool unnormalize = true,
        Action<int, Tensor>? onState = null, float minSigma = 0f)
    {
        float[] grid = TimeGrid(numSteps, minSigma);
        Tensor x = ControlFoleyOps.Clone(noise);
        try
        {
            for (int i = 0; i < numSteps; i++)
            {
                onState?.Invoke(i, x);
                using Tensor flow = net.OdeWrapper(backend, grid[i], x, cond, empty, cfgStrength);
                float dt = grid[i + 1] - grid[i];
                float* xp = (float*)x.DataPointer, fp = (float*)flow.DataPointer;
                for (long k = 0; k < x.ElementCount; k++)
                {
                    xp[k] += dt * fp[k];
                }
            }

            if (unnormalize)
            {
                net.Unnormalize(x);
            }

            return x;
        }
        catch
        {
            x.Dispose();
            throw;
        }
    }
}
