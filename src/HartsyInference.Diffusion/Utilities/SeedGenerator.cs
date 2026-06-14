using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Generates random noise tensors for diffusion sampling. Uses deterministic seeding for reproducibility.</summary>
public static class SeedGenerator
{
    /// <summary>Creates a tensor filled with Gaussian noise (mean=0, std=1) using the given seed.</summary>
    public static unsafe Tensor CreateNoise(TensorShape shape, int seed)
    {
        Tensor noise = new Tensor(shape, DType.F32);
        float* ptr = (float*)noise.DataPointer;
        int count = (int)shape.ElementCount;

        Random rng = new Random(seed);

        // Box-Muller transform for Gaussian noise
        for (int i = 0; i < count - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble(); // Avoid log(0)
            double u2 = rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            ptr[i] = (float)(radius * Math.Cos(theta));
            ptr[i + 1] = (float)(radius * Math.Sin(theta));
        }

        // Handle odd count
        if (count % 2 != 0)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            ptr[count - 1] = (float)(radius * Math.Cos(2.0 * Math.PI * u2));
        }

        return noise;
    }

    /// <summary>Generates a random seed.</summary>
    public static int RandomSeed() => Random.Shared.Next();
}
