using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.Cuda.Tests;

/// <summary>Tensor helpers and PTX lookup shared by the CUDA MoE primitive tests.</summary>
internal static unsafe class MoePrimitiveTestData
{
    public static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    public static Tensor F32(float[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < data.Length; i++) p[i] = data[i];
        return t;
    }

    public static Tensor I32(int[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        int* p = (int*)t.DataPointer;
        for (int i = 0; i < data.Length; i++) p[i] = data[i];
        return t;
    }

    public static Tensor EmptyF32(params long[] shape) => new(new TensorShape(shape), DType.F32);

    public static Tensor EmptyI32(params long[] shape) => new(new TensorShape(shape), DType.I32);

    public static float[] ReadF32(Tensor t)
    {
        float[] r = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    public static int[] ReadI32(Tensor t)
    {
        int[] r = new int[t.ElementCount];
        int* p = (int*)t.DataPointer;
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    public static float[] Random(int count, int seed, float scale = 1f)
    {
        Random rng = new(seed);
        float[] r = new float[count];
        for (int i = 0; i < r.Length; i++) r[i] = ((float)rng.NextDouble() * 2f - 1f) * scale;
        return r;
    }
}
