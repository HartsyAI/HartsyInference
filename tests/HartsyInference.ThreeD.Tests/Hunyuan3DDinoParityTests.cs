using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Dinov2;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Real-weight parity for the Hunyuan3D-2 conditioner (DINOv2-giant, SwiGLU FFN) vs the upstream HF
/// <c>Dinov2Model</c>. Gated on <c>HUNYUAN3D_DINO_WEIGHTS</c> (<c>dino_giant.safetensors</c>, the conditioner
/// weights extracted from the DiT checkpoint) + <c>HUNYUAN3D_DINO_REF</c> (<c>dino_giant_ref_io.safetensors</c>
/// from <c>dump_dino_giant_reference.py</c>). Skips cleanly when unset.</summary>
public sealed unsafe class Hunyuan3DDinoParityTests
{
    private readonly ITestOutputHelper _out;
    public Hunyuan3DDinoParityTests(ITestOutputHelper o) => _out = o;

    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    [Fact]
    public void DinoGiant_ImageTokens_MatchReference()
    {
        string? w = Environment.GetEnvironmentVariable("HUNYUAN3D_DINO_WEIGHTS");
        string? r = Environment.GetEnvironmentVariable("HUNYUAN3D_DINO_REF");
        if (w is null || r is null || !File.Exists(w) || !File.Exists(r)) return; // gated

        using IBackend cpu = new CpuBackend();
        using SafeTensorsLoader wl = new(); wl.Load(w);
        using SafeTensorsLoader rl = new(); rl.Load(r);

        Dinov2VisionEncoder dino = new(Dinov2Preset.Giant);
        dino.LoadWeights(wl.GetAllTensors());

        Tensor pixels = rl.GetTensor("pixels");          // [1,3,518,518] in [0,1]
        Tensor norm = ImagenetNormalize(pixels);
        Tensor tokens = dino.Encode(cpu, norm); norm.Dispose();
        Tensor refTok = rl.GetTensor("image_tokens");    // [1,1370,1536]

        long n = tokens.ElementCount;
        float* pa = (float*)tokens.DataPointer; float* pb = (float*)refTok.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
        }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"DINOv2-giant image_tokens: maxAbs={maxAbs:E3} corr={corr:F8}");
        tokens.Dispose();
        Assert.True(maxAbs < 1e-2, $"maxAbs {maxAbs}");
        Assert.True(corr > 0.9999, $"corr {corr}");
    }

    private static Tensor ImagenetNormalize(Tensor pixels)
    {
        int b = (int)pixels.Shape[0], c = (int)pixels.Shape[1], h = (int)pixels.Shape[2], w = (int)pixels.Shape[3];
        Tensor o = new(pixels.Shape, DType.F32);
        float* src = (float*)pixels.DataPointer; float* dst = (float*)o.DataPointer;
        long hw = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                long baseI = ((long)bi * c + ci) * hw;
                for (long i = 0; i < hw; i++) dst[baseI + i] = (src[baseI + i] - Mean[ci]) / Std[ci];
            }
        return o;
    }
}
