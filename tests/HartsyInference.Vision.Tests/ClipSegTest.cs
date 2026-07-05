using System.IO;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation;
using Xunit;

namespace HartsyInference.Vision.Tests;

public class ClipSegTest
{
    private const string ModelDir = "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/clipseg/clipseg-rd64-refined-fp16-safetensors";
    private const string Scratch = "/tmp/claude-1000/-home-hartsy-Desktop-HartsyInference/a035d95f-4f31-468f-aa2e-4b2c832692fe/scratchpad";

    [Fact]
    public void ConvTranspose2d_Isolation()
    {
        HartsyInference.Core.Backends.IBackend backend = new CpuBackend();
        // input [1,2,3,3] = 0..17 ; weight [2,3,3,3] = 0..53 ; k3 s1 p1, no bias -> output [1,3,3,3]
        Tensor inp = new Tensor(new TensorShape(1, 2, 3, 3), DType.F32);
        System.Span<float> ins = inp.AsSpan<float>();
        for (int i = 0; i < ins.Length; i++) ins[i] = i;
        Tensor w = new Tensor(new TensorShape(2, 3, 3, 3), DType.F32);
        System.Span<float> ws = w.AsSpan<float>();
        for (int i = 0; i < ws.Length; i++) ws[i] = i;
        Tensor outp = new Tensor(new TensorShape(1, 3, 3, 3), DType.F32);
        backend.ConvTranspose2d(outp, inp, w, null, 1, 1, 1, 1);
        System.Span<float> os = outp.AsSpan<float>();
        Assert.Equal(27, os.Length);
        bool anyNonZero = false;
        foreach (float v in os) { Assert.True(float.IsFinite(v), "ConvTranspose2d produced non-finite output"); anyNonZero |= v != 0f; }
        Assert.True(anyNonZero, "ConvTranspose2d output is all-zero for non-zero input/weight");
    }

    // Integration: needs a real ClipSeg checkpoint + a preprocessed input bin (skips cleanly when absent).
    // The ModelDir/Scratch consts below are the original author's local diagnostic paths.
    [Trait("Category", "Integration")]
    [Fact]
    public unsafe void ClipSeg_Segment_ProducesMasks()
    {
        string inBin = Path.Combine(Scratch, "clipseg_in.bin");
        if (!File.Exists(Path.Combine(ModelDir, "model.safetensors")) || !File.Exists(inBin))
        {
            return; // skip when assets absent
        }

        using CpuBackend backend = new CpuBackend();
        using ClipSegPipeline seg = new ClipSegPipeline(ModelDir);

        byte[] raw = File.ReadAllBytes(inBin);
        Tensor img = new Tensor(new TensorShape(1, 3, 224, 224), DType.F32);
        System.Span<float> dst = img.AsSpan<float>();
        fixed (byte* rp = raw)
        {
            new System.ReadOnlySpan<float>(rp, raw.Length / sizeof(float)).CopyTo(dst);
        }

        foreach (string prompt in new[] { "house" })
        {
            float[] mask = seg.Segment(backend, img, prompt, out int w, out int h);
            Assert.Equal(224, w);
            Assert.Equal(224 * 224, mask.Length);
            // Sanity: mask is a valid probability map with some spatial variation (not constant / not NaN).
            float min = float.MaxValue, max = float.MinValue, sum = 0;
            foreach (float v in mask)
            {
                Assert.False(float.IsNaN(v) || float.IsInfinity(v), "mask has NaN/Inf");
                min = System.MathF.Min(min, v); max = System.MathF.Max(max, v); sum += v;
            }
            byte[] outBytes = new byte[mask.Length * sizeof(float)];
            fixed (float* mp = mask)
            fixed (byte* op = outBytes)
            {
                Buffer.MemoryCopy(mp, op, outBytes.Length, outBytes.Length);
            }
            File.WriteAllBytes(Path.Combine(Scratch, $"clipseg_mask_{prompt}.bin"), outBytes);
            File.AppendAllText(Path.Combine(Scratch, "clipseg_stats.txt"),
                $"{prompt}: min={min:F3} max={max:F3} mean={sum / mask.Length:F3}\n");
        }
        img.Dispose();
    }
}
