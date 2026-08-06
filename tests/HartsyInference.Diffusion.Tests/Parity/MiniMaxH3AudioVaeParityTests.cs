using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight numeric parity for the MiniMax-H3 audio VAE decoder against the reference module shipped
/// inside the checkpoint (<c>audio_vae/minimax_h3_audio_vae.py</c>). The reference dumps a normalized latent and its
/// decoded stereo waveform; this replays the same latent through the C# decoder. Skips when either is absent.
/// <para>The dumped latent must be an <i>encoded</i> signal, not sampled noise: a random latent is far out of
/// distribution and decodes ~30 dB louder than anything the DiT emits, so error measured on it does not bound the
/// error at the real operating point.</para>
/// <code>
/// python avae/ref.py &lt;dump-dir&gt;
/// MINIMAX_H3_AUDIO_VAE=/path/audio_vae MINIMAX_H3_AUDIO_VAE_DUMP=&lt;dump-dir&gt; \
///   dotnet test --filter MiniMaxH3AudioVaeParity
/// </code></summary>
public unsafe class MiniMaxH3AudioVaeParityTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxH3AudioVaeParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void DecodeMatchesTheShippedReference()
    {
        string? vaeDir = Environment.GetEnvironmentVariable("MINIMAX_H3_AUDIO_VAE");
        string? dumpDir = Environment.GetEnvironmentVariable("MINIMAX_H3_AUDIO_VAE_DUMP");
        if (vaeDir is null || dumpDir is null
            || !File.Exists(Path.Combine(vaeDir, "model.safetensors"))
            || !File.Exists(Path.Combine(dumpDir, "latent.bin")))
        {
            return;
        }

        Tensor latent = ReadRaw(Path.Combine(dumpDir, "latent.bin"), 1, 32, 2, -1);
        int frames = (int)latent.Shape[3];
        Tensor expected = ReadRaw(Path.Combine(dumpDir, "wave_ref.bin"), 1, 2, frames * 800);

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(Path.Combine(vaeDir, "model.safetensors"));
        MiniMaxH3AudioVaeDecoder decoder = new();
        decoder.LoadWeights(new Dictionary<string, Tensor>(loader.GetAllTensors()));

        using IBackend backend = MakeBackend();
        Tensor actual = decoder.Decode(backend, latent);
        Assert.Equal(expected.Shape.ToString(), actual.Shape.ToString());

        float* a = (float*)actual.DataPointer;
        float* e = (float*)expected.DataPointer;
        double num = 0, den = 0, maxAbs = 0;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            double d = a[i] - e[i];
            num += d * d;
            den += (double)e[i] * e[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
        }
        double relL2 = Math.Sqrt(num / Math.Max(den, 1e-30));
        // The GPU convolutions accumulate in reduced precision, so the same 512-sample kernels drift further than
        // the CPU path over the vocoder's seven upsample stages.
        double tolerance = IsCuda ? 5e-3 : 1e-4;
        _output.WriteLine($"relL2={relL2:E3} maxAbs={maxAbs:E3} refRms={Math.Sqrt(den / expected.ElementCount):F5}");
        Assert.True(relL2 < tolerance, $"audio VAE decode relL2 {relL2:E3} exceeds {tolerance:E0} (maxAbs {maxAbs:E3}).");
    }

    private static bool IsCuda =>
        string.Equals(Environment.GetEnvironmentVariable("PARITY_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);

    /// <summary>CPU by default; <c>PARITY_BACKEND=cuda</c> runs the same comparison on the GPU path.</summary>
    private static IBackend MakeBackend()
    {
        if (!IsCuda)
        {
            return new CpuBackend();
        }
        string? d = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && d is not null; i++, d = Path.GetDirectoryName(d))
        {
            string cand = Path.Combine(d, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(cand)) { return new HartsyInference.Cuda.CudaBackend(0, cand); }
        }
        return new HartsyInference.Cuda.CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
    }

    /// <summary>Reads a little-endian F32 dump; one dimension may be -1 to absorb the remainder.</summary>
    private static Tensor ReadRaw(string path, params int[] dims)
    {
        byte[] bytes = File.ReadAllBytes(path);
        long count = bytes.Length / 4;
        long known = 1;
        int wild = -1;
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] < 0) { wild = i; } else { known *= dims[i]; }
        }
        long[] shape = new long[dims.Length];
        for (int i = 0; i < dims.Length; i++) { shape[i] = dims[i]; }
        if (wild >= 0) { shape[wild] = count / known; }
        else if (count != known) { throw new InvalidDataException($"{path}: {count} floats, expected {known}."); }

        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        bytes.AsSpan(0, (int)(t.ElementCount * 4)).CopyTo(new Span<byte>((void*)t.DataPointer, (int)(t.ElementCount * 4)));
        return t;
    }
}
