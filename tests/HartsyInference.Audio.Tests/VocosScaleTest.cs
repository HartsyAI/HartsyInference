using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Isolates Vocos scale: feed F5's OWN reference generated-mel (f5_ref_mel_f32.bin,
/// [1,100,T] f32, python output RMS 0.1524) into our Vocos and compare output RMS. If ours
/// is ~0.15 the vocoder scale is correct; if ~1.6+ Vocos has a constant gain bug.</summary>
public sealed class VocosScaleTest
{
    private readonly ITestOutputHelper _out;
    public VocosScaleTest(ITestOutputHelper o) => _out = o;

    [Fact]
    public async Task VocosOnF5RefMel_ReportsRms()
    {
        string bin = "/tmp/f5_ref_mel_f32.bin";
        if (!File.Exists(bin)) { _out.WriteLine("no ref mel bin — skip"); return; }
        byte[] raw = File.ReadAllBytes(bin);
        int t = raw.Length / 4 / 100;
        Tensor mel = new(new TensorShape(1, 100, t), DType.F32);
        unsafe { fixed (byte* p = raw) Buffer.MemoryCopy(p, (void*)mel.DataPointer, raw.Length, raw.Length); }

        string vocosPath = await AudioModelCache.GetAsync("lucasnewman/vocos-mel-24khz", "model.safetensors");
        SafeTensorsLoader loader = new(); loader.Load(vocosPath);
        Vocos vocos = new(VocosConfig.Mel24k);
        vocos.LoadWeights(loader.GetAllTensors());

        using CpuBackend b = new();
        float[] audio = vocos.Forward(b, mel);
        double sumSq = 0; foreach (float v in audio) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / audio.Length);
        _out.WriteLine($"[VOCOS SCALE] our Vocos on F5 ref mel [100,{t}]: {audio.Length / 24000.0:F2}s | RMS {rms:F4} (F5 python = 0.1524)");
    }
}
