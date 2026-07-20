using System;
using System.IO;
using HartsyInference.Audio.Models.Hubert;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for <see cref="Hubert"/> against the real <c>chinese-hubert-base</c> (the
/// GPT-SoVITS reference-audio SSL encoder). Loads the checkpoint (pos_conv composed from its weight_norm
/// pair), runs the same fixed 1 s audio, and diffs <c>last_hidden_state</c> against the PyTorch reference.
/// Gated on <c>HUBERT_PATH</c> (pytorch_model.bin) + <c>REF_HUBERT_IO</c>.</summary>
public sealed unsafe class HubertParityTests
{
    private readonly ITestOutputHelper _out;
    public HubertParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Hubert_MatchesPythonReference()
    {
        string? hp = Environment.GetEnvironmentVariable("HUBERT_PATH");
        string? refIo = Environment.GetEnvironmentVariable("REF_HUBERT_IO");
        if (string.IsNullOrEmpty(hp) || !File.Exists(hp) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        using PytorchPickleLoader loader = new();
        loader.Load(hp);
        using Hubert hubert = new(new HubertConfig());
        hubert.LoadWeights(loader.GetAllTensors());

        SafeTensorsLoader io = new();
        io.Load(refIo);
        System.Collections.Generic.IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();
        Tensor audio = d["audio"];        // [1, 16000]
        Tensor refOut = d["hubert_out"];  // [1, T, 768]
        int n = (int)audio.Shape[audio.Shape.Rank - 1];
        int tRef = (int)refOut.Shape[1], hid = (int)refOut.Shape[2];

        Tensor pcm = new(new TensorShape(1, 1, n), DType.F32);
        Buffer.MemoryCopy((void*)audio.DataPointer, (void*)pcm.DataPointer, (long)n * 4, (long)n * 4);

        using CpuBackend backend = new();
        using Tensor feats = hubert.Forward(backend, pcm, n);   // [1, 768, T] channels-first
        pcm.Dispose();
        int tMine = (int)feats.Shape[2];
        _out.WriteLine($"mine [1,{feats.Shape[1]},{tMine}]  ref [1,{tRef},{hid}].");
        Assert.Equal(tRef, tMine);

        float* a = (float*)feats.DataPointer;   // [hid, T]
        float* b = (float*)refOut.DataPointer;  // [T, hid]
        double maxAbs = 0, refPeak = 0;
        for (int t = 0; t < tMine; t++)
            for (int c = 0; c < hid; c++)
            {
                double diff = Math.Abs(a[(long)c * tMine + t] - b[(long)t * hid + c]);
                if (diff > maxAbs) maxAbs = diff;
                refPeak = Math.Max(refPeak, Math.Abs(b[(long)t * hid + c]));
            }
        _out.WriteLine($"maxAbs={maxAbs:E4} (ref peak {refPeak:F4}). mine[:,0][:5]=[{a[0]:F5},{a[tMine]:F5},{a[2L*tMine]:F5},{a[3L*tMine]:F5},{a[4L*tMine]:F5}]");
        _out.WriteLine($"  ref[0][:5]=[{b[0]:F5},{b[1]:F5},{b[2]:F5},{b[3]:F5},{b[4]:F5}]");
        Assert.True(maxAbs < 5e-3, $"hubert diverges (maxAbs={maxAbs:E4}).");
    }
}
