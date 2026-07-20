using HartsyInference.ModelAssets.TextEncoders.Bert;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity for the BERT encoder against the real <c>chinese-roberta-wwm-ext-large</c> weights
/// (GPT-SoVITS text features). Golden values are the <c>hidden_states[-3]</c> tap (output of layer 22)
/// computed by an independent reference. Gated on <c>HARTSYINFERENCE_ROBERTA_PT</c>; skips cleanly when absent.</summary>
public sealed unsafe class BertModelTests
{
    private static readonly int[] Ids = [101, 1, 2, 3, 4, 5, 102];     // CLS + 5 + SEP

    [Fact]
    public void Forward_MatchesReference_HiddenStatesMinus3()
    {
        string? pt = Environment.GetEnvironmentVariable("HARTSYINFERENCE_ROBERTA_PT");
        if (string.IsNullOrEmpty(pt) || !File.Exists(pt)) return;

        using CpuBackend backend = new();
        using PytorchPickleLoader loader = new();
        loader.Load(pt);

        BertConfig cfg = BertConfig.ChineseRobertaWwmExtLarge;
        using BertModel bert = new(cfg);
        bert.LoadWeights(loader.GetAllTensors());

        // hidden_states[-3] = output after layer (NumLayers - 2) = 22.
        Tensor feat = bert.Forward(backend, Ids, cfg.NumLayers - 2);
        try
        {
            Assert.Equal(Ids.Length, (int)feat.Shape[1]);
            Assert.Equal(cfg.Hidden, (int)feat.Shape[2]);
            float* p = (float*)feat.DataPointer;
            int h = cfg.Hidden;

            float[] row0 = [-0.06055f, -0.04514f, 0.11249f, 0.15064f, 0.03143f];
            float[] row3 = [-0.54795f, 0.62429f, -0.68094f, 0.46904f, -0.52525f];
            for (int c = 0; c < 5; c++) Assert.Equal(row0[c], p[c], precision: 3);
            for (int c = 0; c < 5; c++) Assert.Equal(row3[c], p[3 * h + c], precision: 3);

            float sum = 0f, sq = 0f;
            for (long i = 0; i < feat.ElementCount; i++) { sum += p[i]; sq += p[i] * p[i]; }
            float mean = sum / feat.ElementCount;
            float std = MathF.Sqrt(sq / feat.ElementCount - mean * mean);
            Assert.Equal(0.01264f, mean, precision: 3);
            Assert.Equal(1.02438f, std, precision: 2);
        }
        finally { feat.Dispose(); }
    }
}
