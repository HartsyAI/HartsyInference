using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the Kyutai/moshi depth transformer (<see cref="MoshiDepformer"/>) against the
/// real <c>kyutai/tts-1.6b-en_fr</c> checkpoint. The Python reference greedily decodes all 32 codebooks for a
/// fixed temporal context + text token and saves the per-codebook logits + tokens; this replays the same greedy
/// decode in C# and diffs both. Gated on <c>KYUTAI_TTS_WEIGHTS</c> + <c>KYUTAI_REF_DEPFORMER</c>.</summary>
public sealed unsafe class KyutaiDepformerParityTests
{
    private readonly ITestOutputHelper _out;
    public KyutaiDepformerParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Depformer_MatchesMoshiReference()
    {
        string? wp = Environment.GetEnvironmentVariable("KYUTAI_TTS_WEIGHTS");
        string? rp = Environment.GetEnvironmentVariable("KYUTAI_REF_DEPFORMER");
        if (string.IsNullOrEmpty(wp) || !File.Exists(wp) || string.IsNullOrEmpty(rp) || !File.Exists(rp)) return;

        using SafeTensorsLoader weights = new(); weights.Load(wp);
        using SafeTensorsLoader io = new(); io.Load(rp);
        IReadOnlyDictionary<string, Tensor> w = weights.GetAllTensors();
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        Tensor tout = d["transformer_out"];   // [1,1,2048]
        Tensor refLogits = d["logits"];        // [32, 2048]
        Tensor refTokens = d["tokens"];        // [32] int32
        int textToken = ((int*)d["text_token"].DataPointer)[0];

        using MoshiDepformer dep = new();
        dep.LoadWeights(w);
        using CpuBackend backend = new();
        using Tensor logits = dep.DecodeFrameGreedy(backend, tout, textToken, out int[] tokens);

        float* a = (float*)logits.DataPointer; float* b = (float*)refLogits.DataPointer;
        int* rt = (int*)refTokens.DataPointer;
        double maxAbs = 0, peak = 0; int tokenMismatches = 0;
        for (int cb = 0; cb < MoshiDepformer.DepQ; cb++)
        {
            if (tokens[cb] != rt[cb]) tokenMismatches++;
            for (int c = 0; c < MoshiDepformer.Card; c++)
            {
                double diff = Math.Abs(a[(long)cb * MoshiDepformer.Card + c] - b[(long)cb * MoshiDepformer.Card + c]);
                if (diff > maxAbs) maxAbs = diff;
                peak = Math.Max(peak, Math.Abs(b[(long)cb * MoshiDepformer.Card + c]));
            }
        }
        _out.WriteLine($"depformer logits [32,2048] maxAbs={maxAbs:E4} (ref peak {peak:F4}); token mismatches={tokenMismatches}/32.");
        _out.WriteLine($"  mine tokens[:8]=[{string.Join(",", tokens[..8])}]");
        _out.WriteLine($"  ref  tokens[:8]=[{rt[0]},{rt[1]},{rt[2]},{rt[3]},{rt[4]},{rt[5]},{rt[6]},{rt[7]}]");
        // The 32 greedy tokens are the functional contract: each codebook's token feeds the next, so a wrong
        // architecture / weight-slice / embedding would compound and flip tokens. Exact 32/32 match is the strong
        // check; the logit residual is f32 attention-accumulation noise on large logits (peak ~6.9).
        Assert.Equal(0, tokenMismatches);
        Assert.True(maxAbs < 3e-2, $"depformer diverges (maxAbs={maxAbs:E4}).");
    }
}
