using HartsyInference.Core.Rope;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>DeepSeek-V2 YaRN rope: HF <c>_compute_yarn_parameters</c> (and the official DeepSeek code) floors the low and
/// ceils the high correction bound. The inverse frequencies below are what transformers' DeepseekV2RotaryEmbedding
/// builds from the DeepSeek-V2-Lite config alone (dim 64, theta 10000, factor 40 over 4096, beta 32/1). They were generated
/// with transformers 5.17 and no weights loaded; attention_scaling is 1.0.</summary>
public sealed class RopeScalingDeepSeekV2Tests
{
    private static readonly double[] HfInvFreq =
    [
        1.0, 0.74989420175552368, 0.56234133243560791, 0.42169651389122009, 0.31622776389122009,
        0.23713736236095428, 0.17782793939113617, 0.13335214555263519, 0.10000000149011612,
        0.07498941570520401, 0.056234128773212433, 0.039006926119327545, 0.026879360899329185,
        0.018378144130110741, 0.012447956018149853, 0.0083345090970396996, 0.0055000004358589649,
        0.0035619973205029964, 0.0022493652068078518, 0.0013705134624615312, 0.0007905694073997438,
        0.00041499041253700852, 0.00017782794020604342, 3.333803397254087e-05, 2.4999999368446879e-05,
        1.8747354260995053e-05, 1.4058532542549074e-05, 1.0542412383074407e-05, 7.9056944741751067e-06,
        5.9284343478793744e-06, 4.4456983232521452e-06, 3.3338035336782923e-06,
    ];

    private readonly ITestOutputHelper _output;

    public RopeScalingDeepSeekV2Tests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Yarn_DeepSeekV2Lite_TruncatedCorrectionRange_MatchesHfInvFreq()
    {
        RopeScaling s = new()
        {
            Type = RopeScalingType.Yarn, Factor = 40.0, OriginalContextLength = 4096.0, BetaFast = 32.0,
            BetaSlow = 1.0, AttentionFactor = 1.0, TruncateYarnCorrectionRange = true,
        };
        (double[] inv, double mscale) = RopeFrequencyBuilder.Build(64, 10_000.0, s, 128);
        Assert.Equal(HfInvFreq.Length, inv.Length);

        double maxRel = 0.0;
        for (int k = 0; k < HfInvFreq.Length; k++)
            maxRel = Math.Max(maxRel, Math.Abs(inv[k] - HfInvFreq[k]) / HfInvFreq[k]);
        _output.WriteLine($"max relative inv_freq diff vs HF: {maxRel:E3}");

        Assert.True(maxRel <= 1e-6, $"max relative inv_freq diff vs HF is {maxRel:E3}, expected <= 1e-6");
        Assert.Equal(1.0, mscale);
    }
}
