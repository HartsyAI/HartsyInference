using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Exact token parity for the S3 speech tokenizer (CosyVoice2 <c>speech_tokenizer_v2_25hz</c>) against
/// the upstream model — which is token-identical to the bundled <c>speech_tokenizer_v2.onnx</c> (verified via
/// ONNX Runtime). Golden token ids come from the <c>s3tokenizer</c> package on a deterministic 128-bin mel.
/// Gated on <c>HARTSYINFERENCE_S3_V2_ST</c> (a clean safetensors export of the model); skips cleanly when absent.</summary>
public sealed unsafe class S3TokenizerParityTests
{
    // s3tokenizer.load_model("speech_tokenizer_v2_25hz").quantize(mel, [200]) on the mel built below.
    private static readonly int[] Golden =
    [
        4018, 4000, 3280, 3999, 4000, 3930, 3271, 3282, 3280, 3280, 3270, 5467, 4000, 3280, 3999, 4000,
        3930, 3271, 3282, 3280, 3280, 3999, 3280, 3999, 3280, 3999, 4000, 3930, 3280, 3279, 3280, 3280,
        3999, 3361, 3999, 3280, 4002, 4000, 4011, 3280, 3279, 3280, 3280, 3999, 3280, 3999, 3271, 4003,
        4000, 3283,
    ];

    [Fact]
    public void Forward_MatchesUpstreamTokens_OnRealWeights()
    {
        string? st = Environment.GetEnvironmentVariable("HARTSYINFERENCE_S3_V2_ST");
        if (string.IsNullOrEmpty(st) || !File.Exists(st)) return;

        using CpuBackend backend = new();
        using SafeTensorsLoader loader = new();
        loader.Load(st);

        using S3Tokenizer s3 = new();
        s3.LoadWeights(loader.GetAllTensors());

        const int t = 200;
        Tensor mel = new(new TensorShape(1, 128, t), DType.F32);
        float* mp = (float*)mel.DataPointer;
        for (int c = 0; c < 128; c++)
            for (int i = 0; i < t; i++)
                mp[c * t + i] = (((c * 5 + i * 11) % 97) / 97f - 0.5f) * 2f;

        int[] tokens = s3.Forward(backend, mel);
        mel.Dispose();

        Assert.Equal(Golden.Length, tokens.Length);
        Assert.Equal(Golden, tokens);
    }
}
