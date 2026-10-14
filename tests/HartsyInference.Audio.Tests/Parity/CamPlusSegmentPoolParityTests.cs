using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>The CAM++ speaker embedding under each upstream's runtime, on real weights, CPU backend, F32: Chatterbox runs
/// PyTorch <c>CAMPPlus</c> (partial context segment divided by the frames it covers), CosyVoice 2 runs
/// <c>campplus.onnx</c> under an optimizing ONNX Runtime session (divided by 100). The reference dump comes from
/// tools/parity/campplus_segment_pool_dump.py (path in <c>CAMPPLUS_SEGMENT_POOL_DUMP</c>); the weights are
/// Chatterbox's <c>s3gen.safetensors</c> under <c>HARTSY_INDEXTTS_MODELS_ROOT</c>.</summary>
[Trait("Category", "Integration")]
public sealed class CamPlusSegmentPoolParityTests(ITestOutputHelper output)
{
    private const string DumpVariable = "CAMPPLUS_SEGMENT_POOL_DUMP";

    [Theory]
    [InlineData(false, "emb_chatterbox")]
    [InlineData(true, "emb_cosyvoice")]
    public void Embedding_MatchesTheUpstreamRuntime(bool onnxSegmentPooling, string key)
    {
        string s3genPath = Path.Combine(IndexTtsFrontendFixture.ModelsRoot, "ResembleAI--chatterbox", "s3gen.safetensors");
        string dumpPath = Environment.GetEnvironmentVariable(DumpVariable) ?? "";
        if (!RealWeightGate.Require(output.WriteLine, s3genPath, dumpPath)) return;

        using SafeTensorsLoader dump = new();
        dump.Load(dumpPath);
        Dictionary<string, Tensor> d = dump.GetAllTensors();
        using SafeTensorsLoader s3gen = new();
        s3gen.Load(s3genPath);
        using CamPlusSpeakerEncoder encoder = new(192, onnxSegmentPooling);
        encoder.LoadWeights(s3gen.GetAllTensors(), "speaker_encoder");
        using CpuBackend backend = new();

        Tensor fbank = d["fbank"];
        using Tensor feat = new(new TensorShape(1, fbank.Shape[0], fbank.Shape[1]), DType.F32);
        fbank.AsSpan<float>().CopyTo(feat.AsSpan<float>());
        using Tensor fromFbank = encoder.Forward(backend, feat);
        using Tensor fromAudio = S3GenReference.SpeakerEmbedding(backend, encoder, d["audio16k"].AsSpan<float>());
        ReadOnlySpan<float> want = d[key].AsSpan<float>();
        double encoderErr = RelRms(fromFbank.AsSpan<float>(), want), frontErr = RelRms(fromAudio.AsSpan<float>(), want);
        double otherErr = RelRms(fromFbank.AsSpan<float>(), d[onnxSegmentPooling ? "emb_chatterbox" : "emb_cosyvoice"].AsSpan<float>());
        output.WriteLine($"{key}: upstream fbank relRMS {encoderErr:E2}, C# fbank relRMS {frontErr:E2}, other runtime {otherErr:E2}");
        Assert.True(encoderErr < 1e-4, $"{key} from the upstream fbank: relRMS {encoderErr:E2}");
        Assert.True(frontErr < 1e-3, $"{key} from the clip: relRMS {frontErr:E2}");
    }

    private static double RelRms(ReadOnlySpan<float> got, ReadOnlySpan<float> want)
    {
        Assert.Equal(want.Length, got.Length);
        double num = 0, den = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double diff = got[i] - want[i];
            num += diff * diff;
            den += (double)want[i] * want[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
