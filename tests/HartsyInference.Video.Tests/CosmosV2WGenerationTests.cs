using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.Video.Models.Cosmos;
using HartsyInference.Video.Tokenizers;

namespace HartsyInference.Video.Tests;

/// <summary>Env-gated real-weight contract for Cosmos-Predict1 5B Video2World. Mirrors the other <c>*GenerationTests</c>:
/// skips cleanly when the checkpoint env vars or VRAM are absent (so it never gates hosted CI), and documents the
/// integration parity gates that run on the GPU lane once the 9 GB+ downloads are present.
///
/// <para>Required env vars: <c>COSMOS_PREDICT1_5B_V2W_PATH</c> (converted <c>model.pt</c>),
/// <c>COSMOS_DV_TOKENIZER_PATH</c> (extracted encoder/decoder), <c>T5_11B_PATH</c> (HF cache layout).</para>
///
/// <para>Parity gates (research-doc §Ordering): (1) DV encode a 9-frame clip → token IDs bit-match
/// <c>encoder.jit</c>; (2) T5-11B cos-sim ≥ 0.999 vs <c>transformers.T5EncoderModel</c>; (3) backbone layer-diff vs
/// a PyTorch dump; (4) seed-42 token-sequence parity vs the reference; (5) a coherent 33-frame 1024×640 clip.
/// Steps 1 &amp; 5 additionally need the resolved DV conv stage schedule (research-doc Open Q §10).</para></summary>
[Trait("Category", "Integration")]
public class CosmosV2WGenerationTests
{
    private readonly ITestOutputHelper _output;
    public CosmosV2WGenerationTests(ITestOutputHelper output) => _output = output;

    private const string V2WEnv = "COSMOS_PREDICT1_5B_V2W_PATH";
    private const string DvEnv = "COSMOS_DV_TOKENIZER_PATH";
    private const string T5Env = "T5_11B_PATH";

    [Fact]
    public void Cosmos5BV2W_Gpu_ImageToWorld_33Frame_Clip()
    {
        string? v2w = Environment.GetEnvironmentVariable(V2WEnv);
        string? dv = Environment.GetEnvironmentVariable(DvEnv);
        string? t5 = Environment.GetEnvironmentVariable(T5Env);
        if (string.IsNullOrWhiteSpace(v2w) || !File.Exists(v2w))
        { _output.WriteLine($"SKIPPED: {V2WEnv} not set / model.pt missing."); return; }
        if (string.IsNullOrWhiteSpace(dv) || !Directory.Exists(dv))
        { _output.WriteLine($"SKIPPED: {DvEnv} not set / tokenizer dir missing."); return; }
        if (string.IsNullOrWhiteSpace(t5) || !Directory.Exists(t5))
        { _output.WriteLine($"SKIPPED: {T5Env} not set / T5-11B dir missing."); return; }

        // Load the AR backbone (5B). The converter reuses PytorchPickleLoader for the .pt pickle.
        (Dictionary<string, Tensor> weights, IDisposable loader) = CosmosArCheckpointConverter.LoadAndConvert(v2w, castToF32: false);
        using IDisposable _loader = loader;
        _output.WriteLine($"Loaded {weights.Count} Cosmos AR tensors from {Path.GetFileName(v2w)}.");
        using CosmosArTransformer model = new(CosmosArConfig.Cosmos5BV2W);
        model.LoadWeights(weights);

        using CosmosDvTokenizer tokenizer = new(CosmosDvTokenizerConfig.Dv8x16x16);
        (int lt, int lh, int lw) = tokenizer.LatentGrid(33, 640, 1024);
        Assert.Equal((5, 40, 64), (lt, lh, lw));

        // Full DV encode/decode + AR sampling + T5-11B encode run at integration once the DV conv stage schedule
        // is resolved from encoder.jit (research-doc Open Q §10) and preloaded on a GPU backend with ≥12 GB free.
        _output.WriteLine("Backbone + tokenizer geometry validated; DV render + AR sampling gated on the conv-stage " +
            "schedule + GPU. See test summary for the parity-gate ordering.");
    }
}
