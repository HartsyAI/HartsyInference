using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight CPU parity of SANA-WM blocks 0 (Gated-DeltaNet) and 3 (softmax) against the upstream unit dumps.
/// Needs <c>HARTSYINFERENCE_MODELS_DIR</c> (the DiT) and <c>SANAWM_REFERENCE_RUN_DIR</c> (a run directory holding
/// <c>units.safetensors</c>, <c>sampler.safetensors</c>, <c>text.safetensors</c> and <c>camera.safetensors</c>); skipped otherwise.</summary>
[Trait("Category", "Integration")]
public sealed class SanaWmBlockRealWeightsTests : IClassFixture<SanaWmBlockRealWeightsTests.Fixture>
{
    private const int Frames = 21;
    private const int Height = 22;
    private const int Width = 40;
    private const double RelativeTarget = 1e-2;
    private readonly Fixture _fixture;
    private readonly ITestOutputHelper _output;

    public SanaWmBlockRealWeightsTests(Fixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SanaWmBlockReferenceFact]
    public void Block0_Gdn_MatchesUpstreamUnitOutput()
    {
        RunBlockParity(0, "unit_block0_in", "unit_block0_out");
    }

    [SanaWmBlockReferenceFact]
    public void Block3_Softmax_MatchesUpstreamUnitOutput()
    {
        RunBlockParity(3, "unit_block3_in", "unit_block3_out");
    }

    // Sub-branches of block 0 against the hooks the dump recorded: the FFN on its own captured input, and the camera branch
    // (after out_proj_cam) on an adaLN input rebuilt here on the host, independently of SanaWmBlock.
    [SanaWmBlockReferenceFact]
    public void Block0_FfnAndCameraBranches_MatchUpstreamCaptures()
    {
        using IBackend backend = new CpuBackend();
        string runDirectory = SanaWmBlockReferenceFactAttribute.ReferenceDirectory();
        using SafeTensorsLoader units = Load(Path.Combine(runDirectory, "units.safetensors"));
        using SafeTensorsLoader sampler = Load(Path.Combine(runDirectory, "sampler.safetensors"));
        using SafeTensorsLoader camera = Load(Path.Combine(runDirectory, "camera.safetensors"));
        SanaWmConfig config = _fixture.Checkpoint.Config;
        IReadOnlyDictionary<string, Tensor> weights = _fixture.Checkpoint.Weights;
        List<Tensor> owned = [];
        List<Tensor> scratch = [];
        try
        {
            SanaWmGlumbTempFfn ffn = new(
                ToF32(backend, weights["blocks.0.mlp.inverted_conv.conv.weight"], owned), weights["blocks.0.mlp.inverted_conv.conv.bias"],
                ToF32(backend, weights["blocks.0.mlp.depth_conv.conv.weight"], owned), weights["blocks.0.mlp.depth_conv.conv.bias"],
                ToF32(backend, weights["blocks.0.mlp.point_conv.conv.weight"], owned), ToF32(backend, weights["blocks.0.mlp.t_conv.weight"], owned));
            using (Tensor ffnOutput = ffn.Forward(backend, ToF32(backend, units.GetTensor("unit_block0_mlp_in"), owned), Frames, Height, Width))
            {
                Report("unit_block0_mlp_out", ffnOutput.AsReadOnlySpan<float>(),
                    ToF32(backend, units.GetTensor("unit_block0_mlp_out"), owned).AsReadOnlySpan<float>());
            }

            SanaWmTimeWeights time = new(weights["t_embedder.mlp.0.weight"], weights["t_embedder.mlp.0.bias"],
                weights["t_embedder.mlp.2.weight"], weights["t_embedder.mlp.2.bias"], weights["t_block.1.weight"],
                weights["t_block.1.bias"]);
            float[] timesteps = Floats(backend, sampler.GetTensor("dit_timestep_step0"), owned);
            (Tensor _, Tensor blockTime) = SanaWmConditioning.Time(backend, time, config.TimestepWidth, timesteps, scratch);
            Tensor x = ToF32(backend, units.GetTensor("unit_block0_in"), owned);
            Tensor attentionInput = HostAttentionInput(x, blockTime, weights["blocks.0.scale_shift_table"], owned);
            (Tensor cos, Tensor sin, Tensor cameraCos, Tensor cameraSin) =
                SanaWmConditioning.Rope(backend, config.HeadDim, Frames, Height, Width, scratch);

            using SanaWmGdnBidi gdn = new(backend, new SanaWmGdnWeights
            {
                Qkv = weights["blocks.0.attn.qkv.weight"],
                ConvK = weights["blocks.0.attn.conv_k.weight"],
                BetaWeight = weights["blocks.0.attn.beta_proj.weight"],
                BetaBias = weights["blocks.0.attn.beta_proj.bias"],
                GateWeight = weights["blocks.0.attn.gate_proj.weight"],
                GateBias = weights["blocks.0.attn.gate_proj.bias"],
                ALog = weights["blocks.0.attn.A_log"],
                DtBias = weights["blocks.0.attn.dt_bias"],
                QNorm = weights["blocks.0.attn.q_norm.weight"],
                KNorm = weights["blocks.0.attn.k_norm.weight"],
            }, config.NumHeads, config.HeadDim);
            (Tensor main, Tensor beta, Tensor decay) = gdn.ForwardWithGates(attentionInput, Frames, Height, Width, cos, sin);
            scratch.Add(main);
            scratch.Add(beta);
            scratch.Add(decay);

            Dictionary<string, Tensor> cameraWeights = new(StringComparer.Ordinal);
            foreach (string key in new[] { "q_proj_cam.weight", "q_proj_cam.bias", "k_proj_cam.weight", "k_proj_cam.bias",
                "v_proj_cam.weight", "v_proj_cam.bias", "q_norm_cam.weight", "k_norm_cam.weight", "out_proj_cam.weight",
                "out_proj_cam.bias", "conv_k_cam.weight" })
                cameraWeights[key] = weights["blocks.0.attn." + key];
            using SanaWmCamBranch cameraBranch = new(config.NumHeads, config.HeadDim);
            cameraBranch.LoadWeights(cameraWeights);
            float[] cameraFloats = Floats(backend, camera.GetTensor("dit_camera_conditions_step0"), owned);
            using Tensor contribution = cameraBranch.Forward(backend, attentionInput, cameraFloats, Frames, Height, Width,
                cameraCos, cameraSin, beta, decay);
            Report("unit_block0_cam_contrib", contribution.AsReadOnlySpan<float>(),
                ToF32(backend, units.GetTensor("unit_block0_cam_contrib"), owned).AsReadOnlySpan<float>());
        }
        finally
        {
            SanaWmTensors.DisposeAll(scratch);
            SanaWmTensors.DisposeAll(owned);
        }
    }

    // LayerNorm (no affine, eps 1e-6) then x·(1 + scale) + shift with per-frame (table + t0) slots 0 and 1, in double.
    private static Tensor HostAttentionInput(Tensor x, Tensor blockTime, Tensor table, List<Tensor> owned)
    {
        int channels = (int)x.Shape[2];
        int tokens = (int)x.Shape[1];
        int tokensPerFrame = tokens / Frames;
        ReadOnlySpan<float> source = x.AsReadOnlySpan<float>();
        ReadOnlySpan<float> time = blockTime.AsReadOnlySpan<float>();
        ReadOnlySpan<float> slots = table.AsReadOnlySpan<float>();
        Tensor result = new(x.Shape, DType.F32, DeviceKind.Cpu);
        owned.Add(result);
        Span<float> destination = result.AsSpan<float>();
        long rows = x.Shape[0] * tokens;
        for (long row = 0; row < rows; row++)
        {
            ReadOnlySpan<float> values = source.Slice((int)(row * channels), channels);
            double mean = 0, variance = 0;
            foreach (float value in values)
                mean += value;
            mean /= channels;
            foreach (float value in values)
                variance += (value - mean) * (value - mean);
            double inverse = 1.0 / Math.Sqrt(variance / channels + 1e-6);
            long frameRow = (row / tokens) * Frames + (row % tokens) / tokensPerFrame;
            int timeOffset = (int)(frameRow * 6 * channels);
            for (int c = 0; c < channels; c++)
            {
                double shift = slots[c] + time[timeOffset + c];
                double scale = slots[channels + c] + time[timeOffset + channels + c];
                destination[(int)(row * channels) + c] = (float)((values[c] - mean) * inverse * (1 + scale) + shift);
            }
        }

        return result;
    }

    private void RunBlockParity(int index, string inputKey, string outputKey)
    {
        using IBackend backend = new CpuBackend();
        string runDirectory = SanaWmBlockReferenceFactAttribute.ReferenceDirectory();
        using SafeTensorsLoader units = Load(Path.Combine(runDirectory, "units.safetensors"));
        using SafeTensorsLoader sampler = Load(Path.Combine(runDirectory, "sampler.safetensors"));
        using SafeTensorsLoader text = Load(Path.Combine(runDirectory, "text.safetensors"));
        using SafeTensorsLoader camera = Load(Path.Combine(runDirectory, "camera.safetensors"));
        SanaWmConfig config = _fixture.Checkpoint.Config;
        IReadOnlyDictionary<string, Tensor> weights = _fixture.Checkpoint.Weights;
        List<Tensor> owned = [];
        List<Tensor> scratch = [];
        try
        {
            Tensor x = ToF32(backend, units.GetTensor(inputKey), owned);
            float[] timesteps = Floats(backend, sampler.GetTensor("dit_timestep_step0"), owned);
            float[] cameraFloats = Floats(backend, camera.GetTensor("dit_camera_conditions_step0"), owned);
            Tensor textTokens = ToF32(backend, text.GetTensor("dit_y_step0"), owned);
            Tensor mask = ToF32(backend, MaskAsFloat(text.GetTensor("dit_mask_step0"), owned), owned);
            Tensor chunkPlucker = ToF32(backend, camera.GetTensor("dit_chunk_plucker_step0"), owned);

            SanaWmTimeWeights time = new(weights["t_embedder.mlp.0.weight"], weights["t_embedder.mlp.0.bias"],
                weights["t_embedder.mlp.2.weight"], weights["t_embedder.mlp.2.bias"], weights["t_block.1.weight"],
                weights["t_block.1.bias"]);
            (Tensor _, Tensor blockTime) = SanaWmConditioning.Time(backend, time, config.TimestepWidth, timesteps, scratch);
            SanaWmCaptionWeights caption = new(
                weights["y_embedder.y_proj.fc1.weight"], weights["y_embedder.y_proj.fc1.bias"],
                weights["y_embedder.y_proj.fc2.weight"], weights["y_embedder.y_proj.fc2.bias"],
                weights["attention_y_norm.weight"]);
            Tensor memory = SanaWmConditioning.Caption(backend, caption, textTokens, 2, 300, scratch);
            Tensor memoryBias = SanaWmConditioning.MaskBias(backend, mask, 2, 300, scratch);
            Tensor pluckerWeight = weights["plucker_embedder.proj.weight"].Reshape(new TensorShape([config.HiddenSize, config.PluckerChannels]));
            owned.Add(pluckerWeight);
            Tensor plucker = SanaWmConditioning.Plucker(backend, pluckerWeight, weights["plucker_embedder.proj.bias"], chunkPlucker, scratch);

            using SanaWmBlock block = new(backend, config, index, weights);
            using SanaWmForwardContext context = SanaWmForwardContext.Create(backend, config, 2, Frames, Height, Width, blockTime, memory,
                memoryBias, plucker, cameraFloats);
            using Tensor output = block.Forward(x, context);
            Tensor expected = ToF32(backend, units.GetTensor(outputKey), owned);
            Report(outputKey, output.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        }
        finally
        {
            SanaWmTensors.DisposeAll(scratch);
            SanaWmTensors.DisposeAll(owned);
        }
    }

    private void Report(string label, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double maxAbs = 0, sumSquaredError = 0, sumSquaredExpected = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double difference = actual[i] - expected[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(difference));
            sumSquaredError += difference * difference;
            sumSquaredExpected += (double)expected[i] * expected[i];
        }

        double relative = Math.Sqrt(sumSquaredError / Math.Max(sumSquaredExpected, double.Epsilon));
        _output.WriteLine($"{label}: max abs {maxAbs:E3}, relative (L2) {relative:E3}");
        Assert.True(relative < RelativeTarget, $"{label}: relative error {relative:E3} exceeds {RelativeTarget:E0}; max abs {maxAbs:E3}.");
    }

    private static SafeTensorsLoader Load(string path)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return loader;
    }

    private static Tensor ToF32(IBackend backend, Tensor source, List<Tensor> owned) => SanaWmTensors.HostF32(source, owned);

    private static float[] Floats(IBackend backend, Tensor source, List<Tensor> owned)
    {
        Tensor values = ToF32(backend, source, owned);
        return values.AsReadOnlySpan<float>().ToArray();
    }

    private static Tensor MaskAsFloat(Tensor mask, List<Tensor> owned)
    {
        Tensor converted = new(mask.Shape, DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<long> source = mask.AsReadOnlySpan<long>();
        Span<float> destination = converted.AsSpan<float>();
        for (int i = 0; i < destination.Length; i++)
            destination[i] = source[i];
        owned.Add(converted);
        return converted;
    }

    /// <summary>Loads the converted checkpoint once for the class: blocks 0 and 3 and the conditioning weights.</summary>
    public sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            string path = SanaWmRealWeightsFactAttribute.ResolveDitPath();
            Checkpoint = SanaWmCheckpointConverter.Load(path, KeepForBlockParity);
        }

        public SanaWmCheckpoint Checkpoint { get; }

        public void Dispose() => Checkpoint.Dispose();

        private static bool KeepForBlockParity(string key) =>
            key.StartsWith("blocks.0.", StringComparison.Ordinal) || key.StartsWith("blocks.3.", StringComparison.Ordinal)
            || key.StartsWith("t_embedder.", StringComparison.Ordinal) || key.StartsWith("t_block.", StringComparison.Ordinal)
            || key.StartsWith("y_embedder.y_proj.", StringComparison.Ordinal) || key.StartsWith("attention_y_norm.", StringComparison.Ordinal)
            || key.StartsWith("plucker_embedder.", StringComparison.Ordinal);
    }
}
