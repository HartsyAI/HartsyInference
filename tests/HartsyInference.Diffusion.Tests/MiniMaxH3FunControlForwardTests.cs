using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Small CPU oracle for native Fun branch scheduling and residual injection semantics.</summary>
[Trait("Category", "SyntheticSmoke")]
public unsafe class MiniMaxH3FunControlForwardTests
{
    [Fact]
    public void BeforeProjectionTransformsEveryPackedRowAfterReplacingOnlyTargetVideoRows()
    {
        MiniMaxH3Config config = TinyConfig(timeDim: 2, curves: true);
        IBackend backend = new CpuBackend();
        using MiniMaxH3Transformer transformer = new MiniMaxH3Transformer(config);
        transformer.LoadWeights(BuildBaseWeights(config));
        Dictionary<string, Tensor> controlWeights = BuildControlWeights(config);
        Replace(controlWeights, "control_proj_in.weight", RectIdentity(config.HiddenSize, 196));
        Replace(controlWeights, "control_blocks.0.before_proj.weight",
            RectIdentity(config.HiddenSize, config.HiddenSize));
        Replace(controlWeights, "control_blocks.0.before_proj.bias", Ramp(config.HiddenSize, 0.01f));
        MiniMaxH3FunControlNet controlNet = new MiniMaxH3FunControlNet(
            MiniMaxH3FunControlConfig.Detect(controlWeights));
        controlNet.LoadWeights(controlWeights);
        int modelIndex = transformer.RegisterFunControlNet(controlNet);

        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(1, 1, 2, 2, 1);
        using Tensor hidden = RowValues(layout.SequenceLength, config.HiddenSize, 1f);
        using Tensor controlRows = MatrixValues(1, 196, 20f);
        MiniMaxH3FunControlCondition condition = new MiniMaxH3FunControlCondition
        {
            ModelIndex = modelIndex,
            ControlRows = controlRows,
            Strength = 1f,
        };

        using Tensor state = transformer.InitializeFunControlStream(backend, hidden, layout, condition);
        MiniMaxH3Segment targetVideo = layout.Segments.Single(segment => segment.Kind == MiniMaxH3SegmentKind.Video);
        float* hiddenPointer = (float*)hidden.DataPointer;
        float* controlPointer = (float*)controlRows.DataPointer;
        float* statePointer = (float*)state.DataPointer;
        for (int row = 0; row < layout.SequenceLength; row++)
        {
            for (int column = 0; column < config.HiddenSize; column++)
            {
                float projectedInput = row >= targetVideo.Start && row < targetVideo.Stop
                    ? controlPointer[(row - targetVideo.Start) * 196 + column]
                    : hiddenPointer[row * config.HiddenSize + column];
                float expected = hiddenPointer[row * config.HiddenSize + column]
                    + projectedInput + (column + 1) * 0.01f;
                Assert.Equal(expected, statePointer[row * config.HiddenSize + column], 5);
            }
        }
    }

    [Fact]
    public void AfterProjectionReturnsFullSequenceSkipAndZerosOnlyTargetAudioRows()
    {
        MiniMaxH3Config config = TinyConfig(timeDim: 2, curves: true);
        IBackend backend = new CpuBackend();
        using MiniMaxH3Transformer transformer = new MiniMaxH3Transformer(config);
        transformer.LoadWeights(BuildBaseWeights(config));
        Dictionary<string, Tensor> controlWeights = BuildControlWeights(config);
        Replace(controlWeights, "control_blocks.0.after_proj.weight",
            RectIdentity(config.HiddenSize, config.HiddenSize));
        Replace(controlWeights, "control_blocks.0.after_proj.bias", Ramp(config.HiddenSize, 0.02f));
        MiniMaxH3FunControlNet controlNet = new MiniMaxH3FunControlNet(
            MiniMaxH3FunControlConfig.Detect(controlWeights));
        controlNet.LoadWeights(controlWeights);
        int modelIndex = transformer.RegisterFunControlNet(controlNet);

        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(1, 1, 2, 2, 1);
        MiniMaxH3Segment targetAudio = layout.Segments.Single(segment => segment.Kind == MiniMaxH3SegmentKind.Audio);
        using Tensor state = RowValues(layout.SequenceLength, config.HiddenSize, 3f);
        using Tensor skip = transformer.BuildFunControlSkip(backend, state, modelIndex, 0, targetAudio);
        float* statePointer = (float*)state.DataPointer;
        float* skipPointer = (float*)skip.DataPointer;
        for (int row = 0; row < layout.SequenceLength; row++)
        {
            bool isTargetAudio = row >= targetAudio.Start && row < targetAudio.Stop;
            for (int column = 0; column < config.HiddenSize; column++)
            {
                float expected = isTargetAudio
                    ? 0f
                    : statePointer[row * config.HiddenSize + column] + (column + 1) * 0.02f;
                Assert.Equal(expected, skipPointer[row * config.HiddenSize + column], 5);
            }
        }
    }

    [Fact]
    public void LastInjectionChangesVideoButNotTargetAudioAndStreamsRemainIndependent()
    {
        MiniMaxH3Config config = TinyConfig(timeDim: 2, curves: true);
        IBackend backend = new CpuBackend();
        using MiniMaxH3Transformer transformer = new MiniMaxH3Transformer(config);
        transformer.LoadWeights(BuildBaseWeights(config));
        Dictionary<string, Tensor> controlWeights = BuildControlWeights(config);
        MiniMaxH3FunControlConfig controlConfig = MiniMaxH3FunControlConfig.Detect(controlWeights);
        MiniMaxH3FunControlNet controlNet = new MiniMaxH3FunControlNet(controlConfig);
        controlNet.LoadWeights(controlWeights);
        int modelIndex = transformer.RegisterFunControlNet(controlNet);

        MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(1, 1, 2, 2, 1);
        using Tensor videoRows = Rand(1, config.VideoPatchDim);
        using Tensor audioRows = Rand(2, config.AudioLatentsDim);
        using Tensor text = Rand(1, config.TextDim);
        using Tensor controlRows = Rand(1, 196);
        (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(
            layout.PositionIds, MiniMaxH3Rope.DefaultInvFreq(config.RopeInvFreqLen), config.AttentionHeadDim);
        using (cos)
        using (sin)
        {
            float[] timesteps = [0.4f, 0.6f];
            IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf = TimestepRows();
            MiniMaxH3FunControlCondition zero = new MiniMaxH3FunControlCondition
            {
                ModelIndex = modelIndex,
                ControlRows = controlRows,
                Strength = 0f,
            };
            MiniMaxH3FunControlCondition one = zero with { Strength = 1f };

            (Tensor baseVideo, Tensor baseAudio) = transformer.Forward(
                backend, layout, videoRows, audioRows, text, cos, sin, timesteps, rowOf);
            (Tensor zeroVideo, Tensor zeroAudio) = transformer.Forward(
                backend, layout, videoRows, audioRows, text, cos, sin, timesteps, rowOf, controls: [zero]);
            (Tensor oneVideo, Tensor oneAudio) = transformer.Forward(
                backend, layout, videoRows, audioRows, text, cos, sin, timesteps, rowOf, controls: [one]);
            (Tensor twoVideo, Tensor twoAudio) = transformer.Forward(
                backend, layout, videoRows, audioRows, text, cos, sin, timesteps, rowOf, controls: [one, one]);
            using (baseVideo)
            using (baseAudio)
            using (zeroVideo)
            using (zeroAudio)
            using (oneVideo)
            using (oneAudio)
            using (twoVideo)
            using (twoAudio)
            {
                AssertTensorEqual(baseVideo, zeroVideo);
                AssertTensorEqual(baseAudio, zeroAudio);
                AssertTensorEqual(baseAudio, oneAudio);
                AssertTensorEqual(baseAudio, twoAudio);
                AssertTensorDifferent(baseVideo, oneVideo);
                AssertTensorDifferent(oneVideo, twoVideo);
            }
        }
    }

    [Theory]
    [InlineData(false, 6)]
    [InlineData(true, 2)]
    public void BranchRegistersAgainstMatchingFullOrPrunedTimeWidth(bool curves, int timeDim)
    {
        MiniMaxH3Config config = TinyConfig(timeDim, curves);
        using MiniMaxH3Transformer transformer = new MiniMaxH3Transformer(config);
        Dictionary<string, Tensor> controlWeights = BuildControlWeights(config);
        MiniMaxH3FunControlConfig controlConfig = MiniMaxH3FunControlConfig.Detect(controlWeights);
        MiniMaxH3FunControlNet controlNet = new MiniMaxH3FunControlNet(controlConfig);
        controlNet.LoadWeights(controlWeights);

        Assert.Equal(0, transformer.RegisterFunControlNet(controlNet));
        Assert.Equal([0, 10, 20, 30, 40], controlConfig.InjectionLayers);
    }

    private static MiniMaxH3Config TinyConfig(int timeDim, bool curves) => new MiniMaxH3Config
    {
        HiddenSize = 12,
        NumLayers = 41,
        TokenRefinerNumLayers = 1,
        NumAttentionHeads = 1,
        AttentionHeadDim = 12,
        FfnHiddenSize = 8,
        LatentsDim = 24,
        AudioLatentsDim = 4,
        TextDim = 6,
        TimestepInputDim = 4,
        TimeEmbedHiddenSize = 8,
        TimeEmbedDim = timeDim,
        RopeInvFreqLen = 2,
        AdalnCurveGrid = curves ? 3 : null,
    };

    private static Dictionary<string, Tensor> BuildBaseWeights(MiniMaxH3Config config)
    {
        int hidden = config.HiddenSize;
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["video_patch_proj.weight"] = Rand(hidden, config.VideoPatchDim),
            ["video_patch_proj.bias"] = Zero(hidden),
            ["audio_patch_proj.weight"] = Rand(hidden, config.AudioLatentsDim),
            ["audio_patch_proj.bias"] = Zero(hidden),
            ["condition_proj.weight"] = Rand(hidden, config.TextDim),
            ["condition_proj.bias"] = Zero(hidden),
            ["rope.inv_freq"] = Rand(config.RopeInvFreqLen),
            ["token_refiner.final_norm.weight"] = Ones(hidden),
            ["final_layer.norm.weight"] = Ones(hidden),
            ["final_layer.adaln_proj.linear.weight"] = Zero(hidden * 2, config.TimeEmbedDim),
            ["final_layer.adaln_proj.linear.bias"] = Zero(hidden * 2),
            ["final_layer.video_out.weight"] = Rand(config.VideoPatchDim, hidden),
            ["final_layer.video_out.bias"] = Zero(config.VideoPatchDim),
            ["final_layer.audio_out.weight"] = Rand(config.AudioLatentsDim, hidden),
            ["final_layer.audio_out.bias"] = Zero(config.AudioLatentsDim),
        };
        if (config.UseAdalnCurves)
        {
            weights["adaln_t_table"] = Rand(config.AdalnCurveGrid!.Value, config.TimeEmbedDim);
        }
        else
        {
            weights["time_embedder.proj_in.weight"] = Rand(config.TimeEmbedHiddenSize, config.TimestepInputDim);
            weights["time_embedder.proj_in.bias"] = Zero(config.TimeEmbedHiddenSize);
            weights["time_embedder.proj_out.weight"] = Rand(config.TimeEmbedDim, config.TimeEmbedHiddenSize);
            weights["time_embedder.proj_out.bias"] = Zero(config.TimeEmbedDim);
        }
        AddBaseBlock(weights, "token_refiner.blocks.0", config, inner, adaln: false);
        for (int index = 0; index < config.NumLayers; index++)
        {
            AddBaseBlock(weights, $"blocks.{index}", config, inner, adaln: true);
        }
        return weights;
    }

    private static Dictionary<string, Tensor> BuildControlWeights(MiniMaxH3Config config)
    {
        int hidden = config.HiddenSize;
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["control_proj_in.weight"] = Zero(hidden, 196),
            ["control_proj_in.bias"] = Zero(hidden),
        };
        for (int index = 0; index < 5; index++)
        {
            string prefix = $"control_blocks.{index}";
            AddBaseBlock(weights, prefix, config, inner, adaln: true);
            weights[prefix + ".after_proj.weight"] = Zero(hidden, hidden);
            weights[prefix + ".after_proj.bias"] = index == 4 ? Ramp(hidden, 0.05f) : Zero(hidden);
            if (index == 0)
            {
                weights[prefix + ".before_proj.weight"] = Zero(hidden, hidden);
                weights[prefix + ".before_proj.bias"] = Zero(hidden);
            }
        }
        return weights;
    }

    private static void AddBaseBlock(Dictionary<string, Tensor> weights, string prefix,
        MiniMaxH3Config config, int inner, bool adaln)
    {
        int hidden = config.HiddenSize;
        weights[prefix + ".norm1.weight"] = Ones(hidden);
        weights[prefix + ".norm2.weight"] = Ones(hidden);
        weights[prefix + ".attn.qkv_proj.weight"] = Zero(inner * 3, hidden);
        weights[prefix + ".attn.q_norm.weight"] = Ones(config.AttentionHeadDim);
        weights[prefix + ".attn.k_norm.weight"] = Ones(config.AttentionHeadDim);
        weights[prefix + ".attn.out_proj.weight"] = Zero(hidden, inner);
        weights[prefix + ".mlp.fc1.weight"] = Zero(config.FfnHiddenSize * 2, hidden);
        weights[prefix + ".mlp.fc2.weight"] = Zero(hidden, config.FfnHiddenSize);
        if (adaln)
        {
            weights[prefix + ".adaln_proj.linear.weight"] = Zero(hidden * 18, config.TimeEmbedDim);
            weights[prefix + ".adaln_proj.linear.bias"] = Zero(hidden * 18);
        }
    }

    private static IReadOnlyDictionary<MiniMaxH3SegmentKind, int> TimestepRows() =>
        new Dictionary<MiniMaxH3SegmentKind, int>
        {
            [MiniMaxH3SegmentKind.Text] = 0,
            [MiniMaxH3SegmentKind.Video] = 0,
            [MiniMaxH3SegmentKind.Cond] = 0,
            [MiniMaxH3SegmentKind.RefImage] = 0,
            [MiniMaxH3SegmentKind.Audio] = 1,
            [MiniMaxH3SegmentKind.CondAudio] = 1,
            [MiniMaxH3SegmentKind.RefAudio] = 1,
        };

    private static int _seed = 37;

    private static Tensor Rand(params long[] shape)
    {
        Tensor tensor = new Tensor(new TensorShape(shape), DType.F32);
        float* pointer = (float*)tensor.DataPointer;
        for (long index = 0; index < tensor.ElementCount; index++)
        {
            _seed = unchecked(_seed * 1103515245 + 12345);
            pointer[index] = ((_seed >> 16) & 0x7fff) / 32768f * 0.04f - 0.02f;
        }
        return tensor;
    }

    private static Tensor Zero(params long[] shape) => Fill(0f, shape);

    private static Tensor Ones(params long[] shape) => Fill(1f, shape);

    private static Tensor Fill(float value, params long[] shape)
    {
        Tensor tensor = new Tensor(new TensorShape(shape), DType.F32);
        float* pointer = (float*)tensor.DataPointer;
        for (long index = 0; index < tensor.ElementCount; index++)
        {
            pointer[index] = value;
        }
        return tensor;
    }

    private static Tensor Ramp(int length, float scale)
    {
        Tensor tensor = new Tensor(new TensorShape(length), DType.F32);
        float* pointer = (float*)tensor.DataPointer;
        for (int index = 0; index < length; index++)
        {
            pointer[index] = (index + 1) * scale;
        }
        return tensor;
    }

    private static Tensor RectIdentity(int rows, int columns)
    {
        Tensor tensor = Zero(rows, columns);
        float* pointer = (float*)tensor.DataPointer;
        for (int index = 0; index < Math.Min(rows, columns); index++)
        {
            pointer[index * columns + index] = 1f;
        }
        return tensor;
    }

    private static Tensor RowValues(int rows, int columns, float rowScale)
        => Values(new TensorShape(rows, 1, columns), rows, columns, rowScale);

    private static Tensor MatrixValues(int rows, int columns, float rowScale)
        => Values(new TensorShape(rows, columns), rows, columns, rowScale);

    private static Tensor Values(TensorShape shape, int rows, int columns, float rowScale)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        float* pointer = (float*)tensor.DataPointer;
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                pointer[row * columns + column] = rowScale * (row + 1) + (column + 1) * 0.001f;
            }
        }
        return tensor;
    }

    private static void Replace(Dictionary<string, Tensor> weights, string key, Tensor value)
    {
        weights[key].Dispose();
        weights[key] = value;
    }

    private static void AssertTensorEqual(Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        float* expectedPointer = (float*)expected.DataPointer;
        float* actualPointer = (float*)actual.DataPointer;
        for (long index = 0; index < expected.ElementCount; index++)
        {
            Assert.Equal(expectedPointer[index], actualPointer[index]);
        }
    }

    private static void AssertTensorDifferent(Tensor first, Tensor second)
    {
        float* firstPointer = (float*)first.DataPointer;
        float* secondPointer = (float*)second.DataPointer;
        bool differs = false;
        for (long index = 0; index < first.ElementCount; index++)
        {
            differs |= Math.Abs(firstPointer[index] - secondPointer[index]) > 1e-6f;
        }
        Assert.True(differs);
    }
}
