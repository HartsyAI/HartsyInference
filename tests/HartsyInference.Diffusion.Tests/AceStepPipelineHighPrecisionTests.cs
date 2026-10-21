using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>ACE-Step v1 generates with high-precision GEMM (bf16 weights against F32 activations NaN the latent otherwise) and
/// hands the backend back as it found it, also when the call throws.</summary>
public sealed class AceStepPipelineHighPrecisionTests
{
    public class FlagProbe : DispatchProxy
    {
        public IBackend Inner { get; set; } = null!;
        public bool Flag { get; set; }
        public List<bool> Sets { get; } = [];

        protected override object? Invoke(MethodInfo? method, object?[]? args)
        {
            switch (method!.Name)
            {
                case "get_HighPrecisionGemm":
                    return Flag;
                case "set_HighPrecisionGemm":
                    Flag = (bool)args![0]!;
                    Sets.Add(Flag);
                    return null;
                default:
                    return method.Invoke(Inner, args);
            }
        }
    }

    [Fact]
    public void Generate_TurnsHighPrecisionOn_AndRestoresItWhenTheCallThrows()
    {
        using CpuBackend cpu = new();
        IBackend backend = DispatchProxy.Create<IBackend, FlagProbe>();
        FlagProbe probe = (FlagProbe)(object)backend;
        probe.Inner = cpu;
        AceStepConfig config = AceStepSyntheticWeights.TinyConfig;
        AceStepPipeline pipeline = new(backend, new AceStepDit(config), new MusicDcaeDecoder(), new AdaMosHiFiGanV1(), config);
        using Tensor text = new(new TensorShape(1, 3, config.TextDim), DType.F32);

        Assert.Throws<ArgumentOutOfRangeException>(() => pipeline.Generate(text, [], durationSeconds: 0.5));

        Assert.Equal([true, false], probe.Sets);
        Assert.False(probe.Flag);
    }
}
