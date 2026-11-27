using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat paths on CUDA and Vulkan against the CPU backend, on the tiny config: the same weights and inputs, and the same final
/// latent within <see cref="Bound"/>. Covered: the uncached avatar modes, the row-tiled self-attention projection and FFN, the offloaded
/// key and value cache, the gated distillation LoRA and the two-LoRA mix, the multi-speaker loops (uncached, cached and offloaded cache), and
/// (Vulkan only; CUDA has <see cref="LongCatCudaParityTests"/>) the cached video and avatar loops. CUDA runs under <see cref="ExactGemmScope"/>.
/// Skipped where the device does not open.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class LongCatDevicePathsGpuTests(ITestOutputHelper output)
{
    private const int Frames = 5, Height = 4, Width = 4, TextLength = 6, Steps = 3, DistillSteps = 8;
    // Smaller than any run's token count, and not a divisor of it, so every tiled projection has several tiles and a short last one.
    private const int TileRows = 3;
    private const float Bound = 1e-5f;
    private static readonly int Hw = Height * Width;

    private enum Device { Cuda, Vulkan }

    [CudaDeviceFact]
    public void Cuda_UncachedAvatarModes_MatchTheCpu() => Uncached(Device.Cuda);

    [VulkanDeviceFact]
    public void Vulkan_UncachedAvatarModes_MatchTheCpu() => Uncached(Device.Vulkan);

    [CudaDeviceFact]
    public void Cuda_TiledProjectionsAndFeedForward_MatchTheUntiledCpu() => Tiled(Device.Cuda);

    [VulkanDeviceFact]
    public void Vulkan_TiledProjectionsAndFeedForward_MatchTheUntiledCpu() => Tiled(Device.Vulkan);

    [CudaDeviceFact]
    public void Cuda_OffloadedKvCache_MatchesTheCpuCachedRun() => Offloaded(Device.Cuda);

    [VulkanDeviceFact]
    public void Vulkan_OffloadedKvCache_MatchesTheCpuCachedRun() => Offloaded(Device.Vulkan);

    [CudaDeviceFact]
    public void Cuda_GatedDistillationLoras_MatchTheCpu() => GatedLoras(Device.Cuda);

    [VulkanDeviceFact]
    public void Vulkan_GatedDistillationLoras_MatchTheCpu() => GatedLoras(Device.Vulkan);

    [CudaDeviceFact]
    public void Cuda_MultiSpeakerLoops_MatchTheCpu() => MultiSpeaker(Device.Cuda);

    [VulkanDeviceFact]
    public void Vulkan_MultiSpeakerLoops_MatchTheCpu() => MultiSpeaker(Device.Vulkan);

    [VulkanDeviceFact]
    public void Vulkan_CachedVideoAndAvatarLoops_MatchTheCpu()
    {
        WithWeights(LongCatTestData.Tiny, avatar: false, (config, weights) =>
            Compare(Device.Vulkan, "video loop", backend => RunVideo(backend, config, weights)));
        WithWeights(LongCatTestData.TinyAvatar, avatar: true, (config, weights) =>
            Compare(Device.Vulkan, "avatar cached", backend => RunAvatar(backend, config, weights, LongCatAvatarMode.Avc, cached: true)));
    }

    private void Uncached(Device device) => WithWeights(LongCatTestData.TinyAvatar, avatar: true, (config, weights) =>
    {
        foreach (LongCatAvatarMode mode in new[] { LongCatAvatarMode.At2v, LongCatAvatarMode.Ai2v, LongCatAvatarMode.Avc })
            Compare(device, $"uncached {mode}", backend => RunAvatar(backend, config, weights, mode, cached: false));
    });

    private void Tiled(Device device) => WithWeights(LongCatTestData.TinyAvatar, avatar: true, (config, weights) =>
    {
        foreach ((LongCatAvatarMode mode, bool cached) in new[] { (LongCatAvatarMode.Ai2v, false), (LongCatAvatarMode.Avc, true) })
            Compare(device, $"tiled {mode}{(cached ? " cached" : "")}",
                reference: backend => RunAvatar(backend, config, weights, mode, cached),
                onDevice: backend => RunAvatar(backend, config, weights, mode, cached, tileRows: TileRows));
    });

    private void Offloaded(Device device) => WithWeights(LongCatTestData.TinyAvatar, avatar: true, (config, weights) =>
        Compare(device, "offloaded cache",
            reference: backend => RunAvatar(backend, config, weights, LongCatAvatarMode.Avc, cached: true),
            onDevice: backend => RunAvatar(backend, config, weights, LongCatAvatarMode.Avc, cached: true, offload: true)));

    private void MultiSpeaker(Device device) => WithWeights(LongCatTestData.TinyAvatar, avatar: true, (config, weights) =>
    {
        // Two speakers and a silent track over the 2 x 2 token grid: one region each, and one silent person.
        LongCatSpeakers speakers = new(3, 4, Height / 2, Width / 2, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
        foreach ((LongCatAvatarMode mode, bool cached, bool offload) in new[]
        {
            (LongCatAvatarMode.Ai2v, false, false), (LongCatAvatarMode.Avc, false, false), (LongCatAvatarMode.Avc, true, false),
            (LongCatAvatarMode.Avc, true, true),
        })
        {
            // The reference side holds the cache on the CPU either way; the device side stages it block by block when offloaded.
            Compare(device, $"speakers {mode}{(cached ? offload ? " offloaded" : " cached" : "")}",
                reference: backend => RunAvatar(backend, config, weights, mode, cached, speakers: speakers),
                onDevice: backend => RunAvatar(backend, config, weights, mode, cached, offload: offload, speakers: speakers));
        }
    });

    private void GatedLoras(Device device) => WithWeights(LongCatTestData.TinyAvatar, avatar: true, (config, weights) =>
    {
        string directory = Path.Combine(Path.GetTempPath(), "longcat-device-lora-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            using LongCatLoraFixture dmd = new(Path.Combine(directory, "dmd.safetensors"), config, 17);
            using LongCatLoraFixture opsd = new(Path.Combine(directory, "opsd.safetensors"), config, 19);
            foreach (bool mixed in new[] { false, true })
            {
                string label = mixed ? "two-LoRA mix" : "gated DMD";
                Compare(device, label, backend => RunDistilled(backend, config, weights, dmd.Path, mixed ? opsd.Path : null));
                // The adapters reach the result: the same run with nothing attached to the gates differs.
                using CpuBackend cpu = new();
                float[] with = RunDistilled(cpu, config, weights, dmd.Path, mixed ? opsd.Path : null);
                float[] without = RunDistilled(cpu, config, weights, null, null, mixed);
                Assert.True(Worst(with, without) > 1e-3f, $"{label}: the LoRA did not change the result");
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    });

    private static void WithWeights(LongCatConfig config, bool avatar, Action<LongCatConfig, Dictionary<string, Tensor>> body)
    {
        Dictionary<string, Tensor> weights = avatar ? LongCatTestData.AvatarWeights(config, 113) : LongCatTestData.Weights(config, 111);
        try
        {
            body(config, weights);
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private void Compare(Device device, string label, Func<IBackend, float[]> run) => Compare(device, label, run, run);

    private void Compare(Device device, string label, Func<IBackend, float[]> reference, Func<IBackend, float[]> onDevice)
    {
        float[] expected;
        using (CpuBackend cpu = new())
            expected = reference(cpu);
        using IBackend backend = device == Device.Cuda ? GpuDevices.OpenCuda() : GpuDevices.OpenVulkan();
        using ExactGemmScope exact = new(backend);
        float[] actual = onDevice(backend);

        float worst = Worst(expected, actual);
        output.WriteLine($"{device} {label}: worst absolute difference {worst:E3} (bound {Bound:E0})");
        Assert.True(worst <= Bound, $"{device} {label}: drifted from the CPU by {worst:E3}, above {Bound:E0}");
    }

    /// <summary>The largest absolute difference; NaN when either side holds one.</summary>
    private static float Worst(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        return worst;
    }

    private static float[] RunVideo(IBackend backend, LongCatConfig config, Dictionary<string, Tensor> weights)
    {
        Random rng = new(3);
        LongCatPrompt positive = Prompt(rng, config), negative = Prompt(rng, config);
        float[] noise = LongCatTestData.RandomValues(rng, config.InChannels * Frames * Hw);
        using LongCatTransformer transformer = new(config);
        transformer.LoadWeights(weights);
        LongCatRun run = new(config.InChannels, Frames, Height, Width, noise, positive, negative, Steps, Distill: false, Shift: 3f, Guidance: 4f);
        return new LongCatVideoPipeline(backend, transformer).Denoise(run).Latents;
    }

    /// <summary>One avatar segment: the three-pass guidance on the linear schedule, or the distilled 8-step path when gates are given.</summary>
    private static float[] RunAvatar(IBackend backend, LongCatConfig config, Dictionary<string, Tensor> weights, LongCatAvatarMode mode, bool cached,
        bool offload = false, int? tileRows = null, AdjunctGate? gate = null, LongCatAvatarLoraSchedule? mix = null,
        LongCatSpeakers? speakers = null)
    {
        int cond = mode switch { LongCatAvatarMode.At2v => 0, LongCatAvatarMode.Ai2v => 1, _ => 2 };
        bool distilled = gate is not null || mix is not null;
        Random rng = new(5);
        LongCatPrompt positive = Prompt(rng, config), negative = Prompt(rng, config);
        float[] noise = LongCatTestData.RandomValues(rng, config.InChannels * Frames * Hw);
        float[]? condLatents = cond == 0 ? null : LongCatTestData.RandomValues(rng, config.InChannels * cond * Hw);
        float[] audio = LongCatTestData.RandomValues(rng, Frames * (speakers?.Tracks ?? 1) * config.AudioContextTokens * config.AudioOutputDim);
        float[] unconditionalAudio = LongCatTestData.RandomValues(rng, audio.Length);
        using LongCatTransformer transformer = new(config);
        transformer.LoadWeights(weights);
        if (tileRows is int rows)
        {
            for (int b = 0; b < transformer.BlockCount; b++)
            {
                LongCatBlock block = (LongCatBlock)transformer.GetBlock(b);
                block.SetQkvTileRows(rows);
                block.SetFfnTileRows(rows);
            }
        }
        LongCatAvatarRun run = new(config.InChannels, Frames, Height, Width, noise, positive, negative, audio,
            distilled ? null : unconditionalAudio, distilled ? DistillSteps : Steps,
            distilled ? LongCatAvatarSchedule.Kind.Distill8 : LongCatAvatarSchedule.Kind.Linear, 7f, distilled ? 1f : 2f, distilled ? 1f : 1.5f,
            mode, cond, condLatents, UseKvCache: cached, LoraGate: gate, OffloadKv: offload, LoraMix: mix, Speakers: speakers);
        return new LongCatAvatarPipeline(backend, transformer).Denoise(run).Latents;
    }

    /// <summary>The distilled run with its LoRA files attached as gated adapters: the DMD gate alone, or the two-LoRA mix when
    /// <paramref name="opsdPath"/> is given or <paramref name="mixedWithoutFiles"/> is set. A null path attaches nothing to that gate.</summary>
    private static float[] RunDistilled(IBackend backend, LongCatConfig config, Dictionary<string, Tensor> weights, string? dmdPath,
        string? opsdPath, bool mixedWithoutFiles = false)
    {
        AdjunctGate dmdGate = new(), opsdGate = new();
        using LoraFile? dmd = dmdPath is null ? null : LoraFile.Load(dmdPath);
        using LoraFile? opsd = opsdPath is null ? null : LoraFile.Load(opsdPath);
        using LoraStack stack = new();
        Dictionary<string, Tensor> patched = new(weights, StringComparer.Ordinal);
        if (dmd is not null)
            stack.AddGated(dmd, dmdGate);
        if (opsd is not null)
            stack.AddGated(opsd, opsdGate);
        if (dmd is not null || opsd is not null)
            stack.ApplyTo(patched, LoraTarget.Transformer, backend);
        bool mixed = opsdPath is not null || mixedWithoutFiles;
        LongCatAvatarLoraSchedule? mix = mixed ? new LongCatAvatarLoraSchedule(dmdGate, opsdGate, Alpha: 0.5f, MixSteps: 4) : null;
        return RunAvatar(backend, config, patched, LongCatAvatarMode.Ai2v, cached: false, gate: mixed ? null : dmdGate, mix: mix);
    }

    private static LongCatPrompt Prompt(Random rng, LongCatConfig config) =>
        new(LongCatTestData.RandomValues(rng, TextLength * config.CaptionChannels), TextLength, TextLength);
}
