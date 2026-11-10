using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Pipelines.SanaWm;
using Xunit;

namespace HartsyInference.World.Tests.Pipelines;

/// <summary>Scripted SANA-WM components. Every stage open and release is logged in order, and the denoiser records each call.</summary>
internal sealed class SanaWmFakeComponents : ISanaWmComponents
{
    public const float FirstLatentValue = 0.3f;
    public const float UnconditionalVelocity = 1f;
    public const float ConditionalVelocity = 3f;

    private readonly List<string> _events = [];

    /// <summary>Stage events in order: <c>open:text</c>, <c>release:text</c>, and so on.</summary>
    public IReadOnlyList<string> Events => _events;

    /// <summary>The denoiser calls, in order.</summary>
    public List<DenoiserCall> Calls { get; } = [];

    /// <summary>Runs before each denoiser call with the call's zero-based index.</summary>
    public Action<int>? BeforePredict { get; init; }

    /// <summary>Whether the denoiser accepts a single guidance branch.</summary>
    public bool SingleBranch { get; init; } = true;

    public SanaWmStage<ISanaWmTextEncoder> OpenTextEncoder() => Open<ISanaWmTextEncoder>("text", new TextEncoder());

    public SanaWmStage<ISanaWmImageEncoder> OpenImageEncoder() => Open<ISanaWmImageEncoder>("image", new ImageEncoder());

    public SanaWmStage<ISanaWmDenoiser> OpenDenoiser(SanaWmTextConditioning text, SanaWmCameraConditioning camera) =>
        Open<ISanaWmDenoiser>("dit", new Denoiser(this));

    public SanaWmStage<ISanaWmVideoDecoder> OpenVideoDecoder() => Open<ISanaWmVideoDecoder>("decoder", new VideoDecoder());

    private SanaWmStage<T> Open<T>(string name, T component)
        where T : class
    {
        _events.Add($"open:{name}");
        return new SanaWmStage<T>(component, () => _events.Add($"release:{name}"));
    }

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32, DeviceKind.Cpu);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }

    /// <summary>One denoiser call: its batch layout and the timesteps it was given.</summary>
    public sealed record DenoiserCall(SanaWmGuidanceBranch[] Branches, float[] Timesteps);

    private sealed class TextEncoder : ISanaWmTextEncoder
    {
        public SanaWmTextConditioning Encode(string prompt, string negativePrompt) => new(
            Filled(new TensorShape([1, 1, 300, 8]), 0.1f), Filled(new TensorShape([1, 300]), 1f),
            Filled(new TensorShape([1, 1, 300, 8]), 0.2f), Filled(new TensorShape([1, 300]), 1f));
    }

    private sealed class ImageEncoder : ISanaWmImageEncoder
    {
        public Tensor Encode(Tensor video)
        {
            Assert.Equal(new TensorShape([1, 3, 1, SanaWmPipeline.Height, SanaWmPipeline.Width]), video.Shape);
            return Filled(new TensorShape([1, SanaWmPipeline.LatentChannels, 1, 22, 40]), FirstLatentValue);
        }
    }

    private sealed class Denoiser(SanaWmFakeComponents owner) : ISanaWmDenoiser
    {
        public bool SupportsSingleBranch => owner.SingleBranch;

        public Tensor Predict(Tensor latents, ReadOnlySpan<float> timesteps, ReadOnlySpan<SanaWmGuidanceBranch> branches)
        {
            owner.BeforePredict?.Invoke(owner.Calls.Count);
            owner.Calls.Add(new DenoiserCall(branches.ToArray(), timesteps.ToArray()));
            Tensor velocity = new(latents.Shape, DType.F32, DeviceKind.Cpu);
            Span<float> values = velocity.AsSpan<float>();
            int item = values.Length / branches.Length;
            for (int b = 0; b < branches.Length; b++)
            {
                float value = branches[b] == SanaWmGuidanceBranch.Conditional ? ConditionalVelocity : UnconditionalVelocity;
                values.Slice(b * item, item).Fill(value);
            }
            return velocity;
        }
    }

    private sealed class VideoDecoder : ISanaWmVideoDecoder
    {
        public Tensor Decode(Tensor latent)
        {
            long frames = 8 * (latent.Shape[2] - 1) + 1;
            return Filled(new TensorShape([1, 3, frames, SanaWmPipeline.Height, SanaWmPipeline.Width]), 0f);
        }
    }
}
