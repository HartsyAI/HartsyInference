using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.Video;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Pipelines.SanaWm;
using Xunit;

namespace HartsyInference.World.Tests.Pipelines;

/// <summary><see cref="SanaWmStreamingPipeline"/> and <see cref="SanaWmStream"/> with scripted components: stage order and
/// release, the self-forcing pass sequence per chunk (four denoise passes and a t=0 save pass, frame 0 of the first chunk held
/// clean), the upstream chunk plan (4 latent frames, then 3), live actions steering the camera across chunks, guidance doubling
/// the passes, cancellation before commit, and a fixed camera path refusing live actions.</summary>
public sealed class SanaWmStreamTests
{
    [Fact]
    public void LiveUnbounded_GeneratesChunksAndFollowsActions()
    {
        Components components = new();
        using SanaWmStream stream = new SanaWmStreamingPipeline(components).Open(Request(null, opening: "w-24"), CancellationToken.None);
        Assert.Equal(["open:text", "release:text", "open:image", "release:image", "open:dit", "open:decoder"], components.Events);
        Assert.Equal(TokenWindowPolicy.RecentChunks(2, true), components.Window);
        Assert.Equal(25, stream.NextChunkFrames);

        IReadOnlyList<VideoFrame> first = stream.GenerateChunk(CancellationToken.None);
        Assert.Equal(25, first.Count);
        Assert.Equal(0, first[0].Index);
        Assert.Equal(SanaWmPipeline.Width * SanaWmPipeline.Height * 3, first[0].Rgb.Length);
        Assert.Equal(159, first[0].Rgb[0]);   // decoded 0.25 maps to 127.5 * 0.25 + 127.5
        Assert.Equal(5, components.Calls.Count);
        Assert.Equal([0f, 1000f, 1000f, 1000f], components.Calls[0].Timesteps);
        Assert.Equal([0f, 960f, 960f, 960f], components.Calls[1].Timesteps);
        Assert.Equal([0f, 889f, 889f, 889f], components.Calls[2].Timesteps);
        Assert.Equal([0f, 727f, 727f, 727f], components.Calls[3].Timesteps);
        Assert.Equal([0f, 0f, 0f, 0f], components.Calls[4].Timesteps);
        Assert.All(components.Calls.Take(4), call => Assert.False(call.Commit));
        Assert.True(components.Calls[4].Commit);
        Assert.Equal(0f, components.Calls[0].FirstPoseTz);
        Assert.True(components.Calls[0].LastPoseTz > 0.5f);   // 24 frames driving forward at up to 0.025 a frame

        stream.SendAction("none");
        float before = components.Calls[^1].LastPoseTz;
        IReadOnlyList<VideoFrame> second = stream.GenerateChunk(CancellationToken.None);
        Assert.Equal(24, second.Count);
        Assert.Equal(25, second[0].Index);
        Assert.Equal([1000f, 1000f, 1000f], components.Calls[5].Timesteps);
        // Released keys coast to rest: still moving forward, but well short of the 0.6 that 24 frames at full speed would cover.
        float coasted = components.Calls[5].LastPoseTz - before;
        Assert.InRange(coasted, 0.05f, 0.45f);

        stream.SendAction("s");
        stream.GenerateChunk(CancellationToken.None);
        Assert.True(components.Calls[^1].LastPoseTz < components.Calls[^1].FirstPoseTz);   // reversing
        Assert.Equal([4, 3, 3], components.Commits);
        Assert.Equal(73, stream.FramesGenerated);
        Assert.False(stream.Completed);
        Assert.Throws<FormatException>(() => stream.SendAction("zz"));
    }

    [Fact]
    public void Bounded_Guided_CompletesAndReleases()
    {
        Components components = new();
        SanaWmStream stream = new SanaWmStreamingPipeline(components).Open(Request(49, guidance: 2.0), CancellationToken.None);
        int total = 0;
        while (!stream.Completed)
            total += stream.GenerateChunk(CancellationToken.None).Count;

        Assert.Equal(49, total);
        Assert.Equal(20, components.Calls.Count);   // 2 chunks x 5 passes x 2 branches
        Assert.Equal(10, components.Calls.Count(call => call.Branch == SanaWmGuidanceBranch.Conditional));
        Assert.Throws<InvalidOperationException>(() => stream.GenerateChunk(CancellationToken.None));
        stream.Dispose();
        Assert.Equal(["release:decoder", "release:dit"], components.Events.TakeLast(2));
    }

    [Fact]
    public void Cancel_BeforeCommit_DiscardsAndCanRetry()
    {
        Components components = new();
        using SanaWmStream stream = new SanaWmStreamingPipeline(components).Open(Request(null), CancellationToken.None);
        using CancellationTokenSource stop = new();
        stop.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => stream.GenerateChunk(stop.Token));
        Assert.Equal(0, stream.ChunksGenerated);
        Assert.Empty(components.Commits);
        Assert.Equal(25, stream.GenerateChunk(CancellationToken.None).Count);
        Assert.Equal([4], components.Commits);
    }

    [Fact]
    public void FixedPath_RefusesLiveActions()
    {
        Components components = new();
        SanaWmStreamRequest request = Request(49) with { CameraToWorld = SanaWmAction.ToCameraToWorld("w-48") };
        using SanaWmStream stream = new SanaWmStreamingPipeline(components).Open(request, CancellationToken.None);
        Assert.Throws<InvalidOperationException>(() => stream.SendAction("w"));
        Assert.Equal(25, stream.GenerateChunk(CancellationToken.None).Count);
        Assert.Equal(24, stream.GenerateChunk(CancellationToken.None).Count);
        Assert.True(stream.Completed);
    }

    private static SanaWmStreamRequest Request(int? frames, double guidance = 1.0, string? opening = null) => new()
    {
        ImageRgb = new byte[64 * 36 * 3],
        ImageWidth = 64,
        ImageHeight = 36,
        Prompt = "a road",
        Intrinsics = [new PinholeIntrinsics(40f, 40f, 32f, 18f)],
        Frames = frames,
        Guidance = guidance,
        Seed = 7,
        InitialActions = opening,
    };

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32, DeviceKind.Cpu);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }

    private sealed record Call(SanaWmGuidanceBranch Branch, float[] Timesteps, bool Commit, int Frames, float FirstPoseTz, float LastPoseTz);

    /// <summary>Scripted stage components that log every open, release, denoiser call and commit.</summary>
    private sealed class Components : ISanaWmStreamingComponents
    {
        public List<string> Events { get; } = [];
        public List<Call> Calls { get; } = [];
        public List<int> Commits { get; } = [];
        public TokenWindowPolicy? Window { get; private set; }

        public SanaWmStage<ISanaWmTextEncoder> OpenTextEncoder() => Open<ISanaWmTextEncoder>("text", new TextEncoder());

        public SanaWmStage<ISanaWmImageEncoder> OpenImageEncoder() => Open<ISanaWmImageEncoder>("image", new ImageEncoder());

        public SanaWmStage<ISanaWmChunkDenoiser> OpenChunkDenoiser(SanaWmTextConditioning text, TokenWindowPolicy softmaxWindow)
        {
            Window = softmaxWindow;
            return Open<ISanaWmChunkDenoiser>("dit", new Denoiser(this));
        }

        public SanaWmStage<IStreamingVideoDecoder> OpenStreamingDecoder() => Open<IStreamingVideoDecoder>("decoder", new Decoder());

        private SanaWmStage<T> Open<T>(string name, T component)
            where T : class
        {
            Events.Add($"open:{name}");
            return new SanaWmStage<T>(component, () => Events.Add($"release:{name}"));
        }

        private sealed class TextEncoder : ISanaWmTextEncoder
        {
            public SanaWmTextConditioning Encode(string prompt, string negativePrompt) => new(
                Filled(new TensorShape([1, 1, 300, 8]), 0.1f), Filled(new TensorShape([1, 300]), 1f),
                Filled(new TensorShape([1, 1, 300, 8]), 0.2f), Filled(new TensorShape([1, 300]), 1f));
        }

        private sealed class ImageEncoder : ISanaWmImageEncoder
        {
            public Tensor Encode(Tensor video) => Filled(new TensorShape([1, SanaWmPipeline.LatentChannels, 1, 22, 40]), 0.3f);
        }

        private sealed class Denoiser(Components owner) : ISanaWmChunkDenoiser
        {
            public int CommittedFrames { get; private set; }

            public Tensor Predict(Tensor latents, ReadOnlySpan<float> timesteps, SanaWmGuidanceBranch branch,
                ReadOnlySpan<float> cameraConditions, Tensor chunkPlucker, bool commit)
            {
                int frames = (int)latents.Shape[2];
                Assert.Equal(frames * 20, cameraConditions.Length);
                Assert.Equal(new TensorShape([1, 48, frames, 22, 40]), chunkPlucker.Shape);
                owner.Calls.Add(new Call(branch, timesteps.ToArray(), commit, frames, cameraConditions[11],
                    cameraConditions[(frames - 1) * 20 + 11]));
                return Filled(latents.Shape, branch == SanaWmGuidanceBranch.Conditional ? 1f : 0.5f);
            }

            public void Commit(int latentFrames)
            {
                owner.Commits.Add(latentFrames);
                CommittedFrames += latentFrames;
            }

            public void Discard() => owner.Events.Add("discard");
        }

        private sealed class Decoder : IStreamingVideoDecoder
        {
            public int TemporalStride => 8;
            public int DecodedLatentFrames { get; private set; }

            public Tensor DecodeNext(Tensor latent)
            {
                int frames = PixelFrames((int)latent.Shape[2]);
                DecodedLatentFrames += (int)latent.Shape[2];
                return Filled(new TensorShape([1, 3, frames, SanaWmPipeline.Height, SanaWmPipeline.Width]), 0.25f);
            }

            public int PixelFrames(int latentFrames) => DecodedLatentFrames == 0 ? (latentFrames - 1) * 8 + 1 : latentFrames * 8;

            public void Reset() => DecodedLatentFrames = 0;

            public void Dispose()
            {
            }
        }
    }
}
