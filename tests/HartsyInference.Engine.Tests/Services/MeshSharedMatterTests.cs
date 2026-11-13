using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshSharedMatterTests
{
    private sealed class FakeMatter : ISharedMatter
    {
        public string? LoadedFrom { get; private set; }

        public int Calls { get; private set; }

        public int Disposals { get; private set; }

        public bool IsAvailable => true;

        public float[] Alpha(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height)
        {
            Calls++;
            LoadedFrom = Path.Combine(RepoPaths.ModelsRoot(), "Vision", "BiRefNet-matting", "model.safetensors");
            float[] alpha = new float[width * height];
            Array.Fill(alpha, 1f);
            return alpha;
        }

        public void Dispose()
        {
            Disposals++;
            LoadedFrom = null;
        }
    }

    private static ImageData Plain() => new() { Rgb = new byte[4 * 4 * 3], Width = 4, Height = 4 };

    private static ImageData Cutout() => Plain() with { Alpha = [.. Enumerable.Range(0, 16).Select(i => i < 8 ? (byte)0 : (byte)255)] };

    private static (InferenceEngine Engine, MeshPreparationService Service, List<FakeMatter> Created) EngineWithFakeMatter()
    {
        InferenceEngine engine = new("cpu");
        List<FakeMatter> created = [];
        MeshPreparationService service = new(engine, () =>
        {
            FakeMatter matter = new();
            created.Add(matter);
            return matter;
        });
        FieldInfo field = typeof(InferenceEngine).GetField("_meshPreparation", BindingFlags.NonPublic | BindingFlags.Instance)!;
        field.SetValue(engine, new Lazy<MeshPreparationService>(() => service));
        return (engine, service, created);
    }

    [Fact]
    public void EveryCallerOfTheSharedMatterGetsTheSameOneLoadedOnce()
    {
        (InferenceEngine engine, MeshPreparationService service, List<FakeMatter> created) = EngineWithFakeMatter();
        using (engine)
        {
            IForegroundMatter pipelineSide = service.SharedMatter;

            service.Prepare("triposr", Plain(), CancellationToken.None);
            pipelineSide.Alpha(engine.Backend, new byte[4 * 4 * 3], 4, 4);

            Assert.Same(service.SharedMatter, service.SharedMatter);
            FakeMatter only = Assert.Single(created);
            Assert.Equal(2, only.Calls);
        }
    }

    [Fact]
    public void ReleasingDisposesTheMatterOnceAndTheNextMatteLoadsAgain()
    {
        (InferenceEngine engine, MeshPreparationService service, List<FakeMatter> created) = EngineWithFakeMatter();
        using (engine)
        {
            service.SharedMatter.Alpha(engine.Backend, new byte[4 * 4 * 3], 4, 4);

            Assert.True(service.ReleaseMatter());
            Assert.False(service.ReleaseMatter());
            service.SharedMatter.Alpha(engine.Backend, new byte[4 * 4 * 3], 4, 4);

            Assert.Equal(2, created.Count);
            Assert.Equal([1, 0], created.Select(m => m.Disposals));
        }
    }

    [Fact]
    public void AMatterThatWasOnlyProbedIsDroppedButNotReportedAsFreed()
    {
        (InferenceEngine engine, MeshPreparationService service, List<FakeMatter> created) = EngineWithFakeMatter();
        using (engine)
        {
            Assert.True(service.SharedMatter.IsAvailable);

            Assert.False(service.ReleaseMatter());
            Assert.Equal(1, Assert.Single(created).Disposals);
        }
    }

    [Fact]
    public void TheResidentMatterIsListedAndFreedByNameOrWithEverythingElse()
    {
        (InferenceEngine engine, MeshPreparationService service, List<FakeMatter> created) = EngineWithFakeMatter();
        using (engine)
        {
            Assert.Empty(engine.MeshLifecycle.Loaded());
            service.SharedMatter.Alpha(engine.Backend, new byte[4 * 4 * 3], 4, 4);

            MeshLoadedModel listed = Assert.Single(engine.MeshLifecycle.Loaded());
            Assert.Equal(("matting", "Vision/BiRefNet-matting/model.safetensors"), (listed.Kind, listed.Name));
            Assert.Equal(0, engine.MeshLifecycle.Free("some-other-model"));
            Assert.Equal(0, created[0].Disposals);
            Assert.Equal(1, engine.MeshLifecycle.Free("vision/birefnet-matting/MODEL.safetensors"));
            Assert.Empty(engine.MeshLifecycle.Loaded());

            service.SharedMatter.Alpha(engine.Backend, new byte[4 * 4 * 3], 4, 4);
            Assert.Equal(1, engine.MeshLifecycle.Free());
            Assert.Equal([1, 1], created.Select(m => m.Disposals));
        }
    }

    [Fact]
    public void FreeingTheModelsLeavesNoMatterBehindEvenWhenNothingElseIsResident()
    {
        (InferenceEngine engine, MeshPreparationService service, _) = EngineWithFakeMatter();
        using (engine)
        {
            service.SharedMatter.Alpha(engine.Backend, new byte[4 * 4 * 3], 4, 4);

            Assert.Equal(1, engine.MeshLifecycle.Free());
            Assert.Null(service.MatterLoadedFrom);
        }
    }

    [Fact]
    public void TheMattingModelIsCountedInThePeakOfARequestThatMattesItsImage()
    {
        long recorded = MeshMemoryProfile.Estimate("pixal3d", new MeshRequest { Image = Cutout() })!.PeakResidentBytes;

        long withMatte = MeshMemoryProfile.Estimate("pixal3d", new MeshRequest { Image = Plain() })!.PeakResidentBytes;

        Assert.Equal(recorded + MeshMemoryProfile.MatterWeightBytes, withMatte);
        Assert.Equal(recorded, MeshMemoryProfile.Estimate("pixal3d", new MeshRequest())!.PeakResidentBytes);
    }

    [Fact]
    public void AModelThatNeedsAskingForPreprocessingCountsTheMatterOnlyUnderAuto()
    {
        long none = MeshMemoryProfile.Estimate("hunyuan3d-paint21", new MeshRequest { Image = Plain() })!.PeakResidentBytes;
        long auto = MeshMemoryProfile.Estimate("hunyuan3d-paint21", new MeshRequest { Image = Plain(), Preprocess = MeshPreprocess.Auto })!.PeakResidentBytes;
        long autoWithCutout = MeshMemoryProfile.Estimate("hunyuan3d-paint21", new MeshRequest { Image = Cutout(), Preprocess = MeshPreprocess.Auto })!.PeakResidentBytes;

        Assert.Equal(none + MeshMemoryProfile.MatterWeightBytes, auto);
        Assert.Equal(none, autoWithCutout);
    }

    [Fact]
    public void AnOpaqueAlphaPlaneIsMattedSoItsPeakCountsTheMatterToo()
    {
        ImageData opaque = Plain() with { Alpha = [.. Enumerable.Repeat((byte)255, 16)] };

        Assert.True(MeshMemoryProfile.Mattes("pixal3d", new MeshRequest { Image = opaque }));
        Assert.False(MeshMemoryProfile.Mattes("pixal3d", new MeshRequest { Image = Cutout() }));
    }
}
