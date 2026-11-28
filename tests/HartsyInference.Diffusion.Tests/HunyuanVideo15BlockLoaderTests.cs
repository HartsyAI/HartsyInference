using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The streamed block loader must give the resident load's forward bit for bit: with the same converter and the same cast, the velocity from
/// blocks loaded as the forward reaches them equals the velocity from the whole DiT loaded at once. Uses the tiny synthetic DiT fixture on the CPU.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class HunyuanVideo15BlockLoaderTests
{
    private const int Blocks = 2;
    private const int Hidden = 32;

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = 4, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private static Dictionary<string, Tensor> Source(SafeTensorsLoader loader) => loader.GetAllTensors()
        .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
            && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
        .ToDictionary(pair => pair.Key, pair => pair.Value);

    private static Tensor RandomTensor(Random rng, params long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        foreach (ref float v in tensor.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return tensor;
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void StreamedBlocks_GiveTheResidentVelocityBitForBit(bool castToF16, bool readFromFile)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        Dictionary<string, Tensor> source = Source(loader);
        List<DType> castSources = castToF16 ? [DType.F32] : [];

        // Resident: the whole DiT converted, cast and loaded at once.
        Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(source, Blocks, Hidden);
        if (castToF16)
        {
            foreach (string key in converted.Keys.ToList())
            {
                if (converted[key].DType == DType.F32) converted[key] = converted[key].CastTo(DType.F16);
            }
        }
        using HunyuanVideo15Dit resident = new(config);
        resident.LoadWeights(converted);

        // Streamed: the shared weights once, then each block as the forward reaches it, released after it.
        using CpuBackend backend = new();
        using HunyuanVideo15Dit streamed = new(config);
        using HunyuanVideo15BlockLoader blockLoader = new(streamed, config, backend, source, castSources, readBlocksFromFile: readFromFile);
        blockLoader.LoadShared();
        streamed.LoadBlock = blockLoader.LoadBlock;
        streamed.UnloadBlock = blockLoader.UnloadBlock;
        Assert.True(streamed.StreamsBlockWeights);

        Random rng = new(7);
        using Tensor latent = RandomTensor(rng, 1, 4, 2, 4, 4);
        using Tensor qwen = RandomTensor(rng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(rng, 1, 2, 16);
        float[] frameTimesteps = [1000f, 600f];

        using Tensor expected = resident.ForwardPerFrame(backend, latent, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        using Tensor actual = streamed.ForwardPerFrame(backend, latent, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        Assert.Equal(expected.Shape, actual.Shape);
        Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(actual.AsReadOnlySpan<float>()),
            "The streamed forward must equal the resident forward bit for bit.");

        // A second forward reloads the blocks, and still matches.
        using Tensor again = streamed.ForwardPerFrame(backend, latent, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(again.AsReadOnlySpan<float>()));
    }

    /// <summary>The CPU's no-op work, with the tensors it was asked to upload and to release recorded, and with chosen calls failing (1-based; 0 never
    /// fails): the upload, the device release, and the host release.</summary>
    private sealed class FailingPlacement : IBlockPlacement
    {
        private readonly BackendBlockPlacement _inner = new(new CpuBackend());
        private int _preloads;
        private int _frees;

        public int FailPreloadOn { get; init; }

        public int FailFreeOn { get; init; }

        public int FailDisposeHostOn { get; init; }

        /// <summary>Every tensor an upload was asked to place, including the uploads that failed.</summary>
        public List<Tensor> Preloaded { get; } = [];

        /// <summary>Every tensor a host release was asked to release, including the releases that failed.</summary>
        public HashSet<Tensor> HostDisposed { get; } = new(ReferenceEqualityComparer.Instance);

        /// <summary>Calls to the device free, including the one that failed.</summary>
        public int FreeCalls => _frees;

        /// <summary>Calls to the host release, including the one that failed.</summary>
        public int HostReleaseCalls { get; private set; }

        public void Preload(IEnumerable<Tensor> weights)
        {
            _uploadStarted = true;
            List<Tensor> list = [.. weights];
            Preloaded.AddRange(list);
            if (++_preloads == FailPreloadOn) throw new InvalidOperationException("forced upload failure");
            _inner.Preload(list);
        }

        private bool _uploadStarted;
        private int _hostAfterUpload;

        public void Free(IEnumerable<Tensor> weights)
        {
            if (++_frees == FailFreeOn) throw new InvalidOperationException("forced release failure");
            _inner.Free(weights);
        }

        public void Sync() => _inner.Sync();

        public void DisposeHost(Tensor tensor)
        {
            HostDisposed.Add(tensor);
            HostReleaseCalls++;
            // FailDisposeHostOn counts the host releases made after the first upload began, so it targets the unload or the rollback, not the
            // loader's own releases of converter outputs before the upload.
            if (_uploadStarted && ++_hostAfterUpload == FailDisposeHostOn)
            {
                tensor.Dispose();
                throw new InvalidOperationException("forced host release failure");
            }
            ((IBlockPlacement)_inner).DisposeHost(tensor);
        }
    }

    [Fact]
    public void AFailedHostReleaseStillClearsTheBlock()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        using HunyuanVideo15Dit dit = new(config);
        using HunyuanVideo15BlockLoader blockLoader = new(dit, config, new FailingPlacement { FailDisposeHostOn = 1 }, Source(loader), [], true);
        blockLoader.LoadShared();
        blockLoader.LoadBlock(0);

        Assert.Throws<InvalidOperationException>(() => blockLoader.UnloadBlock(0));
        // The device copy was freed first; one host copy failed to release, and the state is clear with the other copies released too.
        Assert.Equal(-1, blockLoader.LoadedBlock);
        Assert.Equal(0, blockLoader.LoadedBlockTensorCount);
        Assert.Equal(blockLoader.BlockLoads, blockLoader.BlockReleases);
        blockLoader.UnloadBlock(0);
    }

    [Fact]
    public void AFailedRollbackReleaseDoesNotMaskTheLoadException()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        using HunyuanVideo15Dit dit = new(config);
        FailingPlacement placement = new() { FailPreloadOn = 1, FailDisposeHostOn = 1 };
        using HunyuanVideo15BlockLoader blockLoader = new(dit, config, placement, Source(loader), [], true);
        blockLoader.LoadShared();

        InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() => blockLoader.LoadBlock(0));
        Assert.Equal("forced upload failure", thrown.Message);
        Assert.Equal(-1, blockLoader.LoadedBlock);
        Assert.Equal(0, blockLoader.LoadedBlockTensorCount);
        Assert.Equal(blockLoader.BlockLoads, blockLoader.BlockReleases);
        // Every tensor the failed upload was asked to place is released, even though one release failed.
        Assert.NotEmpty(placement.Preloaded);
        Assert.All(placement.Preloaded, tensor => Assert.Contains(tensor, placement.HostDisposed));
    }

    [Fact]
    public void AFailedUploadRollsTheBlockBackAndTheLoaderRecovers()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        using HunyuanVideo15Dit dit = new(config);
        FailingPlacement placement = new() { FailPreloadOn = 1 };
        using HunyuanVideo15BlockLoader blockLoader = new(dit, config, placement, Source(loader), [], true);
        blockLoader.LoadShared();

        Assert.Throws<InvalidOperationException>(() => blockLoader.LoadBlock(0));
        Assert.Equal(-1, blockLoader.LoadedBlock);
        Assert.Equal(0, blockLoader.LoadedBlockTensorCount);
        Assert.Equal(blockLoader.BlockLoads, blockLoader.BlockReleases);
        // The tensors the failed upload was asked to place are disposed by the rollback.
        Assert.NotEmpty(placement.Preloaded);
        Assert.All(placement.Preloaded, tensor => Assert.Contains(tensor, placement.HostDisposed));

        // The next load works: the failed one left nothing behind to trip over.
        blockLoader.LoadBlock(0);
        Assert.Equal(0, blockLoader.LoadedBlock);
        blockLoader.UnloadBlock(0);
        Assert.Equal(-1, blockLoader.LoadedBlock);
        Assert.Equal(0, blockLoader.LoadedBlockTensorCount);
    }

    [Fact]
    public void AFailedDeviceReleaseKeepsTheBlockForARetry()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        using HunyuanVideo15Dit dit = new(config);
        FailingPlacement placement = new() { FailFreeOn = 1 };
        using HunyuanVideo15BlockLoader blockLoader = new(dit, config, placement, Source(loader), [], true);
        blockLoader.LoadShared();
        blockLoader.LoadBlock(0);
        int held = blockLoader.LoadedBlockTensorCount;
        Assert.True(held > 0);

        // The device free failed: the block stays loaded with its host copies, so the release can be retried.
        int freesBefore = placement.FreeCalls;
        int hostBefore = placement.HostReleaseCalls;
        Assert.Throws<InvalidOperationException>(() => blockLoader.UnloadBlock(0));
        Assert.Equal(freesBefore + 1, placement.FreeCalls);
        Assert.Equal(hostBefore, placement.HostReleaseCalls);
        Assert.Equal(0, blockLoader.LoadedBlock);
        Assert.Equal(held, blockLoader.LoadedBlockTensorCount);
        Assert.Equal(blockLoader.BlockLoads - 1, blockLoader.BlockReleases);

        // The retry frees the device copy a second time and releases each host copy exactly once.
        blockLoader.UnloadBlock(0);
        Assert.Equal(freesBefore + 2, placement.FreeCalls);
        Assert.Equal(hostBefore + held, placement.HostReleaseCalls);
        Assert.Equal(-1, blockLoader.LoadedBlock);
        Assert.Equal(0, blockLoader.LoadedBlockTensorCount);
        Assert.Equal(blockLoader.BlockLoads, blockLoader.BlockReleases);

        // A further release of a block that is not loaded does nothing.
        blockLoader.UnloadBlock(0);
        Assert.Equal(blockLoader.BlockLoads, blockLoader.BlockReleases);
    }

    [Fact]
    public void DisposeWithAFailingDeviceReleaseStillReleasesTheHostCopies()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        using HunyuanVideo15Dit dit = new(config);
        FailingPlacement placement = new() { FailFreeOn = 1 };
        HunyuanVideo15BlockLoader blockLoader = new(dit, config, placement, Source(loader), [], true);
        blockLoader.LoadShared();
        blockLoader.LoadBlock(0);
        int heldBlock = blockLoader.LoadedBlockTensorCount;
        int disposedBefore = placement.HostDisposed.Count;

        // Dispose reports the failed device release, but the block's host copies and the shared copies are released regardless.
        InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() => blockLoader.Dispose());
        Assert.Equal("forced release failure", thrown.Message);
        Assert.Equal(-1, blockLoader.LoadedBlock);
        Assert.Equal(0, blockLoader.LoadedBlockTensorCount);
        Assert.Equal(blockLoader.BlockLoads, blockLoader.BlockReleases);
        Assert.True(placement.HostDisposed.Count - disposedBefore > heldBlock, "The shared copies were not released.");

        // Disposing again is a no-op.
        blockLoader.Dispose();
    }

    [Fact]
    public void DisposeWithAFailingHostReleaseCountsTheReleaseOnce()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        using HunyuanVideo15Dit dit = new(config);
        FailingPlacement placement = new() { FailDisposeHostOn = 1 };
        HunyuanVideo15BlockLoader blockLoader = new(dit, config, placement, Source(loader), [], true);
        blockLoader.LoadShared();
        blockLoader.LoadBlock(0);

        // The first host release fails inside Dispose's own release of the block. That release already counted the block as released, so Dispose
        // must not count it again.
        Assert.Throws<InvalidOperationException>(() => blockLoader.Dispose());
        Assert.Equal(-1, blockLoader.LoadedBlock);
        Assert.Equal(0, blockLoader.LoadedBlockTensorCount);
        Assert.Equal(blockLoader.BlockLoads, blockLoader.BlockReleases);
    }
}
