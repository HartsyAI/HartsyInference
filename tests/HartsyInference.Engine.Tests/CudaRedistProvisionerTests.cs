using System.Formats.Tar;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;

namespace HartsyInference.Engine.Tests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class CudaRedistProvisionerCollection
{
    public const string Name = "CudaRedistProvisioner";
}

/// <summary>cuBLAS/cuDNN redist provisioning without a network or a GPU: archives are built locally and passed as a path, and the
/// cache root is redirected to a temp folder. Runs alone (no other class in parallel), because the cache root and the knobs are
/// process-wide and the resolver's probe folders are fixed the first time anything touches it.</summary>
[Collection(CudaRedistProvisionerCollection.Name)]
public sealed class CudaRedistProvisionerTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-redist-" + Guid.NewGuid().ToString("N"));
    private static readonly bool Windows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);

    public CudaRedistProvisionerTests()
    {
        Directory.CreateDirectory(_root);
        CudaRedistProvisioner.CacheRootOverride = Path.Combine(_root, "cache");
    }

    public void Dispose()
    {
        CudaRedistProvisioner.CacheRootOverride = null;
        KnobStore.Clear(EngineKnobs.CublasUrl);
        KnobStore.Clear(EngineKnobs.CublasVersion);
        try { Directory.Delete(_root, recursive: true); } catch { /* temp */ }
    }

    [Theory]
    [InlineData(13, new[] { 13, 12, 11 })]
    [InlineData(14, new[] { 13, 12, 11 })]
    [InlineData(12, new[] { 12, 11 })]
    [InlineData(0, new[] { 13, 12, 11 })]
    public void PickMajors_NeverOffersANewerCudaThanTheDriver(int driver, int[] expected) =>
        Assert.Equal(expected, CublasRuntime.PickMajors(driver));

    [Fact]
    public void Soname_FollowsThePlatformNaming()
    {
        Assert.Equal("cublas64_13.dll", CublasRuntime.Soname(13, windows: true));
        Assert.Equal("cublasLt64_12.dll", CublasRuntime.Soname(12, lt: true, windows: true));
        Assert.Equal("libcublas.so.13", CublasRuntime.Soname(13, windows: false));
        Assert.Equal("libcublasLt.so.11", CublasRuntime.Soname(11, lt: true, windows: false));
    }

    [Theory]
    [InlineData(13, 13, "13.10.0.20")]
    [InlineData(14, 13, "13.10.0.20")]
    [InlineData(12, 12, "12.9.2.10")]
    public void FetchSpec_PinsTheDefaultBuildAndItsHash(int driver, int major, string version)
    {
        CudaRedistSpec? spec = CublasRuntime.FetchSpec(driver);
        if (CudaRedistProvisioner.Platform() is null)
        {
            // NVIDIA publishes no x86-64-named redist for this platform (e.g. an arm64 host): no download is offered.
            Assert.Null(spec);
            return;
        }
        Assert.NotNull(spec);
        Assert.Equal(major, spec.CudaMajor);
        Assert.Contains(version, spec.Url);
        Assert.StartsWith("https://developer.download.nvidia.com/compute/cuda/redist/libcublas/", spec.Url);
        CublasRuntime.RedistBuild build = CublasRuntime.DefaultBuilds.Single(b => b.CudaMajor == major);
        Assert.Equal(Windows ? build.WindowsSha256 : build.LinuxSha256, spec.ExpectedSha256);
        Assert.Equal(CublasRuntime.Soname(major), spec.Sentinel);
    }

    [Fact]
    public void FetchSpec_WithOverridesSkipsTheHash()
    {
        KnobStore.Set(EngineKnobs.CublasVersion, "13.9.0.1");
        CudaRedistSpec? versioned = CublasRuntime.FetchSpec(13);
        Assert.NotNull(versioned);
        Assert.Contains("13.9.0.1", versioned.Url);
        Assert.Null(versioned.ExpectedSha256);

        KnobStore.Clear(EngineKnobs.CublasVersion);
        KnobStore.Set(EngineKnobs.CublasUrl, "/srv/mirror/libcublas.tar.xz");
        CudaRedistSpec? mirrored = CublasRuntime.FetchSpec(13);
        Assert.NotNull(mirrored);
        Assert.Equal("/srv/mirror/libcublas.tar.xz", mirrored.Url);
        Assert.Null(mirrored.ExpectedSha256);
    }

    [Fact]
    public void FetchSpec_RefusesADriverWithNoPublicBuild() => Assert.Null(CublasRuntime.FetchSpec(11));

    [Fact]
    public void FetchSpec_RefusesAVersionOverrideNewerThanTheDriver()
    {
        // A CUDA 13 cuBLAS on a CUDA 12 driver loads and then fails at cublasCreate; nothing should be downloaded for it.
        KnobStore.Set(EngineKnobs.CublasVersion, "13.10.0.20");
        Assert.Null(CublasRuntime.FetchSpec(12));
    }

    [Fact]
    public void StalledReadFailsInsteadOfHanging()
    {
        TimeSpan saved = CudaRedistProvisioner.StallTimeout;
        CudaRedistProvisioner.StallTimeout = TimeSpan.FromMilliseconds(200);
        try
        {
            using NeverStream stalled = new();
            Assert.Throws<TimeoutException>(() => CudaRedistProvisioner.ReadWithStallTimeout(stalled, new byte[16]));
        }
        finally
        {
            CudaRedistProvisioner.StallTimeout = saved;
        }
    }

    /// <summary>A stream whose reads never complete, as a connection that stalls mid-body.</summary>
    private sealed class NeverStream : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => 0; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            await Task.Delay(Timeout.Infinite, cancellationToken);
            return 0;
        }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    [Fact]
    public void FetchInto_UnpacksAndFlattensAVerifiedLocalArchive()
    {
        (string archive, string sha) = BuildArchive(13);
        string? libDir = CudaRedistProvisioner.FetchInto(Spec(archive, sha));

        Assert.Equal(CudaRedistProvisioner.CacheLibDir("cublas", 13), libDir);
        Assert.True(File.Exists(Path.Combine(libDir!, CublasRuntime.Soname(13))));
        Assert.True(File.Exists(Path.Combine(libDir!, CublasRuntime.Soname(13, lt: true))));
        Assert.False(File.Exists(Path.Combine(libDir!, "README.txt")));   // only the library patterns are kept
        Assert.False(Directory.Exists(Path.Combine(Path.GetDirectoryName(libDir)!, "download")));
    }

    [Fact]
    public void FetchInto_RejectsAHashMismatchWithoutInstallingAnything()
    {
        (string archive, _) = BuildArchive(13);
        string? libDir = CudaRedistProvisioner.FetchInto(Spec(archive, new string('0', 64)));

        Assert.Null(libDir);
        Assert.False(File.Exists(Path.Combine(CudaRedistProvisioner.CacheLibDir("cublas", 13), CublasRuntime.Soname(13))));
    }

    [Fact]
    public void FetchInto_ReportsFailureWhenTheSentinelIsMissing()
    {
        (string archive, string sha) = BuildArchive(12);   // holds the CUDA 12 names, spec expects 13
        Assert.Null(CudaRedistProvisioner.FetchInto(Spec(archive, sha)));
    }

    [Fact]
    public void FetchInto_ReportsAMissingArchive() =>
        Assert.Null(CudaRedistProvisioner.FetchInto(Spec(Path.Combine(_root, "absent.tar"), null)));

    [Fact]
    public void Flatten_RecreatesSonameLinksInsteadOfCopyingTwice()
    {
        if (Windows) return;   // NVIDIA's Windows archives hold no links
        string tree = Path.Combine(_root, "tree", "lib");
        Directory.CreateDirectory(tree);
        File.WriteAllText(Path.Combine(tree, "libcublas.so.13.10.0.20"), "real");
        File.CreateSymbolicLink(Path.Combine(tree, "libcublas.so.13"), "libcublas.so.13.10.0.20");
        string libDir = Path.Combine(_root, "flat");
        Directory.CreateDirectory(libDir);

        Assert.Equal(2, CudaRedistProvisioner.Flatten(Path.Combine(_root, "tree"), libDir, ["libcublas*.so*"]));
        Assert.Equal("libcublas.so.13.10.0.20", new FileInfo(Path.Combine(libDir, "libcublas.so.13")).LinkTarget);
        Assert.Equal("real", File.ReadAllText(Path.Combine(libDir, "libcublas.so.13")));
    }

    private static CudaRedistSpec Spec(string url, string? sha) => new(
        Component: "cublas",
        Label: "cuBLAS",
        Url: url,
        CudaMajor: 13,
        ExpectedSha256: sha,
        ExpectedBytes: null,
        LibraryPatterns: [Windows ? "cublas*.dll" : "libcublas*.so*"],
        Sentinel: CublasRuntime.Soname(13),
        ApproxSize: "test",
        KnobHint: "paths.cublasAutofetch");

    /// <summary>A tar shaped like NVIDIA's redist (<c>libcublas-…-archive/bin|lib/…</c>) holding fake library files.</summary>
    private (string Path, string Sha) BuildArchive(int major)
    {
        string source = Path.Combine(_root, $"src{major}");
        string libs = Path.Combine(source, $"libcublas-test-{major}-archive", Windows ? "bin" : "lib");
        Directory.CreateDirectory(libs);
        File.WriteAllText(Path.Combine(libs, CublasRuntime.Soname(major)), "cublas");
        File.WriteAllText(Path.Combine(libs, CublasRuntime.Soname(major, lt: true)), "cublasLt");
        File.WriteAllText(Path.Combine(source, $"libcublas-test-{major}-archive", "README.txt"), "not a library");
        string archive = Path.Combine(_root, $"libcublas-test-{major}.tar");
        TarFile.CreateFromDirectory(source, archive, includeBaseDirectory: false);
        using FileStream stream = File.OpenRead(archive);
        return (archive, Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant());
    }
}
