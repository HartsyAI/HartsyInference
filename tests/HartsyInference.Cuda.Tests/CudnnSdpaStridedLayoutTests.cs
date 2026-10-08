using System.Diagnostics;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Feasibility probe for feeding cuDNN SDPA token-major [b,s,h,d] buffers via strides instead of the
/// contiguous head-major [b,h,s,d] layout. A fused QKV projection already emits token-major, so accepting it
/// directly would remove a permute on both the input and the output side; this asserts the strided plan is
/// buildable, numerically identical, and not slower than the contiguous one.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaStridedLayoutTests
{
    private readonly ITestOutputHelper _output;
    public CudnnSdpaStridedLayoutTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x9E3779B9u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 65535f - 0.5f; }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir)
            ? dir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private bool Skip()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return true; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return true; }
        return false;
    }

    private static ulong Upload(Half[] host)
    {
        ulong dptr = CudaMemory.Allocate((nuint)(host.Length * sizeof(ushort)));
        fixed (Half* p = host) CudaMemory.CopyHostToDevice(dptr, p, (nuint)(host.Length * sizeof(ushort)));
        return dptr;
    }

    private static Half[] Download(ulong dptr, int count)
    {
        Half[] host = new Half[count];
        fixed (Half* p = host) CudaMemory.CopyDeviceToHost(p, dptr, (nuint)(count * sizeof(ushort)));
        return host;
    }

    /// <summary>Fills head-major [h,s,d] and token-major [s,h,d] copies of the same logical tensor.</summary>
    private static (Half[] HeadMajor, Half[] TokenMajor) MakePair(int heads, int s, int d)
    {
        Half[] hm = new Half[heads * s * d];
        Half[] tm = new Half[heads * s * d];
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < s; i++)
                for (int e = 0; e < d; e++)
                {
                    Half value = (Half)Rand();
                    hm[((long)h * s + i) * d + e] = value;
                    tm[((long)i * heads + h) * d + e] = value;
                }
        return (hm, tm);
    }

    [Theory]
    [InlineData(4, 64, 64)]
    [InlineData(32, 512, 128)]
    public void StridedTokenMajor_MatchesContiguousHeadMajor(int heads, int s, int d)
    {
        if (Skip()) return;
        float scale = 1f / MathF.Sqrt(d);
        int n = heads * s * d;

        using CudaBackend backend = new(0, PtxDir());
        using CudnnSdpa sdpa = new(backend.Stream.Handle);

        (Half[] qh, Half[] qt) = MakePair(heads, s, d);
        (Half[] kh, Half[] kt) = MakePair(heads, s, d);
        (Half[] vh, Half[] vt) = MakePair(heads, s, d);

        ulong dq = 0, dk = 0, dv = 0, doHm = 0, dqT = 0, dkT = 0, dvT = 0, doTm = 0;
        try
        {
            dq = Upload(qh); dk = Upload(kh); dv = Upload(vh);
            doHm = CudaMemory.Allocate((nuint)(n * sizeof(ushort)));
            sdpa.Execute(dq, dk, dv, doHm, 1, heads, s, s, d, scale, CudnnSdpa.SdpaLayout.HeadMajor);
            backend.Sync();
            Half[] outHm = Download(doHm, n);

            dqT = Upload(qt); dkT = Upload(kt); dvT = Upload(vt);
            doTm = CudaMemory.Allocate((nuint)(n * sizeof(ushort)));
            sdpa.Execute(dqT, dkT, dvT, doTm, 1, heads, s, s, d, scale, CudnnSdpa.SdpaLayout.TokenMajor);
            backend.Sync();
            Half[] outTm = Download(doTm, n);

            float maxDiff = 0f;
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < s; i++)
                    for (int e = 0; e < d; e++)
                    {
                        float a = (float)outHm[((long)h * s + i) * d + e];
                        float bVal = (float)outTm[((long)i * heads + h) * d + e];
                        Assert.True(float.IsFinite(bVal), $"token-major output is non-finite at h={h} s={i} d={e}");
                        maxDiff = MathF.Max(maxDiff, MathF.Abs(a - bVal));
                    }
            _output.WriteLine($"heads={heads} s={s} d={d}: max |head-major - token-major| = {maxDiff:E3}");
            Assert.True(maxDiff <= 1e-3f, $"strided token-major SDPA diverges from contiguous head-major by {maxDiff:E3}");
        }
        finally
        {
            foreach (ulong p in new[] { dq, dk, dv, doHm, dqT, dkT, dvT, doTm })
                if (p != 0) CudaMemory.Free(p);
        }
    }

    /// <summary>LTX-2.5's real self-attention shape. If the strided descriptors push cuDNN off the fused
    /// flash engine onto a generic kernel, the saved permutes would be paid back with interest here.</summary>
    [Fact]
    public void StridedTokenMajor_ThroughputAtLtxShape()
    {
        if (Skip()) return;
        const int Heads = 32, S = 4992, D = 128, Reps = 20;
        float scale = 1f / MathF.Sqrt(D);
        long n = (long)Heads * S * D;

        using CudaBackend backend = new(0, PtxDir());
        using CudnnSdpa sdpa = new(backend.Stream.Handle);

        Half[] host = new Half[n];
        for (long i = 0; i < n; i++) host[i] = (Half)Rand();

        ulong dq = 0, dk = 0, dv = 0, dOut = 0;
        try
        {
            dq = Upload(host); dk = Upload(host); dv = Upload(host);
            dOut = CudaMemory.Allocate((nuint)(n * sizeof(ushort)));

            double Time(CudnnSdpa.SdpaLayout layout)
            {
                for (int i = 0; i < 3; i++)
                    sdpa.Execute(dq, dk, dv, dOut, 1, Heads, S, S, D, scale, layout);
                backend.Sync();
                Stopwatch sw = Stopwatch.StartNew();
                for (int i = 0; i < Reps; i++)
                    sdpa.Execute(dq, dk, dv, dOut, 1, Heads, S, S, D, scale, layout);
                backend.Sync();
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds / Reps;
            }

            double headMajorMs = Time(CudnnSdpa.SdpaLayout.HeadMajor);
            double tokenMajorMs = Time(CudnnSdpa.SdpaLayout.TokenMajor);
            _output.WriteLine(
                $"b=1 h={Heads} s={S} d={D}: head-major {headMajorMs:F3} ms/call, " +
                $"token-major {tokenMajorMs:F3} ms/call ({tokenMajorMs / headMajorMs:F2}×)");
        }
        finally
        {
            foreach (ulong p in new[] { dq, dk, dv, dOut })
                if (p != 0) CudaMemory.Free(p);
        }
    }
}
