using SharpInference.Audio.Cache;
using SharpInference.Audio.Pipelines;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Live F5-TTS forward test. Unlike the older <c>F5TtsSmokeTests</c> entry
/// that's hardcoded <c>Skip = ...</c>, this one runs unconditionally — if F5 weights
/// aren't on disk we early-out (test still passes), otherwise we exercise the full
/// DiT forward path on synthetic inputs and assert the output shape + finiteness.
///
/// <para>This is the first half of the TTS → STT loop: validating that F5 produces
/// a valid mel tensor we can later vocode and pipe into Whisper.</para></summary>
public sealed class F5ForwardLiveTest
{
    [Fact]
    public async Task F5Dit_SmokeForward_ProducesFiniteOutput()
    {
        string repoDir = AudioModelCache.GetRepoDirectory("SWivid/F5-TTS");
        string ditPath = Path.Combine(repoDir, "F5TTS_v1_Base", "model_1250000.safetensors");
        if (!File.Exists(ditPath))
        {
            // No weights — skip without failing.
            return;
        }

        // Load + forward on a small synthetic input. SmokeForward uses t=8 mel frames and
        // textLen=5 chars (tiny — runs in seconds on CPU even on the full 335M-param DiT).
        using F5TtsPipeline pipeline = await F5TtsPipeline.LoadAsync();
        using CpuBackend backend = new();
        Tensor v = pipeline.SmokeForward(backend, t: 8, textLen: 5);

        try
        {
            Assert.Equal(3, v.Shape.Rank);
            Assert.Equal(1, (int)v.Shape[0]);             // batch
            Assert.Equal(8, (int)v.Shape[1]);             // time = T
            Assert.Equal(100, (int)v.Shape[2]);           // mel_dim

            // Finite, not NaN.
            unsafe
            {
                float* p = (float*)v.DataPointer;
                long n = v.ElementCount;
                int nonZero = 0;
                for (long i = 0; i < n; i++)
                {
                    Assert.False(float.IsNaN(p[i]), $"output[{i}] is NaN");
                    Assert.False(float.IsInfinity(p[i]), $"output[{i}] is Inf");
                    if (MathF.Abs(p[i]) > 1e-6f) nonZero++;
                }
                // Most elements must be non-zero — if the forward produced an all-zero
                // tensor something's badly broken (the layernorms / zero-init final layer
                // could legitimately produce small outputs, but not literally zero across
                // the board on a non-trivial input).
                Assert.True(nonZero > n / 4, $"forward output is mostly zero — only {nonZero}/{n} non-zero elements");
            }
        }
        finally
        {
            v.Dispose();
        }
    }
}
