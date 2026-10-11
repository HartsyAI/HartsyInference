using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The F16 KV default resolves per backend and model type (CPU lane: no GPU needed). The CUDA side of the same rules is in KvF16SaturationTests.</summary>
[Collection("KnobStore")]
public sealed class KvCachesDefaultTests
{
    [Fact]
    public void Unset_KeepsBackendsWithoutF16Caches_AndNonTextUsers_OnF32()
    {
        try
        {
            KnobStore.Clear(EngineKnobs.KvF16);
            Assert.Null(EngineKnobs.KvF16.Value);
            using CpuBackend cpu = new();
            Assert.False(((IBackend)cpu).SupportsF16KvCache);
            Assert.False(KvCaches.F16For(cpu));
            Assert.False(KvCaches.F16For(null));
            Assert.False(KvCaches.F16Enabled);            // audio decoders and other FixedKvCache users
            Assert.True(KvCaches.F16ForCudaPlacement);    // placement only plans CUDA devices
        }
        finally { KnobStore.Clear(EngineKnobs.KvF16); }
    }

    [Fact]
    public void ExplicitValues_WinOverTheBackendDefault()
    {
        try
        {
            using CpuBackend cpu = new();
            KnobStore.Set(EngineKnobs.KvF16, true);
            Assert.True(KvCaches.F16For(cpu));
            Assert.True(KvCaches.F16Enabled);
            KnobStore.Set(EngineKnobs.KvF16, false);
            Assert.False(KvCaches.F16For(cpu));
            Assert.False(KvCaches.F16Enabled);
            Assert.False(KvCaches.F16ForCudaPlacement);
        }
        finally { KnobStore.Clear(EngineKnobs.KvF16); }
    }
}
