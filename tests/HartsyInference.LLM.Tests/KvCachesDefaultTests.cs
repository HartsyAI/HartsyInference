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
    private static void Reset()
    {
        KnobStore.Clear(EngineKnobs.TextKvF16);
        KnobStore.Clear(EngineKnobs.KvF16);
    }

    [Fact]
    public void Unset_KeepsBackendsWithoutF16Caches_AndNonTextUsers_OnF32()
    {
        try
        {
            Reset();
            Assert.False(EngineKnobs.TextKvF16.Value.HasValue);
            Assert.False(EngineKnobs.KvF16.Value);           // the published opt-in keeps its meaning and default
            using CpuBackend cpu = new();
            Assert.False(((IBackend)cpu).SupportsF16KvCache);
            Assert.False(KvCaches.F16For(cpu));
            Assert.False(KvCaches.F16For(null));
            Assert.False(KvCaches.F16Enabled);               // audio decoders and other FixedKvCache users
            Assert.True(KvCaches.F16ForCudaPlacement);       // placement only plans CUDA devices
        }
        finally { Reset(); }
    }

    [Fact]
    public void ExplicitValues_WinOverTheBackendDefault()
    {
        try
        {
            using CpuBackend cpu = new();
            KnobStore.Set(EngineKnobs.TextKvF16, true);
            Assert.True(KvCaches.F16For(cpu), "an explicit text setting is honored whatever the backend reports");
            Assert.False(KvCaches.F16Enabled, "the text setting does not force other model types");
            KnobStore.Set(EngineKnobs.TextKvF16, false);
            Assert.False(KvCaches.F16For(cpu));
            Assert.False(KvCaches.F16ForCudaPlacement);

            Reset();
            KnobStore.Set(EngineKnobs.KvF16, true);          // the old opt-in still forces F16 for everything
            Assert.True(KvCaches.F16Enabled);
            Assert.True(KvCaches.F16For(cpu));
            KnobStore.Set(EngineKnobs.TextKvF16, false);     // but an explicit text setting decides for text models
            Assert.False(KvCaches.F16For(cpu));
            Assert.True(KvCaches.F16Enabled);
        }
        finally { Reset(); }
    }
}
