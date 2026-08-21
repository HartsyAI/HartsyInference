using HartsyInference.Core.Memory;
using Xunit.Sdk;

[assembly: HartsyInference.Diffusion.Tests.TrimHostMemoryAfterEachTest]

namespace HartsyInference.Diffusion.Tests;

/// <summary>Root cause of the OOM-kills this project used to produce: dozens of *RealWeight*/*Vram*/
/// *ShardingEngine* classes construct pipelines directly (<c>new XyzRecipe().Construct(...)</c>), bypassing
/// <see cref="HartsyInference.Engine.InferenceEngine"/>'s model-switch eviction sweep — the only place that
/// already calls <see cref="HostMemory.Trim"/>. Without it, freed weight buffers strand their glibc arena heap
/// (see <see cref="HostMemory"/>'s remarks — this is the exact mechanism recorded for SwarmUI in
/// swarmui-glibc-arena-retention: freeing is not returning), so anon RSS ratchets upward across the whole
/// process as different model families rotate through, until the kernel OOM-kills the test host outright
/// (reproduced repeatedly: 45-49 GB anon-rss, confirmed via dmesg/journalctl).
///
/// <para>Runs <see cref="HostMemory.Trim"/> after every test in the assembly, mirroring exactly what the engine
/// already does on its own eviction path — proven numerically inert elsewhere (bit-identical output across the
/// fix), so this changes memory footprint only, never pipeline behavior or test semantics.</para></summary>
public sealed class TrimHostMemoryAfterEachTest : BeforeAfterTestAttribute
{
    public override void After(System.Reflection.MethodInfo methodUnderTest) =>
        HostMemory.TrimAndLog($"test: {methodUnderTest.DeclaringType?.Name}.{methodUnderTest.Name}");
}
