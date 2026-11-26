using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>One CPU backend shared by a Duo test's cache and its forward: a cache refuses a forward on any other instance.</summary>
internal static class TestBackends
{
    public static readonly IBackend Cpu = new CpuBackend();
}
