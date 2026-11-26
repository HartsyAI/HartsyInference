using HartsyInference.Core.Backends;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Audio.Tests.Separation;

/// <summary>Skips, with the reason, unless the named GPU backend can be constructed: a device alone is not enough without its kernels and driver.</summary>
public sealed class GpuBackendFactAttribute : FactAttribute
{
    public GpuBackendFactAttribute(string backend)
    {
        try
        {
            using IBackend probe = BackendFactory.Create(backend);
        }
        catch (Exception error)
        {
            Skip = $"SKIPPED: no usable {backend} backend ({error.Message}).";
        }
    }
}
