using System.Reflection;
using HartsyInference.Core.Backends;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

public sealed class ExactGemmScopeTests
{
    [Fact]
    public void Scope_TurnsHighPrecisionOn_AndRestoresOff()
    {
        IBackend backend = Fake(false);
        using (new ExactGemmScope(backend))
            Assert.True(backend.HighPrecisionGemm);
        Assert.False(backend.HighPrecisionGemm);
    }

    [Fact]
    public void Scope_RestoresAnExplicitOnSetting()
    {
        IBackend backend = Fake(true);
        using (new ExactGemmScope(backend))
            Assert.True(backend.HighPrecisionGemm);
        Assert.True(backend.HighPrecisionGemm);
    }

    [Fact]
    public void Scope_RestoresWhenTheBodyThrows()
    {
        IBackend backend = Fake(false);
        void Forward()
        {
            using ExactGemmScope exact = new(backend);
            throw new InvalidOperationException("forward failed");
        }

        Assert.Throws<InvalidOperationException>(Forward);
        Assert.False(backend.HighPrecisionGemm);
    }

    [Fact]
    public void Scope_NestedScopesRestoreInOrder()
    {
        IBackend backend = Fake(false);
        using (new ExactGemmScope(backend))
        {
            using (new ExactGemmScope(backend))
                Assert.True(backend.HighPrecisionGemm);
            Assert.True(backend.HighPrecisionGemm);
        }

        Assert.False(backend.HighPrecisionGemm);
    }

    private static IBackend Fake(bool highPrecision)
    {
        IBackend backend = DispatchProxy.Create<IBackend, SettingsProxy>();
        ((SettingsProxy)(object)backend).HighPrecision = highPrecision;
        return backend;
    }

    /// <summary>Implements only the <see cref="IBackend.HighPrecisionGemm"/> property; any other member is a test bug.</summary>
    public class SettingsProxy : DispatchProxy
    {
        public bool HighPrecision { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            switch (targetMethod?.Name)
            {
                case "get_HighPrecisionGemm":
                    return HighPrecision;
                case "set_HighPrecisionGemm":
                    HighPrecision = (bool)args![0]!;
                    return null;
                default:
                    throw new NotSupportedException(targetMethod?.Name);
            }
        }
    }
}
