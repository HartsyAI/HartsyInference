using HartsyInference.Core.Runtime;
using Xunit;

namespace HartsyInference.Core.Tests.Runtime;

public sealed class CompositeDisposableTests
{
    [Fact]
    public void AThrowingPartDoesNotStopTheLaterParts()
    {
        Recording first = new(throws: true);
        Recording second = new(throws: false);
        CompositeDisposable composite = new(first, null, second);

        Assert.Throws<AggregateException>(composite.Dispose);
        Assert.True(first.Disposed);
        Assert.True(second.Disposed);
    }

    private sealed class Recording(bool throws) : IDisposable
    {
        public bool Disposed { get; private set; }

        public void Dispose()
        {
            Disposed = true;
            if (throws)
            {
                throw new InvalidOperationException("the part failed to release");
            }
        }
    }
}
