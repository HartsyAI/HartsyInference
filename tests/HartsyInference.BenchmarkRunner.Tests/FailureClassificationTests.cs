using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.Core.Exceptions;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;

/// <summary>Native status codes must retain their failure meaning without scraping exception messages.</summary>
public sealed class FailureClassificationTests
{
    [Fact]
    public void KnownOutOfMemoryCodesAreClassified()
    {
        Assert.Equal("oom", Worker.ClassifyFailure(new OutOfVramException("test")));
        Assert.Equal("oom", Worker.ClassifyFailure(new CudaException(2, "test")));
        Assert.Equal("oom", Worker.ClassifyFailure(new VulkanException(-2, "test")));
        Assert.Equal("failed", Worker.ClassifyFailure(new CudaException(218, "test")));
        Assert.Equal("cancelled", Worker.ClassifyFailure(new OperationCanceledException()));
    }
}
