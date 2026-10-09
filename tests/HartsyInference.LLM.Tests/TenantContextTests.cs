using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The tenant a request runs under: a tenant the request names wins for a library caller, the caller's identity is used when the request names none, and a request
/// with no identity is the local tenant. The identity does not leak out of the flow that set it.</summary>
public sealed class TenantContextTests
{
    [Fact]
    public void An_Unidentified_Request_Runs_As_The_Local_Tenant()
    {
        Assert.Null(TenantContext.Current);
        Assert.Equal(TenantContext.Local, TenantContext.Resolve(null));
    }

    [Fact]
    public async Task The_Current_Identity_Is_Used_When_The_Request_Names_No_Tenant_And_Does_Not_Leak()
    {
        Task inner = Task.Run(() =>
        {
            TenantContext.Current = "alice";
            Assert.Equal("alice", TenantContext.Resolve(null));
            Assert.Equal("bob", TenantContext.Resolve("bob"));
        });
        await inner;

        Assert.Null(TenantContext.Current);
    }
}
