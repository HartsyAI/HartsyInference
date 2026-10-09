using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A blank requested tenant is no tenant: it resolves as if none had been named, so it cannot form a tenant of its own.</summary>
public sealed class TenantResolveBlankTests
{
    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void A_Blank_Requested_Tenant_Resolves_As_No_Tenant(string requested)
    {
        Assert.Equal(TenantContext.Resolve(null), TenantContext.Resolve(requested));
    }

    [Fact]
    public void A_Named_Tenant_Is_Used_As_Given()
    {
        Assert.Equal("acme", TenantContext.Resolve("acme"));
    }
}
