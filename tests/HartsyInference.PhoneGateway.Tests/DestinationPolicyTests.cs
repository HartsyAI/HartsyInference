using HartsyInference.PhoneGateway.Admin;
using HartsyInference.PhoneGateway.Sip;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Where an agent may dial and what the admin endpoint answers are both silent failures when wrong: an open
/// dial plan is toll fraud, and a mapped status that drifts to 502 hides a busy line from the caller of the API.</summary>
public sealed class DestinationPolicyTests
{
    private static readonly string[] _usPrefix = ["+1555"];

    [Theory]
    [InlineData("+15551234", true)]
    [InlineData("+19005551234", false)]
    [InlineData("sip:+15551234@trunk.example.net", true)]
    [InlineData("sip:+19005551234@trunk.example.net", false)]
    [InlineData("+15551234@trunk.example.net", true)]
    [InlineData("sip:trunk.example.net", false)]
    [InlineData("not a uri:::", false)]
    public void Prefixes_MatchTheDialledNumber(string destination, bool allowed)
    {
        Assert.Equal(allowed, CallController.IsDestinationAllowed(destination, _usPrefix));
    }

    [Theory]
    [InlineData("+19005551234")]
    [InlineData("sip:anyone@anywhere.example")]
    public void NoPrefixes_AllowAnyDestination(string destination)
    {
        Assert.True(CallController.IsDestinationAllowed(destination, []));
    }

    [Theory]
    [InlineData(CallPlacementStatus.Placed, 202)]
    [InlineData(CallPlacementStatus.Busy, 409)]
    [InlineData(CallPlacementStatus.HostUnavailable, 503)]
    [InlineData(CallPlacementStatus.NotAllowed, 403)]
    [InlineData(CallPlacementStatus.Invalid, 400)]
    [InlineData(CallPlacementStatus.NotAnswered, 502)]
    [InlineData(CallPlacementStatus.Failed, 502)]
    public void PlacementStatus_MapsToItsHttpStatus(CallPlacementStatus status, int http)
    {
        Assert.Equal(http, AdminEndpoint.HttpStatusFor(status));
    }
}
