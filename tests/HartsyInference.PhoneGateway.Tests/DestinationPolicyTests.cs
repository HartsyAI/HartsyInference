using HartsyInference.PhoneGateway.Admin;
using HartsyInference.PhoneGateway.Sip;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Where an agent may dial and what the admin endpoint answers are both silent failures when wrong: an open
/// dial plan is toll fraud (and a matching number at a foreign host hands that host the account's digest), and a
/// mapped status that drifts to 502 hides a busy line from the caller of the API.</summary>
public sealed class DestinationPolicyTests
{
    private const string Trunk = "trunk.example.net";

    private static readonly string[] _usPrefix = ["+1555"];

    [Theory]
    [InlineData("+15551234", true)]
    [InlineData("+19005551234", false)]
    [InlineData("sip:+15551234@trunk.example.net", true)]
    [InlineData("sip:+15551234@TRUNK.example.net:5060", true)]
    [InlineData("+15551234@trunk.example.net", true)]
    [InlineData("sip:+15551234@attacker.example", false)]
    [InlineData("+15551234@attacker.example", false)]
    [InlineData("sip:+19005551234@trunk.example.net", false)]
    [InlineData("sip:trunk.example.net", false)]
    [InlineData("not a uri:::", false)]
    public void Prefixes_AllowOnlyMatchingNumbersThroughTheRegistrar(string destination, bool allowed)
    {
        Assert.Equal(allowed, CallController.IsDestinationAllowed(destination, _usPrefix, Trunk));
    }

    [Fact]
    public void Prefixes_WithoutARegistrar_AllowNothing()
    {
        Assert.False(CallController.IsDestinationAllowed("+15551234", _usPrefix, ""));
        Assert.False(CallController.IsDestinationAllowed("sip:+15551234@anywhere.example", _usPrefix, ""));
    }

    [Theory]
    [InlineData("+19005551234")]
    [InlineData("sip:anyone@anywhere.example")]
    public void NoPrefixes_AllowAnyDestination(string destination)
    {
        Assert.True(CallController.IsDestinationAllowed(destination, [], Trunk));
    }

    [Theory]
    [InlineData("+15551234", "sip:+15551234@trunk.example.net")]
    [InlineData("+15551234@other.example", "sip:+15551234@other.example")]
    [InlineData("sip:+15551234@other.example:5070", "sip:+15551234@other.example:5070")]
    public void DialString_RoutesBareNumbersThroughTheRegistrar(string destination, string expected)
    {
        Assert.Equal(expected, CallController.DialString(destination, Trunk));
    }

    [Fact]
    public void DialString_BareNumberWithoutRegistrar_IsNotDialable()
    {
        Assert.Null(CallController.DialString("+15551234", ""));
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
