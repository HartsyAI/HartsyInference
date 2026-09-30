using HartsyInference.PhoneGateway.Admin;
using HartsyInference.PhoneGateway.Sip;
using SIPSorcery.SIP;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>Where an agent may dial and what the admin endpoint answers are both silent failures when wrong: an open
/// dial plan is toll fraud (and a matching number that reaches any host but the registrar hands it the account's
/// digest), and a mapped status that drifts to 502 hides a busy line from the caller of the API.</summary>
public sealed class DestinationPolicyTests
{
    private const string Trunk = "trunk.example.net";

    private static readonly string[] _usPrefix = ["+1555"];

    [Theory]
    [InlineData("+15551234", true)]
    [InlineData("+19005551234", false)]
    [InlineData("sip:+15551234@trunk.example.net", true)]
    [InlineData("sip:+15551234@TRUNK.example.net", true)]
    [InlineData("sips:+15551234@trunk.example.net", true)]
    [InlineData("+15551234@trunk.example.net", true)]
    [InlineData("tel:+15551234", true)]
    [InlineData("tel:+19005551234", false)]
    [InlineData("sip:+19005551234@trunk.example.net", false)]
    [InlineData("sip:trunk.example.net", false)]
    [InlineData("sip:+15551234", false)]
    [InlineData("not a uri:::", false)]
    [InlineData("sip:+1555abc@trunk.example.net", false)]
    public void Prefixes_AllowOnlyMatchingNumbersThroughTheRegistrar(string destination, bool allowed)
    {
        Assert.Equal(allowed, CallController.IsDestinationAllowed(destination, _usPrefix, Trunk));
    }

    /// <summary>Anything beyond a number at the registrar's host is refused, not stripped: <c>maddr</c> (RFC 3261
    /// §19.1.1) overrides where the INVITE is sent, <c>transport</c> how, a port or a header changes the request, and a
    /// foreign host takes the call outright. The destination can come from the far end of the call through the
    /// agent.</summary>
    [Theory]
    [InlineData("sip:+15551234@trunk.example.net;maddr=attacker.example")]
    [InlineData("sip:+15551234@trunk.example.net;transport=tcp")]
    [InlineData("sip:+15551234@trunk.example.net:5060")]
    [InlineData("+15551234@trunk.example.net:5060")]
    [InlineData("sip:+15551234@trunk.example.net?Route=%3Csip:attacker.example%3E")]
    [InlineData("sip:+15551234@attacker.example")]
    [InlineData("+15551234@attacker.example")]
    [InlineData("sip:+15551234;maddr=attacker.example@trunk.example.net")]
    public void Prefixes_RefuseParametersPortsHeadersAndForeignHosts(string destination)
    {
        Assert.Null(CallController.AuthorizeDestination(destination, _usPrefix, Trunk, out CallPlacementStatus refusal));
        Assert.Equal(CallPlacementStatus.NotAllowed, refusal);
    }

    [Theory]
    [InlineData("sip:+1555%3Bmaddr=attacker.example@trunk.example.net")]
    [InlineData("tel:+15551234;phone-context=example.com")]
    [InlineData("tel:+15551234@attacker.example")]
    [InlineData("sip:+15551234@trunk.example.net%3Bmaddr=attacker.example")]
    [InlineData("sip:+15551234@trunk.example.net ;maddr=attacker.example")]
    public void Prefixes_RefuseEscapedOrDisguisedParameters(string destination)
    {
        Assert.False(CallController.IsDestinationAllowed(destination, _usPrefix, Trunk));
    }

    /// <summary>What is dialled is rebuilt from the validated number and the configured registrar, never passed
    /// through.</summary>
    [Theory]
    [InlineData("+15551234", Trunk, "sip:+15551234@trunk.example.net")]
    [InlineData("tel:+15551234", Trunk, "sip:+15551234@trunk.example.net")]
    [InlineData("+15551234@TRUNK.example.net", Trunk, "sip:+15551234@trunk.example.net")]
    [InlineData("sip:+15551234@TRUNK.example.net", Trunk, "sip:+15551234@trunk.example.net")]
    [InlineData("sips:+15551234@trunk.example.net", Trunk, "sips:+15551234@trunk.example.net")]
    [InlineData("sip:+15551234@trunk.example.net", "trunk.example.net:5070", "sip:+15551234@trunk.example.net:5070")]
    [InlineData("sip:+15551234@trunk.example.net", "TRUNK.Example.NET", "sip:+15551234@TRUNK.Example.NET")]
    [InlineData("+15551234", "[2001:db8::1]", "sip:+15551234@[2001:db8::1]")]
    public void Prefixes_DialTheNumberRebuiltAtTheRegistrar(string destination, string registrar, string expected)
    {
        Assert.Equal(expected, CallController.AuthorizeDestination(destination, _usPrefix, registrar, out _));
    }

    [Fact]
    public void Refusals_SayWhetherTheDestinationIsMalformedOrNotAllowed()
    {
        Assert.Null(CallController.AuthorizeDestination("sip:+19005551234@trunk.example.net", _usPrefix, Trunk, out CallPlacementStatus policy));
        Assert.Equal(CallPlacementStatus.NotAllowed, policy);
        Assert.Null(CallController.AuthorizeDestination("+15551234", _usPrefix, "", out CallPlacementStatus noRegistrar));
        Assert.Equal(CallPlacementStatus.Invalid, noRegistrar);
        Assert.Null(CallController.AuthorizeDestination("+15551234", [], "", out CallPlacementStatus open));
        Assert.Equal(CallPlacementStatus.Invalid, open);
    }

    [Fact]
    public void Prefixes_AnIPv6RegistrarMatchesOnlyTheNumberForms()
    {
        const string registrar = "[2001:db8::1]";
        Assert.Null(CallController.AuthorizeDestination("sip:+15551234@[2001:db8::1]", _usPrefix, registrar, out CallPlacementStatus refusal));
        Assert.Equal(CallPlacementStatus.NotAllowed, refusal);
        Assert.Equal("sip:+15551234@[2001:db8::1]", CallController.AuthorizeDestination("tel:+15551234", _usPrefix, registrar, out _));
    }

    [Fact]
    public void Prefixes_WithoutARegistrar_AllowNothing()
    {
        Assert.False(CallController.IsDestinationAllowed("+15551234", _usPrefix, ""));
        Assert.False(CallController.IsDestinationAllowed("sip:+15551234@anywhere.example", _usPrefix, ""));
    }

    /// <summary>The dial plan fails closed: no prefixes and no <c>allowAnyDestination</c> refuses everything, even a
    /// number that would go through the registrar.</summary>
    [Theory]
    [InlineData("+19005551234")]
    [InlineData("+15551234")]
    [InlineData("sip:anyone@anywhere.example")]
    [InlineData("tel:+15551234")]
    public void NoPrefixes_RefuseEveryDestinationByDefault(string destination)
    {
        Assert.False(CallController.IsDestinationAllowed(destination, [], Trunk));
        Assert.Null(CallController.AuthorizeDestination(destination, [], Trunk, allowAnyDestination: false, out CallPlacementStatus refusal));
        Assert.Equal(CallPlacementStatus.NotAllowed, refusal);
    }

    [Theory]
    [InlineData("+19005551234")]
    [InlineData("sip:anyone@anywhere.example")]
    public void AllowAnyDestination_AllowsAnyDestination(string destination)
    {
        Assert.NotNull(CallController.AuthorizeDestination(destination, [], Trunk, allowAnyDestination: true, out _));
    }

    [Fact]
    public void AllowAnyDestination_DialsTheDestinationAsGiven()
    {
        const string uri = "sip:anyone@anywhere.example:5070;transport=tcp";
        Assert.Equal(uri, CallController.AuthorizeDestination(uri, [], Trunk, allowAnyDestination: true, out _));
    }

    [Fact]
    public void AllowAnyDestination_NeverOverridesPrefixes()
    {
        Assert.Null(CallController.AuthorizeDestination("sip:anyone@anywhere.example", _usPrefix, Trunk, allowAnyDestination: true, out CallPlacementStatus refusal));
        Assert.Equal(CallPlacementStatus.NotAllowed, refusal);
    }

    /// <summary>Anything without a <c>sip:</c>/<c>sips:</c> scheme gets one; a colon alone (a port) is not a scheme,
    /// or <c>user@host:5060</c> would reach the strict URI parser in transfer unchanged.</summary>
    [Theory]
    [InlineData("+15551234", "sip:+15551234@trunk.example.net")]
    [InlineData("+15551234@other.example", "sip:+15551234@other.example")]
    [InlineData("+15551234@trunk.example.net:5060", "sip:+15551234@trunk.example.net:5060")]
    [InlineData("alice@pbx.example:5060", "sip:alice@pbx.example:5060")]
    [InlineData("sip:+15551234@other.example:5070", "sip:+15551234@other.example:5070")]
    [InlineData("SIPS:alice@secure.example", "SIPS:alice@secure.example")]
    [InlineData("tel:+15551234", "sip:+15551234@trunk.example.net")]
    [InlineData("tel:+15551234;phone-context=example.com", "sip:+15551234@trunk.example.net")]
    public void DialString_AddsTheSchemeAndRoutesNumbersThroughTheRegistrar(string destination, string expected)
    {
        Assert.Equal(expected, CallController.DialString(destination, Trunk));
    }

    /// <summary>The transfer tool parses with sipsorcery's strict <see cref="SIPURI.ParseSIPURI"/>, which throws on a
    /// string without a scheme; every destination it gets must arrive with one.</summary>
    [Theory]
    [InlineData("alice@pbx.example:5060")]
    [InlineData("+15551234@trunk.example.net:5060")]
    [InlineData("+15551234")]
    [InlineData("tel:+15551234")]
    public void DialString_AlwaysSatisfiesTheStrictParser(string destination)
    {
        string? dial = CallController.AuthorizeDestination(destination, [], Trunk, allowAnyDestination: true, out _);
        Assert.NotNull(dial);
        SIPURI uri = SIPURI.ParseSIPURI(dial);
        Assert.Equal(SIPSchemesEnum.sip, uri.Scheme);
    }

    [Theory]
    [InlineData("+15551234", "")]
    [InlineData("tel:+15551234", "")]
    [InlineData("tel:+15551234@attacker.example", Trunk)]
    [InlineData("tel:", Trunk)]
    public void DialString_NumberWithoutARegistrarOrMalformedTel_IsNotDialable(string destination, string registrar)
    {
        Assert.Null(CallController.DialString(destination, registrar));
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
