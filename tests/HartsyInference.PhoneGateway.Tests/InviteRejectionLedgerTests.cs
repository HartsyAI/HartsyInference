using HartsyInference.PhoneGateway.Sip;
using SIPSorcery.SIP;
using Xunit;

namespace HartsyInference.PhoneGateway.Tests;

/// <summary>The ledger is what keeps a retransmitted INVITE from being counted twice or offered as a new call; a wrong
/// key or a wrong lifetime would do either silently.</summary>
public sealed class InviteRejectionLedgerTests
{
    [Fact]
    public void Copies_AreRecordedOnceAndReportTheFirstRefusal()
    {
        InviteRejectionLedger ledger = new();
        SIPRequest invite = NewInvite();
        Assert.True(ledger.TryRecord(invite, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        SIPRequest copy = SIPRequest.ParseSIPRequest(invite.ToString());
        Assert.False(ledger.TryRecord(copy, SIPResponseStatusCodesEnum.Decline, "Decline", out InviteRejection first));
        Assert.Equal(SIPResponseStatusCodesEnum.BusyHere, first.Status);
        Assert.True(ledger.TryGet(copy, out InviteRejection known));
        Assert.Equal("Busy Here", known.Reason);
        Assert.Equal(1, ledger.Count);
    }

    [Fact]
    public void NewBranchOrNewCallId_IsANewInvite()
    {
        InviteRejectionLedger ledger = new();
        SIPRequest invite = NewInvite();
        Assert.True(ledger.TryRecord(invite, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        SIPRequest newBranch = SIPRequest.ParseSIPRequest(invite.ToString());
        newBranch.Header.Vias.TopViaHeader.Branch = CallProperties.CreateBranchId();
        Assert.True(ledger.TryRecord(newBranch, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        SIPRequest newCall = SIPRequest.ParseSIPRequest(invite.ToString());
        newCall.Header.CallId = CallProperties.CreateNewCallId();
        Assert.True(ledger.TryRecord(newCall, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        Assert.Equal(3, ledger.Count);
    }

    [Fact]
    public void Entries_ExpireAfterTheTransactionLifetime()
    {
        long now = 1_000;
        InviteRejectionLedger ledger = new(clockMs: () => now);
        SIPRequest invite = NewInvite();
        Assert.True(ledger.TryRecord(invite, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        now += InviteRejectionLedger.DefaultLifetimeMs - 1;
        Assert.True(ledger.TryGet(invite, out _));
        now += 1;
        Assert.False(ledger.TryGet(invite, out _));
        Assert.True(ledger.TryRecord(invite, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        Assert.Equal(1, ledger.Count);
    }

    [Fact]
    public void Capacity_EvictsTheOldestEntry()
    {
        InviteRejectionLedger ledger = new(capacity: 2);
        SIPRequest a = NewInvite();
        SIPRequest b = NewInvite();
        SIPRequest c = NewInvite();
        Assert.True(ledger.TryRecord(a, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        Assert.True(ledger.TryRecord(b, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        Assert.True(ledger.TryRecord(c, SIPResponseStatusCodesEnum.BusyHere, "Busy Here", out _));
        Assert.Equal(2, ledger.Count);
        Assert.False(ledger.TryGet(a, out _));
        Assert.True(ledger.TryGet(b, out _));
        Assert.True(ledger.TryGet(c, out _));
    }

    private static SIPRequest NewInvite() => SIPRequest.GetRequest(SIPMethodsEnum.INVITE, SIPURI.ParseSIPURI("sip:agent@127.0.0.1:5060"));
}
