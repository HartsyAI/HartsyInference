using System.Net;
using System.Net.Sockets;
using HartsyInference.Core.Distributed;
using Xunit;

namespace HartsyInference.Core.Tests.Distributed;

/// <summary>Rendezvous protocol over loopback TCP: one process hosts several ranks as tasks. Covers the handshake, late-starting
/// clients, token and rank rejection, the deadline's missing-rank report, and cancellation.</summary>
// TODO(unsure): FreePort() is racy; another process can take the port before the test binds it.
// TODO(unsure): ClientStartedBeforeRoot_* uses a fixed 600 ms delay, so it can pass or fail depending on machine load.
public sealed class UniqueIdRendezvousTests
{
    private static readonly byte[] RootPayload = Enumerable.Range(0, 128).Select(i => (byte)(i * 3 + 1)).ToArray();

    private static int FreePort()
    {
        TcpListener probe = new(IPAddress.Loopback, 0);
        probe.Start();
        int port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();
        return port;
    }

    private static DistributedRendezvousOptions Opts(int rank, int worldSize, int port, string token = "job-secret", int timeoutSeconds = 15) => new()
    {
        WorldSize = worldSize,
        Rank = rank,
        MasterAddress = "127.0.0.1",
        MasterPort = port,
        JobToken = token,
        Timeout = TimeSpan.FromSeconds(timeoutSeconds),
    };

    private static Task<byte[]> Join(DistributedRendezvousOptions options, CancellationToken ct = default) =>
        UniqueIdRendezvous.ExchangeAsync(options, () => RootPayload, ct);

    [Fact]
    public async Task ThreeRanks_AllReceiveRootPayload_AndRootGeneratesOnce()
    {
        int port = FreePort();
        int generated = 0;
        Func<byte[]> makeRoot = () =>
        {
            Interlocked.Increment(ref generated);
            return RootPayload;
        };

        Task<byte[]> root = UniqueIdRendezvous.ExchangeAsync(Opts(0, 3, port), makeRoot, default);
        Task<byte[]> rank1 = UniqueIdRendezvous.ExchangeAsync(Opts(1, 3, port), () => throw new InvalidOperationException("only rank 0 generates"), default);
        Task<byte[]> rank2 = UniqueIdRendezvous.ExchangeAsync(Opts(2, 3, port), () => throw new InvalidOperationException("only rank 0 generates"), default);

        byte[][] results = await Task.WhenAll(root, rank1, rank2);
        foreach (byte[] result in results)
        {
            Assert.Equal(RootPayload, result);
        }
        Assert.Equal(1, generated);
    }

    [Fact]
    public async Task ClientStartedBeforeRoot_RetriesUntilRootListens()
    {
        int port = FreePort();
        Task<byte[]> client = Join(Opts(1, 2, port));
        await Task.Delay(600);
        Task<byte[]> root = Join(Opts(0, 2, port));

        byte[][] results = await Task.WhenAll(root, client);
        Assert.Equal(RootPayload, results[1]);
    }

    [Fact]
    public async Task WrongJobToken_IsRejected_AndRightfulRankStillJoins()
    {
        int port = FreePort();
        Task<byte[]> root = Join(Opts(0, 2, port, token: "right-token"));
        Task<byte[]> imposter = Join(Opts(1, 2, port, token: "wrong-token"));

        InvalidOperationException rejected = await Assert.ThrowsAsync<InvalidOperationException>(() => imposter);
        Assert.Contains("rejected rank 1", rejected.Message);

        Task<byte[]> legitimate = Join(Opts(1, 2, port, token: "right-token"));
        byte[][] results = await Task.WhenAll(root, legitimate);
        Assert.Equal(RootPayload, results[1]);
    }

    [Fact]
    public async Task DuplicateRankClaim_IsRejected_FirstClaimantProceeds()
    {
        int port = FreePort();
        Task<byte[]> root = Join(Opts(0, 3, port));
        Task<byte[]> claimA = Join(Opts(1, 3, port));
        Task<byte[]> claimB = Join(Opts(1, 3, port));

        // The accepted claimant cannot finish until rank 2 joins, so the first task to finish is the rejected duplicate.
        Task<byte[]> firstFinished = await Task.WhenAny(claimA, claimB);
        Assert.True(firstFinished.IsFaulted);
        Assert.Contains("already joined", firstFinished.Exception!.InnerException!.Message);
        Task<byte[]> accepted = ReferenceEquals(firstFinished, claimA) ? claimB : claimA;

        Task<byte[]> rank2 = Join(Opts(2, 3, port));
        byte[][] results = await Task.WhenAll(root, accepted, rank2);
        Assert.Equal(RootPayload, results[1]);
        Assert.Equal(RootPayload, results[2]);
    }

    [Fact]
    public async Task MissingRank_TimesOutOnRoot_NamingTheMissingRank()
    {
        int port = FreePort();
        Task<byte[]> root = Join(Opts(0, 3, port, timeoutSeconds: 2));
        Task<byte[]> rank1 = Join(Opts(1, 3, port));

        TimeoutException timeout = await Assert.ThrowsAsync<TimeoutException>(() => root);
        Assert.Contains("ranks [2]", timeout.Message);

        // Rank 0 closes its sockets at the deadline, so an admitted client fails rather than hanging.
        await Assert.ThrowsAsync<InvalidOperationException>(() => rank1);
    }

    [Fact]
    public async Task ExternalCancellation_SurfacesAsCancellation_NotTimeout()
    {
        int port = FreePort();
        using CancellationTokenSource cts = new();
        Task<byte[]> root = Join(Opts(0, 2, port), cts.Token);
        cts.CancelAfter(300);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => root);
    }

    [Fact]
    public void InvalidOptions_AreRejectedBeforeAnyNetworkUse()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Opts(0, 1, 1234).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => Opts(2, 2, 1234).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => Opts(1, 2, 0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => Opts(1, 2, 70000).Validate());
        Assert.Throws<ArgumentException>(() => Opts(1, 2, 1234, token: "").Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => (Opts(1, 2, 1234) with { Timeout = TimeSpan.Zero }).Validate());
    }
}
