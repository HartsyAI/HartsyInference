using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.Distributed;

/// <summary>Out-of-band TCP exchange of one opaque blob from rank 0 to every other rank, so multi-node jobs need no MPI, shared filesystem or launcher-supplied IDs.</summary>
/// <remarks>Each non-root rank sends a fixed hello (magic, protocol version, rank, world size, job-token digest). Rank 0 validates
/// each hello, holds the accepted connections until every rank has joined, then sends the blob to all of them. Plain TCP:
/// run on a private fabric network. A hello that stalls or fails is dropped and the wait continues; the deadline fails the
/// whole exchange and names the ranks still missing.</remarks>
// TODO(untested): only loopback is exercised (UniqueIdRendezvousTests); no multi-host run, IPv6 and dual-stack behaviour unverified.
// TODO(unsure): plain TCP with a shared token, no TLS. Security review needed before any non-private network use.
// TODO(unsure): IPAddress.Any bind on rank 0 may expose the port on every interface; consider binding a configured address.
public static class UniqueIdRendezvous
{
    private const int ProtocolVersion = 1;
    private const int MaxPayloadBytes = 1 << 16;
    private const int DigestBytes = 32;
    private const int HelloBytes = 4 + 4 + 4 + 4 + DigestBytes;
    private const byte StatusAccepted = 0;
    private const byte StatusRejected = 1;
    private static readonly byte[] Magic = "HRV1"u8.ToArray();
    // TODO(unsure): 10 s hello timeout and 64 KiB payload cap are chosen, not measured.
    private static readonly TimeSpan HelloTimeout = TimeSpan.FromSeconds(10);

    /// <summary>Joins the rendezvous and returns the blob rank 0 produced. Rank 0 calls <paramref name="createRootPayload"/> exactly once; other ranks never call it.</summary>
    public static async Task<byte[]> ExchangeAsync(DistributedRendezvousOptions options, Func<byte[]> createRootPayload, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(createRootPayload);
        options.Validate();
        using CancellationTokenSource deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(options.Timeout);
        try
        {
            return options.Rank == 0
                ? await HostAsync(options, createRootPayload, cancellationToken, deadline.Token).ConfigureAwait(false)
                : await JoinAsync(options, deadline.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Rendezvous at {options.MasterAddress}:{options.MasterPort} timed out after {options.Timeout} (rank {options.Rank} of {options.WorldSize}).");
        }
    }

    private static async Task<byte[]> HostAsync(DistributedRendezvousOptions options, Func<byte[]> createRootPayload, CancellationToken external, CancellationToken ct)
    {
        byte[] payload = createRootPayload();
        if (payload.Length == 0 || payload.Length > MaxPayloadBytes)
        {
            throw new InvalidOperationException($"Rendezvous payload must be 1..{MaxPayloadBytes} bytes, got {payload.Length}.");
        }
        byte[] expectedDigest = SHA256.HashData(Encoding.UTF8.GetBytes(options.JobToken));
        TcpClient?[] joined = new TcpClient?[options.WorldSize];
        TcpListener listener = new(IPAddress.Any, options.MasterPort);
        listener.Start();
        try
        {
            int joinedCount = 0;
            // TODO(unsure): hellos are handled sequentially, so a stalled client can hold the accept loop for up to HelloTimeout each.
            while (joinedCount < options.WorldSize - 1)
            {
                TcpClient? candidate = null;
                bool admitted = false;
                try
                {
                    candidate = await listener.AcceptTcpClientAsync(ct).ConfigureAwait(false);
                    int rank = await TryAcceptHelloAsync(candidate, options, expectedDigest, joined, ct).ConfigureAwait(false);
                    if (rank > 0)
                    {
                        joined[rank] = candidate;
                        admitted = true;
                        joinedCount++;
                        Logs.Info($"[Rendezvous] rank {rank} joined ({joinedCount + 1}/{options.WorldSize}).");
                    }
                }
                catch (OperationCanceledException) when (!external.IsCancellationRequested)
                {
                    throw MissingRanksTimeout(options, joined);
                }
                finally
                {
                    // Covers rejected hellos and any exit (caller cancel, I/O fault) before the client is admitted.
                    if (!admitted)
                    {
                        candidate?.Dispose();
                    }
                }
            }
            // TODO(unsure): if one rank drops after admission, the whole exchange fails; no re-admission path exists.
            for (int rank = 1; rank < options.WorldSize; rank++)
            {
                await WriteReplyAsync(joined[rank]!, StatusAccepted, payload, ct, rank).ConfigureAwait(false);
            }
            return payload;
        }
        finally
        {
            listener.Stop();
            foreach (TcpClient? client in joined)
            {
                client?.Dispose();
            }
        }
    }

    private static TimeoutException MissingRanksTimeout(DistributedRendezvousOptions options, TcpClient?[] joined)
    {
        List<int> missing = [];
        for (int rank = 1; rank < options.WorldSize; rank++)
        {
            if (joined[rank] is null)
            {
                missing.Add(rank);
            }
        }
        return new TimeoutException($"Rendezvous on port {options.MasterPort} timed out after {options.Timeout}; still waiting for ranks [{string.Join(", ", missing)}] of {options.WorldSize}.");
    }

    /// <summary>Reads and checks one hello. Returns the claimed rank when accepted, 0 when rejected or dropped.</summary>
    private static async Task<int> TryAcceptHelloAsync(TcpClient candidate, DistributedRendezvousOptions options, byte[] expectedDigest, TcpClient?[] joined, CancellationToken ct)
    {
        byte[] hello = new byte[HelloBytes];
        try
        {
            using CancellationTokenSource helloDeadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
            helloDeadline.CancelAfter(HelloTimeout);
            await candidate.GetStream().ReadExactlyAsync(hello, helloDeadline.Token).ConfigureAwait(false);
        }
        catch (IOException ex)
        {
            Logs.Warning($"[Rendezvous] dropped a connection that did not send a complete hello: {ex.Message}");
            return 0;
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            Logs.Warning("[Rendezvous] dropped a connection that stalled before its hello.");
            return 0;
        }

        if (!hello.AsSpan(0, Magic.Length).SequenceEqual(Magic.AsSpan()))
        {
            return await RejectAsync(candidate, "not a HartsyInference rendezvous client", ct);
        }
        int version = BinaryPrimitives.ReadInt32LittleEndian(hello.AsSpan(4));
        int rank = BinaryPrimitives.ReadInt32LittleEndian(hello.AsSpan(8));
        int worldSize = BinaryPrimitives.ReadInt32LittleEndian(hello.AsSpan(12));
        if (version != ProtocolVersion)
        {
            return await RejectAsync(candidate, $"protocol version {version} is not {ProtocolVersion}", ct);
        }
        if (worldSize != options.WorldSize)
        {
            return await RejectAsync(candidate, $"world size {worldSize} does not match {options.WorldSize}", ct);
        }
        if (rank <= 0 || rank >= options.WorldSize)
        {
            return await RejectAsync(candidate, $"rank {rank} is outside [1, {options.WorldSize})", ct);
        }
        if (!CryptographicOperations.FixedTimeEquals(hello.AsSpan(16, DigestBytes), expectedDigest))
        {
            Logs.Warning($"[Rendezvous] rejected rank {rank}: job token mismatch.");
            return await RejectAsync(candidate, "job token mismatch", ct);
        }
        // TODO(unsure): first claimant wins; a crashed-then-restarted rank that reconnects after its first claim is rejected until the job times out.
        if (joined[rank] is not null)
        {
            return await RejectAsync(candidate, $"rank {rank} already joined", ct);
        }
        return rank;
    }

    private static async Task<int> RejectAsync(TcpClient candidate, string reason, CancellationToken ct)
    {
        Logs.Warning($"[Rendezvous] rejected a join: {reason}.");
        try
        {
            await WriteReplyAsync(candidate, StatusRejected, Encoding.UTF8.GetBytes(reason), ct, rank: -1).ConfigureAwait(false);
        }
        catch (IOException)
        {
            // The client already went away; the rejection is logged above.
        }
        return 0;
    }

    private static async Task WriteReplyAsync(TcpClient client, byte status, byte[] body, CancellationToken ct, int rank)
    {
        byte[] header = new byte[5];
        header[0] = status;
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(1), body.Length);
        try
        {
            NetworkStream stream = client.GetStream();
            await stream.WriteAsync(header, ct).ConfigureAwait(false);
            await stream.WriteAsync(body, ct).ConfigureAwait(false);
            await stream.FlushAsync(ct).ConfigureAwait(false);
        }
        catch (IOException ex) when (rank > 0)
        {
            throw new InvalidOperationException($"Rendezvous lost rank {rank} while sending the job's shared payload: {ex.Message}", ex);
        }
    }

    private static async Task<byte[]> JoinAsync(DistributedRendezvousOptions options, CancellationToken ct)
    {
        using TcpClient client = await ConnectWithRetryAsync(options, ct).ConfigureAwait(false);
        NetworkStream stream = client.GetStream();
        await stream.WriteAsync(BuildHello(options), ct).ConfigureAwait(false);

        byte[] header = new byte[5];
        try
        {
            await stream.ReadExactlyAsync(header, ct).ConfigureAwait(false);
        }
        catch (IOException ex)
        {
            throw new InvalidOperationException($"Rendezvous at {options.MasterAddress}:{options.MasterPort} closed before rank {options.Rank} was admitted: {ex.Message}", ex);
        }
        int length = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(1));
        if (length < 0 || length > MaxPayloadBytes)
        {
            throw new InvalidOperationException($"Rendezvous sent an invalid payload length {length}.");
        }
        byte[] body = new byte[length];
        try
        {
            await stream.ReadExactlyAsync(body, ct).ConfigureAwait(false);
        }
        catch (IOException ex)
        {
            throw new InvalidOperationException($"Rendezvous closed mid-payload for rank {options.Rank}: {ex.Message}", ex);
        }
        if (header[0] != StatusAccepted)
        {
            throw new InvalidOperationException($"Rendezvous at {options.MasterAddress}:{options.MasterPort} rejected rank {options.Rank}: {Encoding.UTF8.GetString(body)}.");
        }
        return body;
    }

    // TODO(unsure): retries every SocketException until the deadline, including permanent ones such as a DNS typo; the final error is not surfaced.
    private static async Task<TcpClient> ConnectWithRetryAsync(DistributedRendezvousOptions options, CancellationToken ct)
    {
        TimeSpan delay = TimeSpan.FromMilliseconds(250);
        while (true)
        {
            TcpClient client = new();
            try
            {
                await client.ConnectAsync(options.MasterAddress, options.MasterPort, ct).ConfigureAwait(false);
                return client;
            }
            catch (SocketException)
            {
                client.Dispose();
            }
            catch (OperationCanceledException)
            {
                client.Dispose();
                throw;
            }
            await Task.Delay(delay, ct).ConfigureAwait(false);
            delay = TimeSpan.FromTicks(Math.Min(delay.Ticks * 2, TimeSpan.FromSeconds(2).Ticks));
        }
    }

    private static byte[] BuildHello(DistributedRendezvousOptions options)
    {
        byte[] hello = new byte[HelloBytes];
        Magic.CopyTo(hello, 0);
        BinaryPrimitives.WriteInt32LittleEndian(hello.AsSpan(4), ProtocolVersion);
        BinaryPrimitives.WriteInt32LittleEndian(hello.AsSpan(8), options.Rank);
        BinaryPrimitives.WriteInt32LittleEndian(hello.AsSpan(12), options.WorldSize);
        SHA256.HashData(Encoding.UTF8.GetBytes(options.JobToken)).CopyTo(hello, 16);
        return hello;
    }
}
