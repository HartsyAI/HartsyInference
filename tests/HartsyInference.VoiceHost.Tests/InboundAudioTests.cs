using HartsyInference.PhoneLink;
using HartsyInference.VoiceHost.Tests.Support;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>Caller audio from the link into the session: PCM16 becomes ±1 floats at the 1/32768 scale the front end
/// expects, concealed frames are still audio, nothing reaches a session that has not started, and the framing rules hold
/// (a sequence gap or a short <c>InboundAudio</c> is a protocol error that closes the link).</summary>
public sealed class InboundAudioTests
{
    [Fact]
    public async Task CallerAudioReachesTheSessionAsFloatsOnThePcm16Scale()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        short[] first = Frame(i => (short)(i * 100 - 16_000));
        short[] second = Frame(i => i % 2 == 0 ? short.MaxValue : short.MinValue);
        gateway.SendInbound(1, first);
        gateway.SendInbound(1, second, concealed: true);

        Assert.True(HostRig.Wait(() => session.InboundSamples.Length == 2 * LinkProtocol.InboundFrameSamples));
        float[] heard = session.InboundSamples;
        for (int i = 0; i < LinkProtocol.InboundFrameSamples; i++)
        {
            Assert.Equal(first[i] / 32768f, heard[i]);
            Assert.Equal(second[i] / 32768f, heard[LinkProtocol.InboundFrameSamples + i]);
        }
        Assert.Equal(-1f, heard.Min());
        Assert.Equal(2, rig.Call(1)!.InboundFrames);
    }

    [Fact]
    public async Task AudioBeforeTheSessionHasStartedIsDroppedAndCounted()
    {
        await using HostRig rig = HostRig.Start();
        FakeGateway gateway = rig.Connect();
        TaskCompletionSource gate = new(TaskCreationOptions.RunContinuationsAsynchronously);
        rig.Factory.StartGate = gate.Task;
        gateway.SendCallStart(1);
        Assert.True(HostRig.Wait(() => rig.Call(1) is not null && rig.Factory.Sessions.Count == 1));
        gateway.SendInbound(1, Frame(_ => 1000));
        gateway.SendInbound(1, Frame(_ => 1000));
        Assert.True(HostRig.Wait(() => rig.Call(1)!.InboundBeforeStart == 2));

        gate.SetResult();
        rig.WaitUntilReady(1);
        gateway.SendInbound(1, Frame(_ => 2000));
        FakeCallSession session = rig.Factory.Sessions[0];
        Assert.True(HostRig.Wait(() => session.InboundSamples.Length == LinkProtocol.InboundFrameSamples));
        Assert.All(session.InboundSamples, sample => Assert.Equal(2000 / 32768f, sample));
    }

    [Fact]
    public async Task AudioForACallTheHostDoesNotHaveIsIgnored()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        gateway.SendInbound(9, Frame(_ => 1000));
        gateway.SendInbound(1, Frame(_ => 3000));

        Assert.True(HostRig.Wait(() => session.InboundSamples.Length == LinkProtocol.InboundFrameSamples));
        Assert.All(session.InboundSamples, sample => Assert.Equal(3000 / 32768f, sample));
        Assert.False(gateway.IsClosed);
    }

    [Fact]
    public async Task ASequenceGapClosesTheLink()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, FakeCallSession session) = await rig.StartCallAsync();
        gateway.SendWithRestartedSequence(w => w.WriteInboundAudioAsync(1, Frame(_ => 1000), concealed: false, CancellationToken.None));

        Assert.True(gateway.WaitForClose(), "the host kept reading after a sequence gap.");
        Assert.True(HostRig.Wait(() => session.Disposed));
        Assert.Empty(session.InboundSamples);
    }

    [Fact]
    public async Task AShortInboundFrameClosesTheLink()
    {
        await using HostRig rig = HostRig.Start();
        (FakeGateway gateway, _) = await rig.StartCallAsync();
        gateway.Send(w => w.WriteAsync(LinkMessageType.InboundAudio, LinkFrameFlags.None, 1, new byte[100], CancellationToken.None));

        Assert.True(gateway.WaitForClose(), "the host accepted an InboundAudio frame that is not 20 ms.");
    }

    private static short[] Frame(Func<int, short> sample)
    {
        short[] frame = new short[LinkProtocol.InboundFrameSamples];
        for (int i = 0; i < frame.Length; i++)
        {
            frame[i] = sample(i);
        }
        return frame;
    }
}
