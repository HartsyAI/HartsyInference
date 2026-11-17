namespace HartsyInference.Audio.Models.Haru;

/// <summary>Everything a recorded conversation produced: the model's audio (as long as the user's, 24 kHz mono), the user's audio padded to
/// whole frames, the text the model spoke, and the frames.</summary>
internal sealed record HaruReply(float[] ModelAudio, float[] UserAudio, string Text, IReadOnlyList<HaruFrame> Frames);

/// <summary>Plays a recording of the user to the model frame by frame, as in a live conversation, and collects what it answers.</summary>
internal static class HaruConversation
{
    /// <summary>Answers <paramref name="userAudio"/> (mono 24 kHz). <paramref name="reference"/> is the raw voice clip, or empty;
    /// <paramref name="tailFrames"/> extra frames of silence let the model finish speaking after the recording ends. The reply is as long as
    /// the padded recording: model frame t is heard while the user speaks frame t, so the last user frame needs no step.</summary>
    internal static HaruReply Respond(HaruDuplexSession session, HaruConfig config, string card, ReadOnlySpan<float> reference,
        ReadOnlySpan<float> userAudio, int tailFrames = 0)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentOutOfRangeException.ThrowIfNegative(tailFrames);
        int frame = config.FrameSamples;
        int count = Math.Max(1, (userAudio.Length + frame - 1) / frame) + tailFrames;
        float[] user = new float[count * frame];
        userAudio.CopyTo(user);

        float[] prepared = HaruAudio.PrepareReference(reference, config.SampleRate);
        List<HaruFrame> frames = [session.Start(card, prepared)];
        for (int i = 0; i < count - 1; i++)
        {
            frames.Add(session.Step(user.AsSpan(i * frame, frame)));
        }
        float[] reply = new float[count * frame];
        for (int i = 0; i < frames.Count; i++)
        {
            frames[i].Audio.CopyTo(reply, i * frame);
        }
        string text = string.Concat(frames.Select(static f => f.Text)).Trim();
        return new HaruReply(reply, user, text, frames);
    }
}
