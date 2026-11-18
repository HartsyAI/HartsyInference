namespace HartsyInference.Engine.Requests;

/// <summary>What a recorded conversation produced: the model's audio as long as the user's recording, the recording padded to whole frames,
/// the words the model spoke, and the time each frame took to compute.</summary>
public sealed record DuplexResult(float[] ModelAudio, float[] UserAudio, string Text, int SampleRate, int Frames, double MedianFrameSeconds,
    double MaxFrameSeconds)
{
    /// <summary>The conversation as a stereo WAV: the user on the left channel, the model on the right.</summary>
    public byte[] ToStereoWav() => Audio.AudioClipCodec.EncodeWav(UserAudio, ModelAudio, SampleRate);

    /// <summary>The model's side as a mono WAV.</summary>
    public byte[] ToModelWav() => Audio.AudioClipCodec.EncodeWav(ModelAudio, null, SampleRate);
}
