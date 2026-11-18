namespace HartsyInference.Engine.Requests;

/// <summary>One frame of the model in a full-duplex conversation: its audio (mono, <see cref="Services.IDuplexSpeechSession.SampleRate"/>, plus or
/// minus one) and the text it emitted in this frame, which is empty while it is silent.</summary>
public sealed record DuplexFrame(float[] Audio, string Text, int Index, double ComputeSeconds, bool Done);
