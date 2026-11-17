using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>A live full-duplex conversation with a loaded speech-to-speech model, opened by
/// <see cref="ISpeechService.OpenDuplexAsync"/>. <see cref="Start"/> runs the prompt and returns the model's first frame; each
/// <see cref="Step"/> takes the user's next frame and returns the model's next, so the model stays one frame ahead and the frame
/// length is the only structural latency.</summary>
/// <remarks>
/// <list type="bullet">
/// <item>Use a session from one thread. Every call takes the device gate itself, so other services run between frames.</item>
/// <item>The model is pinned while the session is open, as for <see cref="ISynthesizerLease"/>: an engine release waits for the call in
/// flight, then revokes the session, which reports it as <see cref="ObjectDisposedException"/> on its next call.</item>
/// <item>One conversation per loaded model: opening a second while the first is open is refused.</item>
/// </list></remarks>
public interface IDuplexSpeechSession : IDisposable
{
    /// <summary>Sample rate of both streams, in Hz.</summary>
    int SampleRate { get; }

    /// <summary>Samples in one frame of either stream.</summary>
    int FrameSamples { get; }

    /// <summary>How many more <see cref="Step"/> calls the conversation can take before it is done: its frame limit or the model's
    /// position capacity, whichever comes first. Meaningful after <see cref="Start"/>, which is where the prompt takes its share.</summary>
    int RemainingFrames { get; }

    /// <summary>Runs the character card and the voice reference and returns the model's frame 0.</summary>
    DuplexFrame Start();

    /// <summary>Consumes the user's next <see cref="FrameSamples"/> samples (mono, <see cref="SampleRate"/>, plus or minus one) and
    /// returns the model's next frame.</summary>
    DuplexFrame Step(ReadOnlySpan<float> userFrame);
}
