using HartsyInference.Engine.Services;
using HartsyInference.Tools;
using HartsyInference.Voice;

namespace HartsyInference.VoiceHost.Calls;

/// <summary>Sessions on the host's loaded model set, answering with <paramref name="text"/>; the models outlive every
/// call.</summary>
internal sealed class VoiceAgentCallFactory(VoiceModelSet models, ITextService text, VoiceAgentOptions options) : IVoiceCallSessionFactory
{
    public int OutboundSampleRate => options.OutboundSampleRate;

    public IVoiceCallSession Create(uint callId, ToolRegistry tools) => new VoiceAgentCallSession(new VoiceAgentSession(models, text, tools, options));
}
