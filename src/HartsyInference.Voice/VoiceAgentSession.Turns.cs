using System.Text;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Runtime;
using HartsyInference.Engine.Requests;
using HartsyInference.Tools;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Gpu;
using HartsyInference.Voice.Turns;

namespace HartsyInference.Voice;

public sealed partial class VoiceAgentSession
{
    private const double SamplesPerMs = VoiceAudioFrontend.SampleRate / 1000.0;

    private async Task RunTurnsAsync()
    {
        CancellationToken ending = _ending.Token;
        try
        {
            while (await _inputs.Reader.WaitToReadAsync(ending).ConfigureAwait(false))
            {
                while (_inputs.Reader.TryRead(out VoiceTurnInput? input))
                {
                    await RunTurnAsync(input, ending).ConfigureAwait(false);
                }
            }
        }
        catch (OperationCanceledException) when (ending.IsCancellationRequested)
        {
            Logs.Debug("[Voice] Turn loop stopped: the session ended.");
        }
    }

    private async Task RunTurnAsync(VoiceTurnInput input, CancellationToken ending)
    {
        int turnId = Interlocked.Increment(ref _turnCounter);
        // Never disposed: the audio thread may cancel it while the turn is finishing, and it owns no timer or parent
        // link that disposing would release (the session link is a registration, disposed below).
        CancellationTokenSource cancellation = new();
        VoiceTurnHandle handle = new(turnId, cancellation);
        Turn turn = new(this, input, turnId, cancellation.Token);
        Exception? failure = null;
        using (CancellationTokenRegistration link = ending.UnsafeRegister(static s => ((CancellationTokenSource)s!).Cancel(), cancellation))
        {
            _signals.BeginTurn(handle);
            try
            {
                await turn.RunAsync().ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellation.IsCancellationRequested)
            {
                Logs.Debug($"[Voice] Turn {turnId} cancelled.");
            }
            catch (Exception ex)
            {
                failure = ex;
                Logs.Error($"[Voice] Turn {turnId} failed.", ex);
                Emit(new VoiceAgentEvent { Kind = VoiceAgentEventKind.Error, TurnId = turnId, Text = ex.Message, Error = ex, TimestampNs = MonotonicClock.NowNs() });
            }
            finally
            {
                _signals.EndTurn(handle);
            }
        }
        bool bargedIn = _signals.BargedInTurn == turnId;
        if (bargedIn && !ending.IsCancellationRequested)
        {
            try
            {
                // The turn is over once its audio has actually stopped; the next turn would wait for this anyway.
                await _outbound.WaitFlushesAppliedAsync(ending).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (ending.IsCancellationRequested)
            {
                Logs.Debug($"[Voice] Session ended before turn {turnId}'s barge-in flush was applied.");
            }
        }
        turn.Finish(bargedIn);
        if (failure is not null)
        {
            input.Completion?.TrySetException(failure);
        }
        else if (ending.IsCancellationRequested && !bargedIn)
        {
            input.Completion?.TrySetCanceled(ending);
        }
        else
        {
            input.Completion?.TrySetResult();
        }
        _models.Gpu.RequestTrim();
        if (!ending.IsCancellationRequested)
        {
            SetState(VoiceAgentState.Listening, turnId);
        }
    }

    private TextRequest BuildRequest() => new()
    {
        Messages = _conversation.ToRequest(content => _text.CountTokens(_llm, content), _options.MaxHistoryTokens),
        Device = _options.LlmDevice,
        EnableThinking = false,
        MaxTokens = _options.MaxReplyTokens,
        Tools = _tools.Count > 0 ? _tools.Definitions : null,
        AlwaysFreeMemory = false,
    };

    private void AddUserTurn(int turnId, string text)
    {
        _conversation.AddUser(text);
        AddTranscript(turnId, TextRole.User, text, interrupted: false);
        Emit(VoiceAgentEventKind.UserTranscript, turnId, text);
    }

    private static bool HasWords(string text)
    {
        foreach (char c in text)
        {
            if (char.IsLetterOrDigit(c))
            {
                return true;
            }
        }
        return false;
    }

    private static async IAsyncEnumerable<string> Once(string text)
    {
        yield return text;
        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>One turn: recognition, the model's reply with its tool rounds, and the reply's audio.</summary>
    /// <remarks>The reply's text is read by the sentence synthesizer's producer task while this turn's own task writes
    /// the audio, so the fields each side writes are disjoint; everything is read back in <see cref="Finish"/>, after
    /// the producer has completed.</remarks>
    private sealed class Turn(VoiceAgentSession session, VoiceTurnInput input, int id, CancellationToken token)
    {
        private readonly StringBuilder _round = new();
        private readonly StringBuilder _reply = new();
        private readonly List<NativeToolCall> _pendingCalls = [];
        private VoiceTurnOutput? _output;
        private bool _speaking;
        private bool _userAdded;
        private double? _sttMs;
        private long _llmStartNs;
        private long _ttftNs;
        private long _firstSentenceNs;
        private long _firstAudioNs;
        private int _toolCalls;

        public async Task RunAsync()
        {
            session.SetState(VoiceAgentState.Thinking, id);
            IAsyncEnumerable<string> reply;
            if (input.Kind == VoiceTurnKind.Speak)
            {
                string prompt = input.Text!;
                _round.Append(prompt);
                _reply.Append(prompt);
                reply = Once(prompt);
            }
            else
            {
                string heard = input.Kind == VoiceTurnKind.Utterance ? await TranscribeAsync().ConfigureAwait(false) : input.Text!;
                if (!HasWords(heard))
                {
                    session.CountDiscarded(id, heard.Length == 0 ? "the recognizer heard no words" : $"the recognizer heard no words (\"{heard}\")");
                    return;
                }
                session.AddUserTurn(id, heard);
                _userAdded = true;
                reply = ReplyAsync(token);
            }
            await PlayAsync(reply).ConfigureAwait(false);
        }

        /// <summary>Records the reply in the conversation and transcript and logs the turn's metrics.</summary>
        public void Finish(bool interrupted)
        {
            if (_userAdded || input.Kind == VoiceTurnKind.Speak)
            {
                string last = _round.ToString().Trim();
                if (last.Length > 0)
                {
                    session._conversation.AddAssistant(last);
                }
            }
            string reply = _reply.ToString().Trim();
            if (reply.Length > 0)
            {
                session.AddTranscript(id, TextRole.Assistant, reply, interrupted);
                session.Emit(VoiceAgentEventKind.AssistantTranscript, id, reply);
            }
            VoiceTurnMetrics metrics = Metrics(interrupted);
            Logs.Info(metrics.ToLogLine());
            session.Emit(new VoiceAgentEvent { Kind = VoiceAgentEventKind.TurnCompleted, TurnId = id, Metrics = metrics, TimestampNs = MonotonicClock.NowNs() });
        }

        private async Task<string> TranscribeAsync()
        {
            float[] audio = input.Audio!;
            VoiceModelSet models = session._models;
            long start = MonotonicClock.NowNs();
            string text = await models.Gpu.RunAsync(VoiceGpuJobKind.Transcribe, () => models.Transcribe(audio), token).ConfigureAwait(false);
            _sttMs = Ms(MonotonicClock.NowNs() - start);
            return text.Trim();
        }

        /// <summary>The model's reply text as it streams, with the tool rounds recorded into the conversation on the way.</summary>
        private async IAsyncEnumerable<string> ReplyAsync([EnumeratorCancellation] CancellationToken cancel)
        {
            TextRequest request = session.BuildRequest();
            _llmStartNs = MonotonicClock.NowNs();
            IAsyncEnumerable<TextChunk> chunks = ToolLoop.RunAsync(session._text, session._llm, request, session._tools, session._options.MaxToolRoundsPerTurn, cancel);
            await foreach (TextChunk chunk in chunks.WithCancellation(cancel).ConfigureAwait(false))
            {
                switch (chunk.Kind)
                {
                    case TextChunkKind.Chunk when !string.IsNullOrEmpty(chunk.Text):
                        if (_ttftNs == 0)
                        {
                            _ttftNs = MonotonicClock.NowNs();
                        }
                        _round.Append(chunk.Text);
                        _reply.Append(chunk.Text);
                        yield return chunk.Text;
                        break;
                    case TextChunkKind.NativeToolCall when chunk.ToolCall is { } call:
                        _pendingCalls.Add(call);
                        _toolCalls++;
                        session.SetState(VoiceAgentState.ToolRunning, id);
                        session.Emit(VoiceAgentEventKind.ToolCall, id, call: call);
                        break;
                    case TextChunkKind.Status when chunk.Status is { Phase: ToolLoop.ToolResultPhase } && chunk.ToolCall is { } done:
                        string text = chunk.Text ?? "";
                        OnToolResult(done, text.StartsWith(ToolLoop.ToolResultPrefix, StringComparison.Ordinal) ? text[ToolLoop.ToolResultPrefix.Length..] : text);
                        break;
                    case TextChunkKind.Status when chunk.Status is { Phase: ToolLoop.RoundLimitPhase }:
                        Logs.Warning($"[Voice] Turn {id} reached {session._options.MaxToolRoundsPerTurn} model rounds; its last tool call was not run.");
                        break;
                    case TextChunkKind.StopReason when chunk.Stop is StopReason.Error or StopReason.Cancelled:
                        cancel.ThrowIfCancellationRequested();
                        throw new InvalidOperationException($"The language model stopped with {chunk.Stop}: {chunk.Text}");
                }
            }
        }

        private void OnToolResult(NativeToolCall call, string result)
        {
            if (_pendingCalls.Count > 0)
            {
                // The round's text and every call it made go in as one assistant turn ahead of the first result, the
                // shape ToolLoop itself sends back to the model.
                session._conversation.AddAssistant(_round.ToString(), [.. _pendingCalls]);
                _round.Clear();
                _pendingCalls.Clear();
            }
            session._conversation.AddToolResult(call, result);
            session.Emit(VoiceAgentEventKind.ToolResult, id, result, call);
            session.SetState(session._signals.SpeakingTurn == id ? VoiceAgentState.Speaking : VoiceAgentState.Thinking, id);
        }

        private async Task PlayAsync(IAsyncEnumerable<string> reply)
        {
            VoiceModelSet models = session._models;
            _output = new VoiceTurnOutput(session._outbound, models.SynthesisSampleRate, session._options.OutboundSampleRate);
            IAsyncEnumerable<AudioChunk> audio = SentenceChunkedSynthesis.StreamFromDeltas(reply, models.SynthesisSampleRate, Synthesize, RunOnGpu,
                session._options.FirstSentenceMinChars, session._options.MaxSentenceChars, MaxSynthesisInFlight, token);
            await foreach (AudioChunk chunk in audio.ConfigureAwait(false))
            {
                if (_firstAudioNs == 0)
                {
                    _firstAudioNs = MonotonicClock.NowNs();
                }
                if (!_speaking)
                {
                    _speaking = true;
                    session._signals.BeginSpeaking(id);
                    session.SetState(VoiceAgentState.Speaking, id);
                }
                if (!await _output.WriteAsync(chunk.Samples, token).ConfigureAwait(false))
                {
                    break;
                }
            }
            token.ThrowIfCancellationRequested();
            if (!_speaking || _output.Superseded)
            {
                return;
            }
            long end = await _output.CompleteAsync(token).ConfigureAwait(false);
            await session._outbound.WaitPlayedAsync(end, token).ConfigureAwait(false);
            session._signals.EndSpeaking(id);
        }

        /// <summary>One sentence to audio; runs on the GPU thread. A piece with no words (a trailing "..." or dash) is
        /// skipped: the synthesizer produces no audio for it, which the lease reports as a failure.</summary>
        private float[] Synthesize(string sentence, CancellationToken cancel)
        {
            string speakable = SpokenTextNormalizer.ToSpeakable(sentence);
            return HasWords(speakable) ? session._models.Synthesize(speakable) : [];
        }

        /// <summary>The sentence synthesizer's scheduler seam: every sentence becomes one job on the GPU thread.</summary>
        private Task<float[]> RunOnGpu(Func<float[]> job, CancellationToken cancel)
        {
            Interlocked.CompareExchange(ref _firstSentenceNs, MonotonicClock.NowNs(), 0);
            return session._models.Gpu.RunAsync(VoiceGpuJobKind.Synthesize, job, cancel);
        }

        private VoiceTurnMetrics Metrics(bool interrupted)
        {
            bool utterance = input.Kind == VoiceTurnKind.Utterance;
            LatencyHistogram.Summary frames = input.FrameTimes;
            bool timedFrames = utterance && frames.Count > 0;
            double? hangoverMs = utterance ? input.HangoverSamples / SamplesPerMs : null;
            long firstWriteNs = _output?.FirstWriteNs ?? 0;
            long bargeInStopNs = interrupted ? session._outbound.LastDiscardNs - session._signals.BargeInNs : 0;
            return new VoiceTurnMetrics
            {
                TurnId = id,
                Kind = input.Kind,
                UtteranceMs = input.Audio is { } heard ? heard.Length / SamplesPerMs : null,
                EndpointMs = hangoverMs,
                FrontendP50Ms = timedFrames ? frames.P50Us / 1000.0 : null,
                FrontendP99Ms = timedFrames ? frames.P99Us / 1000.0 : null,
                FrontendMaxMs = timedFrames ? frames.MaxUs / 1000.0 : null,
                SttMs = _sttMs,
                LlmTtftMs = Between(_llmStartNs, _ttftNs),
                LlmFirstSentenceMs = _llmStartNs == 0 ? null : Between(_llmStartNs, _firstSentenceNs),
                TtsFirstChunkMs = Between(_firstSentenceNs, _firstAudioNs),
                TransportMs = Between(_firstAudioNs, firstWriteNs),
                TotalMs = firstWriteNs == 0 ? null : (hangoverMs ?? 0) + Ms(firstWriteNs - input.ReceivedNs),
                ToolCalls = _toolCalls,
                Interrupted = interrupted,
                BargeInStopMs = bargeInStopNs > 0 ? Ms(bargeInStopNs) : null,
                InboundDroppedSamples = session._audio.DroppedSamples,
                DiscardedUtterances = session.DiscardedUtterances,
            };
        }

        private static double? Between(long fromNs, long toNs) => fromNs == 0 || toNs == 0 ? null : Ms(toNs - fromNs);

        private static double Ms(long ns) => ns / 1e6;
    }
}
