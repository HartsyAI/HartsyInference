using System.Diagnostics;
using System.Runtime.CompilerServices;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Pipelines;

/// <summary>CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`) non-streaming text-to-speech pipeline. Wires the
/// four-stage path: Qwen2.5-0.5B text→speech-token LM → chunk-aware conditional flow matching
/// (speech-token→mel) → HiFTNet vocoder (mel→24 kHz). For zero-shot cloning the reference clip's mel is
/// run through the S3 tokenizer (prompt speech tokens) and CAM++ (192-d speaker embedding); for
/// preset-voice / precomputed-embedding modes those are passed in directly and the input-side encoders
/// can be omitted.
///
/// <para>Phoneme/text tokenization is the caller's responsibility — <see cref="Synthesize"/> takes Qwen
/// BPE token IDs, matching the token-IDs-in convention of the other HartsyInference audio pipelines.
/// Streaming (the 5:15 text:speech interleave + per-chunk flush) is a follow-up; this pipeline runs the
/// non-streaming "all text, then synthesize" format.</para></summary>
public sealed class CosyVoicePipeline : IDisposable
{
    private readonly CosyVoiceConfig _cfg;
    private readonly CosyVoiceQwenLm _lm;
    private readonly CosyVoiceFlow _flow;
    private readonly HiFTNetVocoder _vocoder;
    private readonly CamPlusSpeakerEncoder? _speakerEncoder;
    private readonly S3Tokenizer? _s3;
    private bool _preloaded;
    private int _disposed;

    public CosyVoicePipeline(CosyVoiceConfig cfg, CosyVoiceQwenLm lm, CosyVoiceFlow flow,
        HiFTNetVocoder vocoder, CamPlusSpeakerEncoder? speakerEncoder = null, S3Tokenizer? s3 = null)
    {
        _cfg = cfg;
        _lm = lm;
        _flow = flow;
        _vocoder = vocoder;
        _speakerEncoder = speakerEncoder;
        _s3 = s3;
    }

    /// <summary>Synthesizes 24 kHz audio for the given Qwen text token IDs.
    /// <list type="bullet">
    ///   <item><b>Precomputed-speaker mode:</b> pass <paramref name="speakerEmbed"/> (<c>[1, 192]</c>);
    ///         leave <paramref name="referenceAudio"/> empty.</item>
    ///   <item><b>Zero-shot mode:</b> pass the raw <paramref name="referenceAudio"/> samples at
    ///         <paramref name="referenceSampleRate"/> + <paramref name="referenceTextTokens"/>. The pipeline
    ///         derives the THREE distinct features each input encoder needs — a 128-bin Whisper log-mel @16 kHz
    ///         for the S3 speech tokenizer, an 80-bin Kaldi fbank (+CMN) @16 kHz for CAM++, and an 80-bin
    ///         matcha mel @24 kHz for the flow's reference conditioning — rather than reusing one mel for all
    ///         three. Requires both input encoders to be attached.</item>
    /// </list></summary>
    public float[] Synthesize(IBackend backend,
        int[] textTokenIds,
        Tensor? speakerEmbed = null,
        ReadOnlySpan<float> referenceAudio = default,
        int referenceSampleRate = 0,
        int[]? referenceTextTokens = null,
        int seed = 0,
        int? chunkCausalSize = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        // Full-F32 GEMM across the whole pipeline. The Qwen speech-token LM is autoregressive, and TF32's ~1e-3
        // per-forward error accumulates over the decode loop and flips sampled argmaxes (verified on Zonos: TF32 →
        // babble, F32 → reference parity). Save/restore so we don't disturb the caller's backend state.
        // A layer-split LM runs each stage on its own backend, so every placement stage backend needs it too.
        List<(IBackend Backend, bool Prev)> saved = new(4) { (backend, backend.HighPrecisionGemm) };
        backend.HighPrecisionGemm = true;
        if (_lm.Placement is { IsSingle: false } lmPlacement)
        {
            foreach (LlmStage stage in lmPlacement.Stages)
            {
                IBackend sb = stage.Backend;
                bool seen = false;
                for (int i = 0; i < saved.Count && !seen; i++) seen = ReferenceEquals(saved[i].Backend, sb);
                if (seen) continue;
                saved.Add((sb, sb.HighPrecisionGemm));
                sb.HighPrecisionGemm = true;
            }
        }
        try
        {
            PreloadWeights(backend);
            return SynthesizeCore(backend, textTokenIds, speakerEmbed, referenceAudio, referenceSampleRate,
                referenceTextTokens, seed, sw, chunkCausalSize);
        }
        finally
        {
            for (int i = saved.Count - 1; i >= 0; i--) saved[i].Backend.HighPrecisionGemm = saved[i].Prev;
        }
    }

    /// <summary>Bulk-uploads every component's weights to the device once (idempotent — the GPU weight cache keys by
    /// tensor reference). Without this each Linear/Conv re-uploads its weight per call: the pre-preload profile was
    /// ~62k <c>H2D_MISS</c> transfers (~3.7 s) dominating the run. No-op on backends without a weight cache. The LM
    /// gets a per-stage asymmetric preload when <see cref="CosyVoiceQwenLm.Placement"/> is a layer-split (mirrors
    /// <c>YuePipeline.PreloadStage1</c>) — preloading the full set on every stage would replicate instead of pool.</summary>
    private void PreloadWeights(IBackend backend)
    {
        if (_preloaded) return;
        if (_lm.Placement is { IsSingle: false } placement)
        {
            for (int s = 0; s < placement.Stages.Count; s++)
            {
                LlmStage stage = placement.Stages[s];
                // Canonical set only (no redundant fused/split views) — matches YuE's resident-preload choice.
                stage.Backend.PreloadWeights(_lm.EnumerateStageWeights(
                    stage.StartLayer, stage.EndLayer, s == 0, s == placement.Stages.Count - 1,
                    includeRedundantSplits: false));
            }
            backend.PreloadWeights(EnumerateNonLmWeights());
        }
        else
        {
            backend.PreloadWeights(EnumerateWeights());
        }
        _preloaded = true;
    }

    private IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _lm.EnumerateWeights()) yield return t;
        foreach (Tensor t in EnumerateNonLmWeights()) yield return t;
    }

    private IEnumerable<Tensor> EnumerateNonLmWeights()
    {
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _vocoder.EnumerateWeights()) yield return t;
        if (_speakerEncoder is not null)
            foreach (Tensor t in _speakerEncoder.EnumerateWeights()) yield return t;
        if (_s3 is not null)
            foreach (Tensor t in _s3.EnumerateWeights()) yield return t;
    }

    private float[] SynthesizeCore(IBackend backend,
        int[] textTokenIds,
        Tensor? speakerEmbed,
        ReadOnlySpan<float> referenceAudio,
        int referenceSampleRate,
        int[]? referenceTextTokens,
        int seed,
        Stopwatch sw,
        int? chunkCausalSize = null)
    {
        int[] promptSpeechTokens = [];
        Tensor? promptMel = null;
        Tensor spk;
        bool ownsSpk = false;

        if (!referenceAudio.IsEmpty)
        {
            if (referenceSampleRate <= 0)
                throw new ArgumentException("referenceSampleRate must be set when referenceAudio is provided.", nameof(referenceSampleRate));
            if (_s3 is null || _speakerEncoder is null)
                throw new InvalidOperationException("Zero-shot mode requires both an S3Tokenizer and a CamPlusSpeakerEncoder.");

            // Each input encoder was trained on a different acoustic feature; compute them independently.
            float[] audio16k = S3GenReference.Resample(referenceAudio, referenceSampleRate, 16_000);
            float[] audio24k = S3GenReference.Resample(referenceAudio, referenceSampleRate, 24_000);

            promptSpeechTokens = S3GenReference.SpeechTokens(backend, _s3, audio16k);
            spk = S3GenReference.SpeakerEmbedding(backend, _speakerEncoder, audio16k);
            ownsSpk = true;
            promptMel = S3GenReference.FlowMel(audio24k);                             // [1, 80, T]

            Logs.Info($"CosyVoice: reference → {promptSpeechTokens.Length} prompt speech tokens + speaker embedding.");
        }
        else if (speakerEmbed is not null)
        {
            spk = speakerEmbed;
        }
        else
        {
            throw new ArgumentException("Provide either speakerEmbed (precomputed) or referenceAudio (zero-shot).");
        }

        int[] refText = referenceTextTokens ?? [];
        List<int> speechTokens = _lm.GenerateSpeechTokens(backend, textTokenIds, refText, promptSpeechTokens, seed: seed);
        Logs.Info($"CosyVoice: LM emitted {speechTokens.Count} speech tokens in {sw.ElapsedMilliseconds}ms.");

        Tensor mel = _flow.Inference(backend, speechTokens.ToArray(), promptSpeechTokens, promptMel, spk, seed, chunkCausalSize);
        if (ownsSpk) spk.Dispose();
        promptMel?.Dispose();

        float[] audio = _vocoder.Forward(backend, mel);
        mel.Dispose();

        sw.Stop();
        Logs.Info($"CosyVoice synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Streaming counterpart to <see cref="Synthesize"/>: runs the identical LM→flow→vocoder pipeline
    /// on a background thread, but hands each newly-settled PCM chunk to the caller as soon as it's ready
    /// instead of buffering to the end. Mirrors <c>VibeVoicePipeline.SynthesizeStream</c>'s shape exactly
    /// (producer on <see cref="Task.Run(Action)"/>, consumer drains an <see cref="AudioStreamer"/>, the
    /// producer <see cref="Task"/> is awaited in a <c>finally</c> so a fault always surfaces to the caller).
    ///
    /// <para>Zero-shot mode ONLY (<paramref name="referenceAudio"/> required) — unlike <see cref="Synthesize"/>,
    /// which also accepts a precomputed <paramref name="speakerEmbed"/> with no reference clip.
    /// <see cref="CosyVoiceFlow.InferenceGrowingWindowed"/>'s <c>cond</c>-pinning design (the mechanism that
    /// avoids the exposure-bias drift a self-conditioned chunked design suffers — see that method's doc
    /// comment) needs a REAL reference mel to pin every windowed call to; this matches how CosyVoice2 is
    /// actually registered in the engine catalog (<c>CosyVoiceModel.cs</c> already requires a reference clip
    /// for the non-streaming path too, so this isn't a new restriction in practice).</para>
    ///
    /// <para><paramref name="chunkSizeTokens"/>/<paramref name="windowSizeTokens"/>/<paramref name="marginFrames"/>
    /// default to the best config found by a 2026-08-11 parameter sweep: 3.45× real-time, bounded (flat, not
    /// growing with utterance length) rather than real-time. ONE known, quantified, ACCEPTED limitation: an
    /// isolated single-word mispronunciation under adversarial content, present identically across every
    /// bounded window/margin/chunk-size combination tested (only a full-history unbounded design avoids it,
    /// at prohibitive cost — see <see cref="CosyVoiceFlow.InferenceGrowingWindowed"/>'s doc comment for the
    /// full 5-way sweep and the decisive discriminating experiment). Callers needing a different
    /// quality/latency/cost tradeoff can override these.</para>
    ///
    /// <para><b>The 3.45× figure above is flow+vocoder ONLY — verified live through the actual running
    /// service, real end-to-end (including LM speech-token generation) runs ~8× real-time, not 3.45×</b>
    /// (2026-08-11, two live WebSocket calls through `swarmui.service`'s real `GenerateText2ImageWS`
    /// endpoint with `streamchunksize: sentence`, a real zero-shot reference clip, real Whisper-verified
    /// correct content, zero errors, 8-9 real incremental chunks delivered before the final result each
    /// time). Per-chunk interval was a steady ~7.6-8.2s live, vs. the ~2.8-3.0s the flow+vocoder-only
    /// scratch harnesses measured in isolation — the LM's own autoregressive token decode for each
    /// <see cref="SynthesizeStreamCore"/>'s chunk-sized token batch is a REAL, substantial additional cost
    /// this pipeline's own tuning sweep never isolated (those harnesses called
    /// <c>GenerateSpeechTokens</c> once, up front, outside the timed loop — this method calls it inline,
    /// as it must for genuine incremental token-arrival streaming). First-chunk latency was 36s cold
    /// (includes first-request model/weight loading) and 25.7s warm (second call, same session) — still
    /// real LM-decode-dominated latency, not a bug, but the honest number for anyone deciding whether this
    /// meets a real-time bar: it does not, today, end to end. The chunked delivery itself is correct and
    /// real (content, boundaries, no errors) — this is a latency finding, not a correctness one.</para></summary>
    public async IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend,
        int[] textTokenIds,
        float[]? referenceAudio,
        int referenceSampleRate,
        int[]? referenceTextTokens = null,
        int seed = 0,
        int chunkSizeTokens = 25,
        int windowSizeTokens = 150,
        int marginFrames = 40,
        int maxTokens = 750,
        int? chunkCausalSize = null,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        if (referenceAudio is null || referenceAudio.Length == 0)
            throw new ArgumentException("CosyVoice 2 streaming is zero-shot — provide referenceAudio.", nameof(referenceAudio));

        List<(IBackend Backend, bool Prev)> saved = new(4) { (backend, backend.HighPrecisionGemm) };
        backend.HighPrecisionGemm = true;
        if (_lm.Placement is { IsSingle: false } lmPlacement)
        {
            foreach (LlmStage stage in lmPlacement.Stages)
            {
                IBackend sb = stage.Backend;
                bool seen = false;
                for (int i = 0; i < saved.Count && !seen; i++) seen = ReferenceEquals(saved[i].Backend, sb);
                if (seen) continue;
                saved.Add((sb, sb.HighPrecisionGemm));
                sb.HighPrecisionGemm = true;
            }
        }

        using AudioStreamer streamer = new();
        long sampleOffset = 0;

        Task producer = Task.Run(() =>
        {
            try
            {
                PreloadWeights(backend);
                SynthesizeStreamCore(backend, textTokenIds, referenceAudio, referenceSampleRate,
                    referenceTextTokens, seed, chunkSizeTokens, windowSizeTokens, marginFrames, maxTokens,
                    chunkCausalSize,
                    onChunk: chunk =>
                    {
                        streamer.Put(new AudioChunk(chunk, _cfg.SampleRate, 1, sampleOffset), cancel).AsTask().GetAwaiter().GetResult();
                        sampleOffset += chunk.Length;
                    });
            }
            finally
            {
                streamer.Complete();
            }
        }, cancel);

        try
        {
            await foreach (AudioChunk chunk in streamer.ReadAllAsync(cancel).ConfigureAwait(false))
            {
                yield return chunk;
            }
        }
        finally
        {
            await producer.ConfigureAwait(false);
            for (int i = saved.Count - 1; i >= 0; i--) saved[i].Backend.HighPrecisionGemm = saved[i].Prev;
        }
    }

    /// <summary>The streaming generation loop: LM speech tokens arrive via <c>onToken</c>; every
    /// <paramref name="chunkSizeTokens"/> new tokens (plus a final flush once the LM stops), a windowed flow
    /// call (<see cref="CosyVoiceFlow.InferenceGrowingWindowed"/>) turns the newly-available tokens into mel,
    /// and the HIFT vocoder's own streaming state (<see cref="HiFTStreamState"/>) turns that into PCM.
    /// <paramref name="onChunk"/> is called synchronously, once per non-empty PCM chunk, on this method's own
    /// calling thread (the background producer thread from <see cref="SynthesizeStream"/>, never a request
    /// thread).</summary>
    private void SynthesizeStreamCore(IBackend backend,
        int[] textTokenIds, float[] referenceAudio, int referenceSampleRate,
        int[]? referenceTextTokens, int seed, int chunkSizeTokens, int windowSizeTokens, int marginFrames,
        int maxTokens, int? chunkCausalSize, Action<float[]> onChunk)
    {
        if (referenceSampleRate <= 0)
            throw new ArgumentException("referenceSampleRate must be set when referenceAudio is provided.", nameof(referenceSampleRate));
        if (_s3 is null || _speakerEncoder is null)
            throw new InvalidOperationException("Zero-shot streaming requires both an S3Tokenizer and a CamPlusSpeakerEncoder.");

        float[] audio16k = S3GenReference.Resample(referenceAudio, referenceSampleRate, 16_000);
        float[] audio24k = S3GenReference.Resample(referenceAudio, referenceSampleRate, 24_000);
        int[] promptSpeechTokens = S3GenReference.SpeechTokens(backend, _s3, audio16k);
        Tensor spk = S3GenReference.SpeakerEmbedding(backend, _speakerEncoder, audio16k);
        Tensor promptMel = S3GenReference.FlowMel(audio24k);
        Logs.Info($"CosyVoice (stream): reference → {promptSpeechTokens.Length} prompt speech tokens + speaker embedding.");

        int[] refText = referenceTextTokens ?? [];
        int melBins = _cfg.Flow.MelBins;
        int ratio = UpsampleConformerEncoder.TokenMelRatio;
        // Pre-drawn once at maxTokens scale, since the LM's actual final token count isn't known until it stops
        // (EOS-terminated) — DeterministicRng's sequential draw is prefix-stable, so drawing MORE than needed and
        // slicing the used prefix is bit-identical to drawing exactly the right amount (see ConditionalCfm.DrawFullNoise).
        int fullNoiseFrames = (promptSpeechTokens.Length + maxTokens) * ratio;
        Tensor fullNoise = ConditionalCfm.DrawFullNoise(melBins, fullNoiseFrames, seed);

        List<int> allTokens = new(256);
        int emittedFrames = 0;
        int lastFlushedCount = 0;
        using HiFTStreamState hiftState = new();

        void FlushChunk(bool isFinal)
        {
            int end = allTokens.Count;
            if (end == lastFlushedCount && !isFinal) return;
            int windowStart = Math.Max(0, end - windowSizeTokens);
            int[] windowTokens = allTokens.GetRange(windowStart, end - windowStart).ToArray();
            Tensor chunkMel = _flow.InferenceGrowingWindowed(backend, windowTokens, windowStart, promptSpeechTokens,
                promptMel, spk, seed, fullNoise, fullNoiseFrames, marginFrames, ref emittedFrames, isFinal, chunkCausalSize);
            lastFlushedCount = end;
            if (chunkMel.Shape[2] > 0)
            {
                float[] audio = _vocoder.ForwardStreaming(backend, chunkMel, hiftState, isFinal);
                if (audio.Length > 0) onChunk(audio);
            }
            chunkMel.Dispose();
        }

        _lm.GenerateSpeechTokens(backend, textTokenIds, refText, promptSpeechTokens, maxTokens: maxTokens, seed: seed,
            onToken: tok =>
            {
                allTokens.Add(tok);
                if (allTokens.Count - lastFlushedCount >= chunkSizeTokens) FlushChunk(isFinal: false);
            });
        FlushChunk(isFinal: true);

        fullNoise.Dispose();
        spk.Dispose();
        promptMel.Dispose();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        _flow.Dispose();
        _vocoder.Dispose();
        _speakerEncoder?.Dispose();
        _s3?.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CosyVoicePipeline));
    }
}
