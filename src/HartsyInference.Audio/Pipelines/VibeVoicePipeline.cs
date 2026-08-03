using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>VibeVoice multi-speaker TTS pipeline (non-streaming variants — 1.5B / 7B).
/// Orchestrates the full inference loop:
///
/// <list type="number">
///   <item><b>Prefill</b>: tokenize the multi-speaker prompt, encode each voice reference
///         through the acoustic VAE, splice the resulting 64-d latents into the LM's
///         input embeddings at <c>&lt;|vision_pad|&gt;</c> positions.</item>
///   <item><b>AR loop</b>: sample one logically-constrained token at a time
///         (<c>{speech_start, speech_end, speech_diffusion, eos}</c>) from the LM head.
///         For <c>speech_diffusion</c>, run a 20-step DPM-Solver denoise on a 64-d latent
///         conditioned on the LM's last hidden state, decode the latent to a 3 200-sample
///         audio chunk through the acoustic VAE, then build the next-step embed via
///         <c>acoustic_connector(latent) + semantic_connector(semantic_features)</c>.</item>
///   <item><b>Assembly</b>: concatenate all audio chunks → 24 kHz mono waveform.</item>
/// </list>
///
/// <para><b>Status:</b> full dual-stream implementation. A parallel negative (unconditional,
/// text/voice-masked) LM stream runs in lockstep with the positive stream; at each diffusion
/// token the head is evaluated on both conditions and combined as
/// <c>eps = uncond + cfg_scale*(cond - uncond)</c> (<see cref="VibeVoiceConfig.CfgScale"/>,
/// default 1.3). Token selection is greedy by default (<see cref="VibeVoiceConfig.DoSample"/>),
/// matching upstream <c>do_sample=False</c>. Voice-embedding splicing and the semantic-feedback
/// loop are implemented. Setting <c>CfgScale = 1</c> reverts to the single-stream path.</para></summary>
public sealed class VibeVoicePipeline : IDisposable
{
    private readonly VibeVoiceConfig _cfg;
    private readonly Qwen2Config _lmCfg;
    private readonly VibeVoiceTokenizer _tokenizer;
    private readonly VibeVoiceProcessor _processor;
    private readonly Qwen2Model _lm;
    private readonly VibeVoiceAcousticTokenizerModel _acoustic;
    private readonly VibeVoiceSemanticTokenizerModel? _semantic;
    private readonly SpeechConnector _acousticConnector;
    private readonly SpeechConnector? _semanticConnector;
    private readonly VibeVoiceDiffusionHead _diffusionHead;

    private float _speechScalingFactor;
    private float _speechBiasFactor;

    private bool _preloaded;
    private int _disposed;

    public VibeVoiceConfig Config => _cfg;

    private VibeVoicePipeline(
        VibeVoiceConfig cfg, Qwen2Config lmCfg,
        VibeVoiceTokenizer tokenizer, VibeVoiceProcessor processor,
        Qwen2Model lm, VibeVoiceAcousticTokenizerModel acoustic,
        VibeVoiceSemanticTokenizerModel? semantic, SpeechConnector acousticConnector,
        SpeechConnector? semanticConnector, VibeVoiceDiffusionHead diffusionHead,
        float scaling, float bias)
    {
        _cfg = cfg; _lmCfg = lmCfg;
        _tokenizer = tokenizer; _processor = processor;
        _lm = lm; _acoustic = acoustic; _semantic = semantic;
        _acousticConnector = acousticConnector; _semanticConnector = semanticConnector;
        _diffusionHead = diffusionHead;
        _speechScalingFactor = scaling;
        _speechBiasFactor = bias;
    }

    /// <summary>Loads VibeVoice-1.5B from the HartsyInference cache (downloads on first
    /// use). Returns a fully-wired pipeline ready to synthesize.</summary>
    public static async Task<VibeVoicePipeline> LoadAsync(CancellationToken ct = default)
    {
        string repoDir = AudioModelCache.GetRepoDirectory("microsoft/VibeVoice-1.5B", "tts");
        await Task.WhenAll(
            AudioModelCache.GetAsync("microsoft/VibeVoice-1.5B", "model-00001-of-00003.safetensors", category: "tts", ct: ct),
            AudioModelCache.GetAsync("microsoft/VibeVoice-1.5B", "model-00002-of-00003.safetensors", category: "tts", ct: ct),
            AudioModelCache.GetAsync("microsoft/VibeVoice-1.5B", "model-00003-of-00003.safetensors", category: "tts", ct: ct),
            AudioModelCache.GetAsync("microsoft/VibeVoice-1.5B", "model.safetensors.index.json", category: "tts", ct: ct),
            AudioModelCache.GetAsync("microsoft/VibeVoice-1.5B", "config.json", category: "tts", ct: ct))
            .ConfigureAwait(false);

        VibeVoiceConfig cfg = VibeVoiceConfig.V15B;
        Qwen2Config lmCfg = Qwen2Config.Qwen25_1_5B;

        SafeTensorsShardLoader loader = new();
        loader.LoadDirectory(repoDir);
        IReadOnlyDictionary<string, Tensor> weights = loader.GetAllTensors();

        VibeVoiceTokenizer tokenizer = new();
        VibeVoiceProcessor processor = new(tokenizer);

        Qwen2Model lm = new(lmCfg);
        lm.LoadWeights(weights, prefix: "model.language_model");

        VibeVoiceAcousticTokenizerModel acoustic = new(cfg.AcousticTokenizer, "model.acoustic_tokenizer");
        acoustic.LoadWeights(weights);

        VibeVoiceSemanticTokenizerModel? semantic = null;
        SpeechConnector? semanticConnector = null;
        if (cfg.SemanticTokenizer is not null)
        {
            semantic = new VibeVoiceSemanticTokenizerModel(cfg.SemanticTokenizer, "model.semantic_tokenizer");
            semantic.LoadWeights(weights);
            semanticConnector = new SpeechConnector(cfg.SemanticVaeDim, lmCfg.HiddenSize);
            semanticConnector.LoadWeights(weights, "model.semantic_connector");
        }

        SpeechConnector acousticConnector = new(cfg.AcousticVaeDim, lmCfg.HiddenSize);
        acousticConnector.LoadWeights(weights, "model.acoustic_connector");

        VibeVoiceDiffusionHead diffusionHead = new(cfg.DiffusionHead, "model.prediction_head");
        diffusionHead.LoadWeights(weights);

        // Scalars: latent normalization buffers, learned during training. Saved as 1-d
        // tensors of length 1.
        float scaling = ReadScalar(weights, "model.speech_scaling_factor");
        float bias = ReadScalar(weights, "model.speech_bias_factor");

        // Do NOT dispose the loader: the LM keeps its projection weights as borrowed bf16
        // mmap views (uploaded/dequanted on first use), so unmapping here segfaults the
        // first H2D copy. The views' keep-alive roots the mapping; file-backed pages are
        // evictable, so this pins no RAM. Audio submodules hold their own F32 copies.
        return new VibeVoicePipeline(cfg, lmCfg, tokenizer, processor, lm, acoustic,
            semantic, acousticConnector, semanticConnector, diffusionHead, scaling, bias);
    }

    /// <summary>Synthesizes audio for a multi-speaker script.
    /// <paramref name="lines"/> are speaker turns ("Speaker 0: Hello", "Speaker 1: Hi", …)
    /// or raw text (round-robin speaker assignment).
    /// <paramref name="voiceWavPaths"/> is one 24 kHz reference WAV per speaker.
    /// <paramref name="maxNewTokens"/> caps the AR loop length.</summary>
    public unsafe float[] Synthesize(IBackend backend, IReadOnlyList<string> lines,
        IReadOnlyList<string> voiceWavPaths, int maxNewTokens = 256, IProgress<int>? progress = null,
        float temperature = 0.95f, float topP = 0.95f, int seed = 0)
    {
        ThrowIfDisposed();
        PreloadWeights(backend);
        uint rng = HartsyInference.Audio.Dsp.DeterministicRng.Seed(seed);
        uint noiseRng = HartsyInference.Audio.Dsp.DeterministicRng.Seed(seed ^ 0x51ED270B);   // separate stream for per-frame diffusion noise
        VibeVoiceProcessor.PreparedPrompt prep = _processor.Prepare(lines, voiceWavPaths);
        int promptLen = prep.TokenIds.Length;

        // Cap the cache at prompt + new-tokens budget, rounded up a little to absorb the
        // diffusion-head's per-frame embed appends (one per emitted speech_diffusion token).
        int cacheCap = Math.Min(_lmCfg.MaxPositionEmbeddings, promptLen + maxNewTokens + 16);
        using IKvCache kvCache = _lm.CreateDecodeCache(cacheCap);

        // Negative (unconditional) LM stream for classifier-free guidance. Unlike the positive
        // stream it holds NO text/voice prefill — only a leading speech_start plus the diffusion
        // embeds of the current speech segment (reset at each speech_start). At every diffusion
        // token both streams' last hidden states condition the denoiser (see DenoiseLatent).
        // Only allocated when CFG is active (cfg_scale != 1).
        bool cfgActive = MathF.Abs(_cfg.CfgScale - 1f) > 1e-6f;
        using IKvCache? negKvCache = cfgActive ? _lm.CreateDecodeCache(cacheCap) : null;

        // ── Prefill ─────────────────────────────────────────────────────────
        // 1. Encode each voice prompt through the acoustic VAE → [1, N_i, 64] mean latents.
        Tensor[] voiceLatents = new Tensor[prep.Voices.Length];
        for (int i = 0; i < prep.Voices.Length; i++)
            voiceLatents[i] = EncodeVoicePcm(backend, prep.Voices[i].Pcm, prep.Voices[i].LatentCount);

        // 2. Apply the learned normalization: features = (latent + bias) * scaling.
        for (int i = 0; i < voiceLatents.Length; i++) NormalizeLatentInPlace(voiceLatents[i], _speechScalingFactor, _speechBiasFactor);

        // 3. Build the prefill embeds: lookup embed_tokens(token_id) by default,
        //    overwritten by acoustic_connector(voice_latent[k]) at speech-mask positions.
        using Tensor prefillEmbeds = BuildPrefillEmbeds(backend, prep, voiceLatents);
        foreach (Tensor v in voiceLatents) v.Dispose();

        // 4. Prefill the LM with these embeddings.
        using Tensor prefillHidden = _lm.ForwardEmbeds(backend, prefillEmbeds, batch: 1, t: promptLen,
            posStart: 0, kvCache);
        // prefillHidden shape: [1, promptLen, hidden]. We need the LAST-position hidden
        // state — that's the LM's read-out at the trailing <|vision_start|> cursor and is
        // used to condition the first diffusion step.

        // ── AR loop ─────────────────────────────────────────────────────────
        List<float[]> audioChunks = new();

        // Persistent streaming caches for the acoustic decoder and semantic encoder. These
        // thread the causal conv/transpose receptive-field state through every AR step,
        // mirroring upstream's `acoustic_tokenizer.decode(..., cache=acoustic_cache,
        // use_cache=True)` and `semantic_tokenizer.encode(..., cache=semantic_cache,
        // use_cache=True)`. Created ONCE here (not per frame) so per-frame conditioning stays
        // coherent — rebuilding them stateless each step is what caused the LM to drift and
        // never emit speech_end/eos. Single stream → sample index 0.
        using VibeVoiceTokenizerStreamingCache acousticCache = new();
        using VibeVoiceTokenizerStreamingCache? semanticCache = _semantic is not null ? new VibeVoiceTokenizerStreamingCache() : null;
        int[] sampleIndices = [0];

        // Initial conditioning hidden state for the (very rare) case where the very first
        // emitted token is a speech_diffusion. Slice last frame of prefillHidden.
        using Tensor _lastHiddenWarm = SliceLastFrame(prefillHidden, _lmCfg.HiddenSize);
        Tensor? lastHidden = CopyOf(_lastHiddenWarm);
        // Negative-stream conditioning hidden state. Lazily primed (by forwarding a lone
        // speech_start through negKvCache) at the first diffusion token after each reset.
        Tensor? negLastHidden = null;
        try
        {
            int prevToken = VibeVoiceTokenizer.SpeechStartTokenId;     // last token in the prompt template
            for (int step = 0; step < maxNewTokens; step++)
            {
                // Project the last hidden state to logits, mask to the constrained vocab,
                // sample (greedy for determinism in v1).
                using Tensor logits = _lm.ProjectLogits(backend, lastHidden!, batch: 1, t: 1);
                // Greedy once CFG is on (upstream do_sample=False); the temperature/top-p path
                // was a workaround for the missing-guidance ramble. DoSample restores stochastic.
                float effTemp = _cfg.DoSample ? temperature : 0f;
                int nextToken = SampleConstrained(logits, effTemp, topP, ref rng);
                progress?.Report(step);

                if (nextToken == VibeVoiceTokenizer.EndOfTextTokenId)
                    break;

                // speech_end: reset the per-speaker streaming state (upstream calls
                // acoustic_cache.set_to_zero / semantic_cache.set_to_zero here) so the next
                // segment starts from a clean receptive field.
                if (nextToken == VibeVoiceTokenizer.SpeechEndTokenId)
                {
                    acousticCache.SetToZero(sampleIndices);
                    semanticCache?.SetToZero(sampleIndices);
                }

                // speech_start: collapse the negative CFG stream back to a bare speech_start
                // (upstream masks its attention to the lone start token at each new segment).
                if (nextToken == VibeVoiceTokenizer.SpeechStartTokenId && negKvCache is not null)
                {
                    negKvCache.Reset();
                    negLastHidden?.Dispose();
                    negLastHidden = null;
                }

                if (nextToken == VibeVoiceTokenizer.SpeechDiffusionTokenId)
                {
                    // ── Diffusion sub-loop ──────────────────────────────────
                    // Positive condition is the LM's last hidden state at this position.
                    using Tensor cond = ExpandTo3D(lastHidden!, _lmCfg.HiddenSize);

                    // Negative condition for CFG: prime the unconditional stream on first use
                    // (forward a lone speech_start through negKvCache), then read its hidden.
                    Tensor? negCond = null;
                    if (negKvCache is not null)
                    {
                        if (negLastHidden is null)
                        {
                            int[] startTok = [VibeVoiceTokenizer.SpeechStartTokenId];
                            Tensor negInit = _lm.Forward(backend, startTok, batch: 1, posStart: negKvCache.CurrentLength, negKvCache);
                            negLastHidden = SliceLastFrame(negInit, _lmCfg.HiddenSize);
                            negInit.Dispose();
                        }
                        negCond = ExpandTo3D(negLastHidden!, _lmCfg.HiddenSize);
                    }

                    // Run 20 DDPM steps, v-prediction, cosine, CFG-combined cond/uncond.
                    Tensor noiseLatent = SampleNoise(_cfg.AcousticVaeDim, ref noiseRng);
                    Tensor denoised = DenoiseLatent(backend, noiseLatent, cond, negCond);
                    negCond?.Dispose();
                    noiseLatent.Dispose();

                    // Un-normalize: raw_latent = latent / scaling - bias.
                    UnnormalizeLatentInPlace(denoised, _speechScalingFactor, _speechBiasFactor);

                    // Decode one frame (1 latent) through the acoustic VAE → 3200 samples.
                    // Stream through the persistent acoustic cache so this frame's decode
                    // sees the prior frames' receptive-field tail.
                    using Tensor frameAudio = _acoustic.Decode(backend, denoised, batch: 1, acousticCache, sampleIndices);
                    float[] chunk = TensorToPcm(frameAudio);
                    audioChunks.Add(chunk);

                    // Build next-step embed: acoustic_connector(latent) + semantic_connector(sem_features).
                    Tensor latentNormalized = ReNormalizeLatent(denoised, _speechScalingFactor, _speechBiasFactor);
                    denoised.Dispose();

                    Tensor acousticEmbed = _acousticConnector.Forward(backend, latentNormalized, batch: 1, seqLen: 1);
                    Tensor? semanticEmbed = null;
                    if (_semantic is not null && _semanticConnector is not null)
                    {
                        using Tensor pcmTensor = PcmToTensor(chunk);
                        // Stream through the persistent semantic cache — the per-frame
                        // semantic features must be continuous with prior frames, else the
                        // LM's next-step conditioning drifts.
                        using Tensor semFeat = _semantic.Encode(backend, pcmTensor, batch: 1, tPcm: chunk.Length, semanticCache!, sampleIndices);
                        semanticEmbed = _semanticConnector.Forward(backend, semFeat, batch: 1, seqLen: 1);
                    }
                    latentNormalized.Dispose();

                    // Combine into a single [1, 1, hidden] embed and run one LM step.
                    using Tensor stepEmbed = AddEmbeds(acousticEmbed, semanticEmbed);
                    acousticEmbed.Dispose();
                    semanticEmbed?.Dispose();

                    Tensor newHidden = _lm.ForwardEmbeds(backend, stepEmbed, batch: 1, t: 1, posStart: kvCache.CurrentLength, kvCache);
                    lastHidden!.Dispose();
                    lastHidden = SliceLastFrame(newHidden, _lmCfg.HiddenSize);
                    newHidden.Dispose();

                    // Advance the negative stream in lockstep on the SAME diffusion embed
                    // (upstream feeds both streams identical inputs_embeds), so its next-token
                    // hidden reflects the same acoustic history minus the text/voice context.
                    if (negKvCache is not null)
                    {
                        Tensor negNew = _lm.ForwardEmbeds(backend, stepEmbed, batch: 1, t: 1, posStart: negKvCache.CurrentLength, negKvCache);
                        negLastHidden!.Dispose();
                        negLastHidden = SliceLastFrame(negNew, _lmCfg.HiddenSize);
                        negNew.Dispose();
                    }
                }
                else
                {
                    // Non-diffusion token: feed its embedding back to the LM.
                    int[] singleTok = [nextToken];
                    Tensor newHidden = _lm.Forward(backend, singleTok, batch: 1, posStart: kvCache.CurrentLength, kvCache);
                    lastHidden!.Dispose();
                    lastHidden = SliceLastFrame(newHidden, _lmCfg.HiddenSize);
                    newHidden.Dispose();
                }

                prevToken = nextToken;

                if (kvCache.CurrentLength >= cacheCap - 4) break;     // running out of cache room
            }
        }
        finally
        {
            lastHidden?.Dispose();
            negLastHidden?.Dispose();
        }

        // ── Assembly ─────────────────────────────────────────────────────────
        int totalSamples = 0;
        foreach (float[] c in audioChunks) totalSamples += c.Length;
        float[] audio = new float[totalSamples];
        int offset = 0;
        foreach (float[] c in audioChunks)
        {
            Array.Copy(c, 0, audio, offset, c.Length);
            offset += c.Length;
        }
        return audio;
    }

    /// <summary>Bulk-uploads every component's weights to the backend once (idempotent). Without
    /// this the first synthesis pays a per-op cache-miss H2D transfer on each weight; the acoustic
    /// VAE decoder + semantic encoder + diffusion head run thousands of ops per clip, so the
    /// scattered first-touch transfers otherwise dominate the cold gen. No-op on backends without
    /// a weight cache (CPU / Vulkan). Never paired with FreeWeights — VibeVoice keeps every
    /// component resident for the whole autoregressive loop.</summary>
    public void PreloadWeights(IBackend backend)
    {
        if (_preloaded) return;
        _preloaded = true;
        backend.PreloadWeights(EnumerateWeights());
    }

    private IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _lm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _acoustic.EnumerateWeights()) yield return t;
        if (_semantic is not null)
            foreach (Tensor t in _semantic.EnumerateWeights()) yield return t;
        foreach (Tensor t in _acousticConnector.EnumerateWeights()) yield return t;
        if (_semanticConnector is not null)
            foreach (Tensor t in _semanticConnector.EnumerateWeights()) yield return t;
        foreach (Tensor t in _diffusionHead.EnumerateWeights()) yield return t;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    private unsafe Tensor EncodeVoicePcm(IBackend backend, float[] pcm, int latentCount)
    {
        // Each acoustic VAE latent frame represents 3200 raw PCM samples
        // (8 × 5 × 5 × 4 × 2 × 2 stride product on the encoder).
        int tPcm = latentCount * 3_200;
        using Tensor pcmTensor = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* dst = (float*)pcmTensor.DataPointer;
        int copy = Math.Min(pcm.Length, tPcm);
        fixed (float* src = pcm) Buffer.MemoryCopy(src, dst, copy * 4, copy * 4);
        for (int j = copy; j < tPcm; j++) dst[j] = 0f;
        return _acoustic.Encode(backend, pcmTensor, batch: 1, tPcm: tPcm);
    }

    private unsafe Tensor BuildPrefillEmbeds(IBackend backend, VibeVoiceProcessor.PreparedPrompt prep, Tensor[] voiceLatents)
    {
        int promptLen = prep.TokenIds.Length;
        int h = _lmCfg.HiddenSize;
        Tensor embeds = new(new TensorShape(1, promptLen, h), DType.F32);
        // Default: text-embedding lookup for every position.
        _lm.EmbedLookup(embeds, prep.TokenIds, batch: 1, t: promptLen);

        // Now overlay voice latents at speech_input_mask positions. Each voice's latent
        // tensor is [1, N_i, 64]; we run it through acoustic_connector to get
        // [1, N_i, hidden], then write each row into the right embed position.
        int voiceIdx = 0;
        int slotIdx = 0;
        int v = 0;
        for (int i = 0; i < promptLen; i++)
        {
            if (!prep.SpeechInputMask[i]) continue;

            // If we're at slot 0 for this voice, project the entire latent block.
            if (slotIdx == 0)
            {
                using Tensor projected = _acousticConnector.Forward(backend, voiceLatents[v], batch: 1, seqLen: voiceLatents[v].Shape[1] is var s ? (int)s : 0);
                CopyLatentRows(embeds, projected, embedPosStart: i, latentRowStart: 0, latentCount: prep.Voices[v].LatentCount, hidden: h);
            }

            slotIdx++;
            voiceIdx++;
            if (slotIdx >= prep.Voices[v].LatentCount)
            {
                slotIdx = 0;
                v++;
            }
        }

        return embeds;
    }

    private static unsafe void CopyLatentRows(Tensor embeds, Tensor projected, int embedPosStart, int latentRowStart, int latentCount, int hidden)
    {
        float* ep = (float*)embeds.DataPointer;
        float* pp = (float*)projected.DataPointer;
        int t = (int)embeds.Shape[1];
        for (int k = 0; k < latentCount; k++)
        {
            float* src = pp + (long)(latentRowStart + k) * hidden;
            float* dst = ep + (long)(embedPosStart + k) * hidden;
            Buffer.MemoryCopy(src, dst, hidden * 4, hidden * 4);
        }
    }

    private Tensor DenoiseLatent(IBackend backend, Tensor noiseLatent, Tensor cond, Tensor? negCond)
    {
        using VibeVoiceCosineDpmSolver scheduler = new();
        scheduler.SetTimesteps(_cfg.DiffusionHead.DdpmNumInferenceSteps);

        float cfg = _cfg.CfgScale;
        bool useCfg = negCond is not null && MathF.Abs(cfg - 1f) > 1e-6f;

        Tensor speech = noiseLatent;
        bool ownsSpeech = false;
        int latent = _cfg.AcousticVaeDim;
        int hidden = _lmCfg.HiddenSize;
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;

        // Batched CFG: upstream feeds the SAME noisy latent to the conditional and unconditional halves, so
        // instead of two N=1 head forwards we stack them into ONE N=2 forward (row 0 = cond, row 1 = uncond) —
        // the head is FFN-only with no cross-frame mixing, so each row is exactly its own pass. Halves the
        // head's tiny-GEMV launch count. condB is fixed across a token's 20 steps → built once here.
        // Combine: eps = uncond + cfg·(cond−uncond).
        // (CUDA-graph capture of the N=2 head step was tried and reverted: after the host-glue→GPU port the
        //  step is GPU-compute-bound, so the graph was wall-neutral — and capture/replay perturbed the
        //  TF32-sensitive AR feedback enough to diverge the generation. Not worth it.)
        Tensor? condB = null;
        if (useCfg)
        {
            condB = new(new TensorShape(1, 2, hidden), DType.F32);
            backend.Concat(condB, [cond, negCond!], dim: 1);
        }
        for (int step = 0; step < timesteps.Length; step++)
        {
            Tensor vPred;
            if (useCfg)
            {
                Tensor speechB = new(new TensorShape(1, 2, latent), DType.F32);
                backend.Concat(speechB, [speech, speech], dim: 1);
                float[] tBatch2 = [timesteps[step], timesteps[step]];
                Tensor vB = _diffusionHead.Forward(backend, speechB, tBatch2, condB!);
                vPred = CombineCfgBatched(vB, cfg, latent);
                speechB.Dispose();
                vB.Dispose();
            }
            else
            {
                float[] tBatch = [timesteps[step]];
                vPred = _diffusionHead.Forward(backend, speech, tBatch, cond);
            }
            Tensor next = scheduler.Step(vPred, speech, step);
            vPred.Dispose();
            if (ownsSpeech) speech.Dispose();
            speech = next;
            ownsSpeech = true;
        }
        condB?.Dispose();

        if (!ownsSpeech)
        {
            // No steps ran somehow — defensive clone.
            Tensor clone = new(speech.Shape, DType.F32);
            unsafe { Buffer.MemoryCopy((void*)speech.DataPointer, (void*)clone.DataPointer, speech.ElementCount * 4, speech.ElementCount * 4); }
            return clone;
        }
        return speech;
    }

    /// <summary>Samples one of the allowed VibeVoice control tokens with temperature + top-p (matching the
    /// official inference). Greedy argmax deadlocks on short prompts: if <c>speech_diffusion</c> edges out
    /// <c>speech_end</c>/<c>eos</c> by any margin it never stops until the token cap, which is exactly the
    /// "rambles ~12s" symptom. Stochastic sampling lets the close stop token win. Deterministic for a fixed
    /// <paramref name="rng"/>. <paramref name="temperature"/> ≤ 0 falls back to greedy.</summary>
    private static unsafe int SampleConstrained(Tensor logits, float temperature, float topP, ref uint rng)
    {
        float* p = (float*)logits.DataPointer;     // [1, 1, vocab]
        ReadOnlySpan<int> allowed =
        [
            VibeVoiceTokenizer.SpeechStartTokenId,
            VibeVoiceTokenizer.SpeechEndTokenId,
            VibeVoiceTokenizer.SpeechDiffusionTokenId,
            VibeVoiceTokenizer.EndOfTextTokenId,
        ];
        int n = allowed.Length;
        if (temperature <= 0f)
        {
            int best = allowed[0];
            float bestV = p[best];
            for (int i = 1; i < n; i++)
                if (p[allowed[i]] > bestV) { bestV = p[allowed[i]]; best = allowed[i]; }
            return best;
        }

        Span<float> probs = stackalloc float[n];
        Span<int> order = stackalloc int[n];
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) { probs[i] = p[allowed[i]] / temperature; order[i] = i; if (probs[i] > max) max = probs[i]; }
        float sum = 0f;
        for (int i = 0; i < n; i++) { probs[i] = MathF.Exp(probs[i] - max); sum += probs[i]; }
        for (int i = 0; i < n; i++) probs[i] /= sum;

        // Sort indices by probability (descending) for the top-p nucleus over this tiny set.
        for (int i = 1; i < n; i++)
            for (int j = i; j > 0 && probs[order[j]] > probs[order[j - 1]]; j--)
                (order[j], order[j - 1]) = (order[j - 1], order[j]);
        float cum = 0f;
        int keep = n;
        for (int r = 0; r < n; r++)
        {
            cum += probs[order[r]];
            if (topP > 0f && topP < 1f && cum >= topP) { keep = r + 1; break; }
        }
        float keptSum = 0f;
        for (int r = 0; r < keep; r++) keptSum += probs[order[r]];
        float draw = HartsyInference.Audio.Dsp.DeterministicRng.NextUniform(ref rng) * keptSum;
        float acc = 0f;
        for (int r = 0; r < keep; r++)
        {
            acc += probs[order[r]];
            if (draw <= acc) return allowed[order[r]];
        }
        return allowed[order[keep - 1]];
    }

    // CFG combine from a batched [1, 2, latent] head output (row 0 = cond, row 1 = uncond):
    //   eps = uncond + cfg_scale * (cond - uncond). Latents are tiny (latent=64) so the tail stays host.
    private static unsafe Tensor CombineCfgBatched(Tensor vBatched, float cfg, int latent)
    {
        Tensor result = new(new TensorShape(1, 1, latent), DType.F32);
        float* v = (float*)vBatched.DataPointer;
        float* r = (float*)result.DataPointer;
        for (int i = 0; i < latent; i++) r[i] = v[latent + i] + cfg * (v[i] - v[latent + i]);
        return result;
    }

    private static unsafe Tensor SliceLastFrame(Tensor hidden, int dim)
    {
        // [B, T, dim] → [B, 1, dim] of the last T position.
        int batch = (int)hidden.Shape[0];
        int t = (int)hidden.Shape[1];
        Tensor result = new(new TensorShape(batch, 1, dim), DType.F32);
        float* sp = (float*)hidden.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long srcOff = ((long)b * t + (t - 1)) * dim;
            long dstOff = (long)b * dim;
            Buffer.MemoryCopy(sp + srcOff, dp + dstOff, dim * 4, dim * 4);
        }
        return result;
    }

    private static unsafe Tensor ExpandTo3D(Tensor x, int dim)
    {
        // Ensure shape is [1, 1, dim]; clone if already so.
        if (x.Shape.Rank == 3 && (int)x.Shape[1] == 1) return CopyOf(x);
        if (x.Shape.Rank == 2)
        {
            Tensor t = new(new TensorShape(1, 1, dim), DType.F32);
            Buffer.MemoryCopy((void*)x.DataPointer, (void*)t.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
            return t;
        }
        return CopyOf(x);
    }

    private static unsafe Tensor CopyOf(Tensor src)
    {
        Tensor copy = new(src.Shape, DType.F32);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)copy.DataPointer, src.ElementCount * 4, src.ElementCount * 4);
        return copy;
    }

    // Draws fresh decorrelated Gaussian noise from one persistent stream (matching torch.randn per frame). The old
    // per-frame `new Random(step+1)` gave adjacent frames near-identical first draws (correlated seeds) — near-duplicate
    // latents that garble short utterances — and ignored the user seed.
    private static unsafe Tensor SampleNoise(int dim, ref uint rng)
    {
        Tensor t = new(new TensorShape(1, 1, dim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < dim; i++)
            p[i] = HartsyInference.Audio.Dsp.DeterministicRng.NextGaussian(ref rng);
        return t;
    }

    private static unsafe Tensor AddEmbeds(Tensor a, Tensor? b)
    {
        Tensor result = CopyOf(a);
        if (b is null) return result;
        long n = result.ElementCount;
        float* rp = (float*)result.DataPointer;
        float* bp = (float*)b.DataPointer;
        for (long i = 0; i < n; i++) rp[i] += bp[i];
        return result;
    }

    private static unsafe void NormalizeLatentInPlace(Tensor latent, float scale, float bias)
    {
        long n = latent.ElementCount;
        float* p = (float*)latent.DataPointer;
        for (long i = 0; i < n; i++) p[i] = (p[i] + bias) * scale;
    }

    private static unsafe void UnnormalizeLatentInPlace(Tensor latent, float scale, float bias)
    {
        long n = latent.ElementCount;
        float* p = (float*)latent.DataPointer;
        float invScale = 1f / scale;
        for (long i = 0; i < n; i++) p[i] = p[i] * invScale - bias;
    }

    private static unsafe Tensor ReNormalizeLatent(Tensor latent, float scale, float bias)
    {
        Tensor copy = CopyOf(latent);
        long n = copy.ElementCount;
        float* p = (float*)copy.DataPointer;
        for (long i = 0; i < n; i++) p[i] = (p[i] + bias) * scale;
        return copy;
    }

    private static unsafe float[] TensorToPcm(Tensor frame)
    {
        // [1, 1, T] → float[T]
        int t = (int)frame.Shape[2];
        float[] result = new float[t];
        float* p = (float*)frame.DataPointer;
        fixed (float* dst = result) Buffer.MemoryCopy(p, dst, t * 4, t * 4);
        return result;
    }

    private static unsafe Tensor PcmToTensor(float[] pcm)
    {
        Tensor t = new(new TensorShape(1, 1, pcm.Length), DType.F32);
        fixed (float* src = pcm) Buffer.MemoryCopy(src, (void*)t.DataPointer, pcm.Length * 4, pcm.Length * 4);
        return t;
    }

    private static unsafe float ReadScalar(IReadOnlyDictionary<string, Tensor> w, string key)
    {
        if (!w.TryGetValue(key, out Tensor? t)) return 1.0f;
        if (t.DType == DType.F32) return *(float*)t.DataPointer;
        using Tensor f = t.CastTo(DType.F32);     // checkpoint stores these as bf16 scalars
        return *(float*)f.DataPointer;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(VibeVoicePipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        _tokenizer.Dispose();
        // The safetensors mmap unroots once the LM's borrowed weight tensors are collected.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}
