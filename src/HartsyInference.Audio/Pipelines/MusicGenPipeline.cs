using System.Diagnostics;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>MusicGen / AudioGen text-to-audio pipeline. Takes <b>precomputed T5-base cross-attn states</b>
/// (the Audio package carries no T5 dependency), AR-generates the delayed EnCodec token grid with
/// classifier-free guidance, reverts the <see cref="MusicGenDelay">delay pattern</see>, and decodes via the
/// built <see cref="EnCodec"/>. Reuses <see cref="NucleusSampler"/> (top-k/top-p draw) and the EnCodec
/// decoder. The special token is masked out of every active codebook's sampling so only real codes are drawn.</summary>
public sealed unsafe class MusicGenPipeline(MusicGenConfig cfg, MusicGenDecoder decoder, EnCodec codec) : IDisposable
{
    private readonly MusicGenConfig _cfg = cfg;
    private readonly MusicGenDecoder _decoder = decoder;
    private readonly EnCodec _codec = codec;
    private int _disposed;

    /// <summary>Generates audio from precomputed T5 states <c>[1, T_text, textDim]</c>. <paramref name="seconds"/>
    /// sets the real frame count (<c>frameRate × seconds</c>). Returns mono PCM at the codec sample rate.
    /// <para>The AudioCraft generation params (<c>set_generation_params</c>) are exposed per-call, defaulting to the
    /// config: <paramref name="guidance"/> = <c>cfg_coef</c> (3.0), <paramref name="temperature"/> (1.0),
    /// <paramref name="topK"/> (250), <paramref name="topP"/> (0.0 = off), <paramref name="useSampling"/> (true;
    /// false = greedy argmax).</para></summary>
    public float[] Synthesize(IBackend backend, Tensor t5States, float seconds = 8f, int seed = 0,
        float? guidance = null, float? temperature = null, int? topK = null, float? topP = null, bool useSampling = true)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int k = _cfg.NumCodebooks;
        float temp = temperature ?? _cfg.Temperature;
        int tk = topK ?? _cfg.TopK;
        float tp = topP ?? _cfg.TopP;
        int tReal = Math.Max(1, (int)MathF.Round(_cfg.CodecFrameRate * seconds));
        int maxDelay = MusicGenDelay.Max(_cfg.DelayPattern);
        int tTotal = tReal + maxDelay;

        uint rng = DeterministicRng.Seed(seed);
        // Delayed grid filled autoregressively; each codebook's lead-in and tail hold the special token.
        int[,] delayed = new int[tTotal, k];

        float g = guidance ?? _cfg.GuidanceScale;
        // CFG cond+uncond are decoded as ONE batch=B pass (B=2 with CFG, else 1): batch row 0 cross-attends the
        // T5 text, row 1 the null (zeros) branch. Halves the per-step kernel count and widens every GEMM vs two
        // sequential passes. Build the batched, once-projected cross states [B, T, hidden].
        int B = g != 1f ? 2 : 1;
        Tensor condCross = _decoder.ProjectText(backend, t5States);   // [1, T, hidden]
        int tText = (int)condCross.Shape[1];
        Tensor batchedCross;
        if (B == 2)
        {
            batchedCross = new(new TensorShape(2, tText, _cfg.Hidden), DType.F32);   // host, zeroed
            long rowBytes = (long)tText * _cfg.Hidden * sizeof(float);
            Buffer.MemoryCopy((void*)condCross.DataPointer, (void*)batchedCross.DataPointer, rowBytes, rowBytes);   // row 0 = cond; row 1 stays 0
            condCross.Dispose();
        }
        else batchedCross = condCross;
        MusicGenKvCache cache = _decoder.CreateCache(backend, batchedCross, tTotal);   // batch=B, cross primed per row
        batchedCross.Dispose();
        _decoder.EnsureGraphBuffers(B);

        // Standard causal-LM AR loop (HF `_sample`): feed the previous delay-masked frame — starting with the
        // all-special BOS/decoder_start frame — and read the next-frame logits it produces to sample row `step`.
        int[] prev = new int[k];
        for (int c = 0; c < k; c++) prev[c] = _cfg.SpecialToken;   // BOS frame (decoder_start_token)

        // CUDA-graph decode: the batched per-step op sequence is identical every step (only the token embedding and
        // the device KV position change), so we capture it ONCE and replay with a single launch. Steps
        // 0..GraphCaptureStep-1 run eagerly THROUGH the fixed buffers (warming resident weights/cross-K/V so nothing
        // inside capture does a sync upload); the capture step records + launches; later steps replay. Any
        // capture-illegal op falls back to eager for the rest of the generation.
        bool graphOn = backend.StepGraphSupported && backend.GraphDecodeSupported
            && Environment.GetEnvironmentVariable("HARTSY_MUSICGEN_GRAPH_OFF") is null;
        bool graphDead = false;
        const int GraphCaptureStep = 3;
        if (graphOn)
        {
            backend.StepGraphOwner = this;
            backend.PreloadWeights(_decoder.EnumerateWeights());   // pin so no pageable upload invalidates capture
        }

        for (int step = 0; step < tTotal; step++)
        {
            int pos = cache.Length;
            // Per-step-varying inputs, refreshed OUTSIDE any capture region.
            _decoder.PrepareEmbed(backend, prev, pos, B);
            if (cache.DevicePos != 0) backend.WriteDevicePos(cache.DevicePos, pos + 1, pos);

            bool ownerOk = graphOn && ReferenceEquals(backend.StepGraphOwner, this);
            if (graphOn && !graphDead && ownerOk && backend.StepGraphReady && step > GraphCaptureStep)
            {
                backend.StepGraphLaunch();
            }
            else
            {
                bool capture = graphOn && !graphDead && ownerOk && step == GraphCaptureStep && !backend.StepGraphReady;
                if (capture) backend.StepGraphBegin();
                try
                {
                    _decoder.RunBatchedIntoFixed(backend, cache);
                }
                catch (Exception ex) when (capture)
                {
                    backend.StepGraphReset(); graphDead = true;
                    Logs.Warning($"[MusicGen graph] capture invalidated — eager fallback: {ex.Message}");
                    _decoder.RunBatchedIntoFixed(backend, cache);
                }
                if (capture)
                {
                    try
                    {
                        backend.StepGraphEndAndLaunch();   // capture records without executing — this runs the step
                        Logs.Info("[MusicGen graph] batched decode step captured; replaying via cuGraphLaunch.");
                    }
                    catch (Exception ex)
                    {
                        backend.StepGraphReset(); graphDead = true;
                        Logs.Warning($"[MusicGen graph] capture failed — eager fallback: {ex.Message}");
                        _decoder.RunBatchedIntoFixed(backend, cache);
                    }
                }
            }
            cache.Length = pos + 1;   // device KV was appended at slot=pos by the launch
            float[][][] batched = _decoder.ReadBatchedLogits(backend, B);
            float[][] condLogits = batched[0];
            float[][] uncondLogits = B == 2 ? batched[1] : batched[0];

            for (int c = 0; c < k; c++)
            {
                int j = step - _cfg.DelayPattern[c];   // this codebook's real-frame index at this step
                if (j < 0 || j >= tReal)
                {
                    delayed[step, c] = _cfg.SpecialToken;   // lead-in (j<0) or tail (j>=tReal) → pad, per delay mask
                    continue;
                }
                float[] logits = condLogits[c];
                if (g != 1f)
                {
                    float[] u = uncondLogits[c];
                    for (int v = 0; v < logits.Length; v++) logits[v] = u[v] + g * (logits[v] - u[v]);
                }
                delayed[step, c] = useSampling
                    ? NucleusSampler.Draw(logits, _cfg.CodebookSize, temp, tk, tp, ref rng, maskToken: _cfg.SpecialToken)
                    : Argmax(logits, _cfg.CodebookSize, _cfg.SpecialToken);
            }

            // The delay-masked frame just produced becomes the next step's input.
            for (int c = 0; c < k; c++) prev[c] = delayed[step, c];
        }

        if (graphOn)
        {
            backend.StepGraphReset();
            if (ReferenceEquals(backend.StepGraphOwner, this)) backend.StepGraphOwner = null;
        }
        _decoder.ReleaseGraphBuffers();
        cache.Dispose();

        int[,] real = MusicGenDelay.Revert(delayed, _cfg.DelayPattern, tReal);

        // Pack into EnCodec's [nQ, batch=1, T] Int32 grid and decode.
        Tensor codes = new(new TensorShape(k, 1, tReal), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int c = 0; c < k; c++)
            for (int j = 0; j < tReal; j++)
                cp[(long)c * tReal + j] = Math.Clamp(real[j, c], 0, _cfg.CodebookSize - 1);

        Tensor audioT = _codec.Decode(backend, codes, batch: 1, tFrames: tReal);
        codes.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();
        sw.Stop();
        Logs.Info($"MusicGen: {tReal} frames → {audio.Length} samples ({audio.Length / (double)_cfg.CodecSampleRate:F1}s) in {sw.ElapsedMilliseconds}ms (batch={B}, graph={(graphOn && !graphDead)}).");
        return audio;
    }

    /// <summary>Greedy (use_sampling=false): argmax over the valid codebook range, skipping the special token.</summary>
    private static int Argmax(float[] logits, int vocab, int maskToken)
    {
        int best = 0; float bv = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            if (v == maskToken) continue;
            if (logits[v] > bv) { bv = logits[v]; best = v; }
        }
        return best;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _decoder.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(MusicGenPipeline));
    }
}
