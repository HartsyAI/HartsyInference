namespace SharpInference.Audio.Models.Kokoro;

/// <summary>Configuration for the Kokoro-82M TTS model (Hexgrad release).
///
/// <para>Kokoro is the production-ready StyleTTS 2 derivative used as the first TTS in
/// SharpInference. The model is small (82M params) but trained on a huge mixed corpus,
/// so end-to-end quality is competitive with much larger systems while running entirely
/// on CPU. Pipeline shape:</para>
///
/// <code>
///   phonemes (IPA) ──► PLBERT (12-layer ALBERT, weight-shared) ──► [1, T, 768]
///         │                                                          │
///         │                                                          ▼  bert_encoder
///         │                                            d_bert = Linear(768→512)
///         ▼                                                          │
///   TextEncoder: Embed(178→512) + 3× Conv1D(k=5)+LayerNorm+LeakyReLU
///                 + BiLSTM(512→512)                                  │
///                                                  ┌─── voice_pack ──┴──┐
///                                                  ▼                    ▼
///                                ProsodyPredictor                  Decoder
///                                (d_bert + s_pred → durations,    (s_dec ⊕ asr ⊕
///                                 F0, energy)                      F0 ⊕ N → audio)
/// </code>
///
/// <para>The voice-pack tensor is sliced into a 128-dim "predictor style" half
/// (<c>s_pred = ref_s[:, 128:]</c>) and a 128-dim "decoder style" half
/// (<c>s_dec = ref_s[:, :128]</c>). The two halves never see each other's path.</para>
///
/// <para>Defaults match the upstream <c>hexgrad/Kokoro-82M</c> <c>config.json</c>.
/// Override per-checkpoint when forks ship with different hyperparameters.</para></summary>
public sealed record KokoroConfig
{
    /// <summary>Vocabulary size (phoneme IDs) — includes gaps. The actual symbol count
    /// is ~114; IDs are sparse over <c>[0, 177]</c> with 0 reserved for BOS/EOS padding.</summary>
    public int NumTokens { get; init; } = 178;

    /// <summary>Mel bins used by the (training-only) acoustic head. Not used at inference
    /// in our pipeline — the decoder skips mel entirely and outputs audio directly via
    /// iSTFT.</summary>
    public int NumMels { get; init; } = 80;

    /// <summary>Style-vector dim consumed by the predictor + decoder halves. Voice packs
    /// are <c>[N, 2*style_dim] = [N, 256]</c>; the first 128 dims condition the decoder,
    /// the trailing 128 dims condition the prosody predictor.</summary>
    public int StyleDim { get; init; } = 128;

    /// <summary>Decoder input bottleneck (asr_res output channels).</summary>
    public int DimIn { get; init; } = 64;

    /// <summary>Hidden dim shared by TextEncoder Conv1D channels, BiLSTM input,
    /// prosody predictor LSTM input, and the per-frame AdainResBlk1d width.</summary>
    public int HiddenDim { get; init; } = 512;

    /// <summary>Max convolution channel dim in the decoder. Same as <see cref="HiddenDim"/>
    /// for the 82M checkpoint, but kept separate for forks.</summary>
    public int MaxConvDim { get; init; } = 512;

    /// <summary>Maximum predicted duration per phoneme. The duration head outputs a 50-way
    /// sigmoid distribution whose sum (divided by the speed knob) is the integer frame
    /// count. 50 frames at the model's 24 kHz / 600× upsample = 1.25 s per phoneme cap.</summary>
    public int MaxDuration { get; init; } = 50;

    /// <summary>Conv1D kernel size used by every layer of <c>KokoroTextEncoder</c>.</summary>
    public int TextEncoderKernelSize { get; init; } = 5;

    /// <summary>Number of Conv1D blocks in the text encoder before the BiLSTM. 3 in the
    /// 82M checkpoint.</summary>
    public int TextEncoderNumLayers { get; init; } = 3;

    /// <summary>Output sample rate of the iSTFT head.</summary>
    public int SampleRate { get; init; } = 24_000;

    /// <summary>PLBERT settings — 12-layer ALBERT (all 12 share one weight set).</summary>
    public PlBertConfig PlBert { get; init; } = new();

    /// <summary>iSTFTNet decoder hyperparameters.</summary>
    public IStftNetConfig IStftNet { get; init; } = new();

    /// <summary>Hexgrad Kokoro-82M preset.</summary>
    public static KokoroConfig V1 => new();

    /// <summary>PLBERT (Phoneme-Level BERT) sub-config. ALBERT-style: all 12 transformer
    /// layers share a single weight set, looped 12 times at inference.</summary>
    public sealed record PlBertConfig
    {
        public int HiddenSize { get; init; } = 768;
        public int NumAttentionHeads { get; init; } = 12;
        public int IntermediateSize { get; init; } = 2_048;
        public int MaxPositionEmbeddings { get; init; } = 512;
        public int NumHiddenLayers { get; init; } = 12;

        /// <summary>Embedding dim for the input token / position / segment embeddings.
        /// ALBERT factorizes the embedding to keep the param count low — the embedding
        /// is 128-dim, then projected to <see cref="HiddenSize"/>=768 by the first
        /// transformer block's input projection.</summary>
        public int EmbeddingSize { get; init; } = 128;

        public float LayerNormEps { get; init; } = 1e-12f;
    }

    /// <summary>iSTFTNet head config (Kokoro / StyleTTS 2 decoder). The upsampling chain
    /// is 10× then 6× (= 60× total) plus the final 5-hop iSTFT for an effective
    /// frame→sample ratio of 300, which matches the predictor's 80 mel-frame-per-second
    /// time base when paired with 24 kHz output.</summary>
    public sealed record IStftNetConfig
    {
        public int[] UpsampleKernelSizes { get; init; } = [20, 12];
        public int[] UpsampleRates { get; init; } = [10, 6];

        /// <summary>iSTFT hop size used by the final synthesis head.</summary>
        public int GenIstftHopSize { get; init; } = 5;

        /// <summary>iSTFT FFT size used by the final synthesis head.</summary>
        public int GenIstftNFft { get; init; } = 20;

        /// <summary>HiFi-GAN MRF dilation pattern per ResBlock — three branches with
        /// dilations [1,3,5] each.</summary>
        public int[][] ResBlockDilationSizes { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

        public int[] ResBlockKernelSizes { get; init; } = [3, 7, 11];

        public int UpsampleInitialChannel { get; init; } = 512;
    }
}
