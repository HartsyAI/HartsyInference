namespace HartsyInference.Audio.Models.Haru;

/// <summary>The positions that open a conversation: the character card as a system turn (text only), then the voice reference as audio on
/// the model's stream (text PAD, EOP on the last frame), then for a model that speaks first one more PAD and EOP position.</summary>
/// <remarks>Latents are raw codec latents, <c>Length * LatentDim</c> of them per stream; a mask marks the positions that carry the stream.</remarks>
internal sealed record HaruPrompt(int[] Tokens, float[] ModelLatents, float[] UserLatents, bool[] ModelMask, bool[] UserMask)
{
    internal int Length => Tokens.Length;

    /// <summary>Builds the prompt. <paramref name="reference"/> is prepared mono 24 kHz audio, or empty for no voice reference.</summary>
    internal static HaruPrompt Build(HaruConfig config, HaruTextCodec text, HaruCodecSession codec, IHaruNoise noise, string card,
        ReadOnlySpan<float> reference, bool speakFirst)
    {
        int d = config.LatentDim;
        int pad = config.PadTokenId;
        int eop = config.EopTokenId;
        int[] cardIds = string.IsNullOrEmpty(card) ? [] : text.Encode($"<|im_start|>system\n{card}<|im_end|>\n");

        float[] referenceLatents = [];
        if (!reference.IsEmpty)
        {
            referenceLatents = codec.Sample(codec.EncodeClip(reference), noise);
        }
        int referenceFrames = referenceLatents.Length / d;
        // A reference already ends in EOP, so an opening turn starts with PAD, EOP.
        List<int> grid = speakFirst ? [pad, eop] : [];
        if (cardIds.Length == 0 && referenceFrames == 0 && grid.Count == 0)
        {
            grid.Add(pad);
        }

        List<int> tokens = [.. cardIds];
        for (int i = 0; i < referenceFrames; i++)
        {
            tokens.Add(i == referenceFrames - 1 ? eop : pad);
        }
        tokens.AddRange(grid);
        int n = tokens.Count;

        // The model stream carries audio from the end of the card on, the user stream only on the grid; positions before generated audio
        // hold the reference's quietest frame (the speaker's noise floor) and the encoded-silence latent.
        float[] modelLatents = new float[n * d];
        float[] userLatents = new float[n * d];
        referenceLatents.CopyTo(modelLatents, cardIds.Length * d);
        if (grid.Count > 0)
        {
            float[] roomTone = RoomTone(config, codec, reference, referenceLatents);
            float[] silence = codec.SilenceLatent();
            for (int i = n - grid.Count; i < n; i++)
            {
                roomTone.CopyTo(modelLatents, i * d);
                silence.CopyTo(userLatents, i * d);
            }
        }
        bool[] modelMask = new bool[n];
        bool[] userMask = new bool[n];
        for (int i = 0; i < n; i++)
        {
            modelMask[i] = i >= cardIds.Length;
            userMask[i] = i >= n - grid.Count;
        }
        return new HaruPrompt([.. tokens], modelLatents, userLatents, modelMask, userMask);
    }

    private static float[] RoomTone(HaruConfig config, HaruCodecSession codec, ReadOnlySpan<float> reference, float[] referenceLatents)
    {
        int d = config.LatentDim;
        int frame = config.FrameSamples;
        int whole = reference.Length / frame;
        if (referenceLatents.Length == 0 || whole == 0)
        {
            return codec.SilenceLatent();
        }
        int quietest = 0;
        double lowest = double.MaxValue;
        for (int f = 0; f < whole; f++)
        {
            double sum = 0;
            foreach (float v in reference.Slice(f * frame, frame))
            {
                sum += (double)v * v;
            }
            if (sum < lowest)
            {
                lowest = sum;
                quietest = f;
            }
        }
        return referenceLatents[(quietest * d)..((quietest + 1) * d)];
    }
}
