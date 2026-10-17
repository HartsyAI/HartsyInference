namespace HartsyInference.LLM.Looped;

/// <summary>The unrolled layer order of one forward for a given loop count: which physical layer each virtual layer runs, which key/value
/// cache slot it reads and writes, and where the recurrence re-injects the input block's output. Slot numbers are stable across loop
/// counts (input and output blocks first, then one group of recurrent slots per loop), so a cache sized by <see cref="MaxSlotCount"/>
/// serves every loop count up to the maximum.</summary>
public sealed class LoopedSchedule
{
    private readonly int[] _physical;
    private readonly int[] _slot;

    /// <summary>The shape this schedule unrolls.</summary>
    public LoopedTransformerConfig Config { get; }

    /// <summary>Times the recurrent block runs.</summary>
    public int Loops { get; }

    /// <summary>Layers run in one forward: input + recurrent × loops + output.</summary>
    public int VirtualLayers => _physical.Length;

    /// <summary>First virtual layer of the recurrence, the point where the input block's output is captured.</summary>
    public int RecurrentStart => Config.InputLayers;

    /// <summary>One past the last recurrent virtual layer.</summary>
    public int RecurrentEnd => Config.InputLayers + Config.RecurrentLayers * Loops;

    /// <summary>Cache slots this schedule touches, at most the cache's slot count.</summary>
    public int SlotsUsed => SlotCount(Config, Loops);

    /// <summary>Unrolls <paramref name="config"/> for <paramref name="loops"/> passes of the recurrent block.</summary>
    public LoopedSchedule(LoopedTransformerConfig config, int loops)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (loops < 1 || loops > config.EffectiveMaxLoops)
            throw new ArgumentOutOfRangeException(nameof(loops), loops, $"Loops must be in [1, {config.EffectiveMaxLoops}] for this model.");
        Config = config;
        Loops = loops;
        int inputAttn = config.AttentionLayersIn(0, config.InputLayers);
        int outputAttn = config.AttentionLayersIn(config.InputLayers + config.RecurrentLayers, config.OutputLayers);
        int recurrentAttn = config.AttentionLayersIn(config.InputLayers, config.RecurrentLayers);
        int virtualCount = config.InputLayers + config.RecurrentLayers * loops + config.OutputLayers;
        _physical = new int[virtualCount];
        _slot = new int[virtualCount];
        int v = 0;
        int inputSlot = 0;
        for (int p = 0; p < config.InputLayers; p++, v++)
        {
            _physical[v] = p;
            _slot[v] = config.IsAttention(p) ? inputSlot++ : -1;
        }
        for (int k = 0; k < loops; k++)
        {
            int loopBase = inputAttn + outputAttn + (config.KvCacheMode == LoopedKvCacheMode.PerLoop ? k * recurrentAttn : 0);
            int j = 0;
            for (int i = 0; i < config.RecurrentLayers; i++, v++)
            {
                int p = config.InputLayers + i;
                _physical[v] = p;
                _slot[v] = config.IsAttention(p) ? loopBase + j++ : -1;
            }
        }
        int outputSlot = inputAttn;
        for (int i = 0; i < config.OutputLayers; i++, v++)
        {
            int p = config.InputLayers + config.RecurrentLayers + i;
            _physical[v] = p;
            _slot[v] = config.IsAttention(p) ? outputSlot++ : -1;
        }
    }

    /// <summary>The physical layer virtual layer <paramref name="virtualLayer"/> runs.</summary>
    public int Physical(int virtualLayer) => _physical[virtualLayer];

    /// <summary>The cache slot virtual layer <paramref name="virtualLayer"/> uses, or -1 for an MLP layer.</summary>
    public int CacheSlot(int virtualLayer) => _slot[virtualLayer];

    /// <summary>Whether virtual layer <paramref name="virtualLayer"/> opens a pass of the recurrent block, where the input is re-injected.</summary>
    public bool IsLoopStart(int virtualLayer) =>
        virtualLayer >= RecurrentStart && virtualLayer < RecurrentEnd && (virtualLayer - RecurrentStart) % Config.RecurrentLayers == 0;

    /// <summary>Whether virtual layer <paramref name="virtualLayer"/> closes a pass of the recurrent block, after which the state is re-normalized.</summary>
    public bool IsLoopEnd(int virtualLayer) =>
        virtualLayer >= RecurrentStart && virtualLayer < RecurrentEnd && (virtualLayer + 1 - RecurrentStart) % Config.RecurrentLayers == 0;

    /// <summary>Cache slots a forward of <paramref name="loops"/> passes needs under <paramref name="config"/>.</summary>
    public static int SlotCount(LoopedTransformerConfig config, int loops)
    {
        ArgumentNullException.ThrowIfNull(config);
        int fixedSlots = config.AttentionLayersIn(0, config.InputLayers)
            + config.AttentionLayersIn(config.InputLayers + config.RecurrentLayers, config.OutputLayers);
        int recurrentAttn = config.AttentionLayersIn(config.InputLayers, config.RecurrentLayers);
        return fixedSlots + recurrentAttn * (config.KvCacheMode == LoopedKvCacheMode.PerLoop ? loops : 1);
    }

    /// <summary>Cache slots a cache must hold to serve any loop count up to <see cref="LoopedTransformerConfig.EffectiveMaxLoops"/>.</summary>
    public static int MaxSlotCount(LoopedTransformerConfig config) => SlotCount(config, config.EffectiveMaxLoops);
}
