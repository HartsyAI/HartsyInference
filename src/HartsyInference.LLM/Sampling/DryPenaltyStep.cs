using System;
using System.Collections.Generic;

namespace HartsyInference.LLM.Sampling;

/// <summary>The "Don't Repeat Yourself" penalty: a token that would extend a sequence that already occurred earlier in the history has
/// <c>min(MaxPenalty, Multiplier * Base^(length - AllowedLength))</c> subtracted from its logit, <c>length</c> being how many of the latest
/// tokens that earlier occurrence matches. Tokens in <c>exempt</c> (silence, padding, end of turn) neither count towards the history nor
/// get penalized.</summary>
/// <remarks>One instance lives for one generation and is driven from one thread, as <see cref="SamplerChain"/> requires: its scratch
/// buffers are sized once and reused for every step.</remarks>
public sealed class DryPenaltyStep : ISamplerStep
{
    private readonly float _multiplier;
    private readonly float _base;
    private readonly int _allowedLength;
    private readonly int _lookback;
    private readonly int _maxMatch;
    private readonly float _maxPenalty;
    private readonly HashSet<int> _exempt;
    private readonly int[] _window;
    private readonly Dictionary<int, int> _longest = [];

    /// <summary>Creates the step; <paramref name="exempt"/> lists the token ids that are ignored.</summary>
    public DryPenaltyStep(DryOptions options, IEnumerable<int> exempt)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(exempt);
        if (options.Lookback < 1 || options.MaxMatchLength < 1 || options.AllowedLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), "DRY needs a positive lookback and match length and a non-negative allowed length.");
        }
        _multiplier = options.Multiplier;
        _base = options.Base;
        _allowedLength = options.AllowedLength;
        _lookback = options.Lookback;
        _maxMatch = options.MaxMatchLength;
        _maxPenalty = options.MaxPenalty;
        _exempt = [.. exempt];
        _window = new int[options.Lookback + options.MaxMatchLength + 1];
    }

    /// <inheritdoc/>
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        int n = Window(history);
        if (n < Math.Max(1, _allowedLength))
        {
            return;
        }
        int last = _window[n - 1];
        _longest.Clear();
        for (int j = n - 2; j >= Math.Max(0, n - 1 - _lookback); j--)
        {
            if (_window[j] != last)
            {
                continue;
            }
            int length = 1;
            while (length < _maxMatch && j - length >= 0 && _window[j - length] == _window[n - 1 - length])
            {
                length++;
            }
            if (length >= _allowedLength)
            {
                int candidate = _window[j + 1];
                if (!_longest.TryGetValue(candidate, out int known) || length > known)
                {
                    _longest[candidate] = length;
                }
            }
        }
        foreach ((int token, int length) in _longest)
        {
            if ((uint)token >= (uint)logits.Length || _exempt.Contains(token))
            {
                continue;
            }
            logits[token] -= Math.Min(_maxPenalty, _multiplier * MathF.Pow(_base, length - _allowedLength));
        }
    }

    /// <summary>The latest non-exempt tokens of <paramref name="history"/>, at most the window size, copied into the scratch window.</summary>
    private int Window(IReadOnlyList<int> history)
    {
        int count = 0;
        for (int i = history.Count - 1; i >= 0 && count < _window.Length; i--)
        {
            int token = history[i];
            if (!_exempt.Contains(token))
            {
                _window[_window.Length - 1 - count] = token;
                count++;
            }
        }
        if (count < _window.Length)
        {
            Array.Copy(_window, _window.Length - count, _window, 0, count);
        }
        return count;
    }
}
