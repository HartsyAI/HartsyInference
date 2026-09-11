using System.Collections.Generic;
using System.Text;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Resolves SwarmUI's <c>&lt;alternate:a, b&gt;</c>/<c>&lt;alt:a, b&gt;</c> (a different phrase every
/// step) and <c>&lt;fromto[N]:a, b&gt;</c> (switch phrase at step/fraction <c>N</c>) tags into a per-step
/// <see cref="PromptSchedule"/>. Replaces the old (never-wired-into-a-live-pipeline) <c>PromptScheduling</c>,
/// which targeted the Comfy-native <c>[a|b]</c>/<c>[from:to:when]</c> bracket grammar SwarmUI's 2026-09-01
/// prompt-parser update stopped emitting even for legacy-typed input — see
/// <see href="https://github.com/mcmonkeyprojects/SwarmUI">SwarmUI</see> commits <c>c800bc51</c>..<c>6e8c001f</c>.
/// A caller feeds this the prompt exactly as <see cref="PromptTagFlattening.Flatten"/> leaves it with
/// <c>flattenScheduling: false</c> — i.e. <c>&lt;weight[N]:...&gt;</c> already converted to <c>(text:N)</c>, but
/// <c>alt</c>/<c>alternate</c>/<c>fromto</c> tags still raw. Every other tag (parens weight groups included)
/// passes through untouched at every step, exactly like the old bracket-grammar version did.</summary>
public static class PromptTagScheduling
{
    /// <summary>True if the prompt contains at least one top-level <c>&lt;alternate:&gt;</c>/<c>&lt;alt:&gt;</c>/
    /// <c>&lt;fromto[N]:&gt;</c> tag (i.e. its conditioning changes across steps).</summary>
    public static bool HasScheduling(string prompt)
    {
        if (prompt is null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }
        int i = 0;
        while (i < prompt.Length)
        {
            if (prompt[i] != '<')
            {
                i++;
                continue;
            }
            int close = PromptTagFlattening.FindMatchingTag(prompt, i);
            if (close == -1)
            {
                i++;
                continue;
            }
            string content = prompt[(i + 1)..close];
            (string prefix, string? predata, _) = PromptTagFlattening.SplitTag(content);
            // A fromto whose predata is not a number is literal prose, not scheduling — same rule the weight
            // tag already follows, and what SwarmUI's own SwarmText.py reference node does.
            if (prefix is "alt" or "alternate" || (prefix == "fromto" && PromptTagFlattening.TryParseWhen(predata, out _)))
            {
                return true;
            }
            i = close + 1;
        }
        return false;
    }

    /// <summary>Resolves the prompt for a single step out of <paramref name="totalSteps"/>.</summary>
    public static string ResolveAt(string prompt, int step, int totalSteps)
    {
        if (prompt is null)
        {
            throw new ArgumentNullException(nameof(prompt));
        }
        StringBuilder result = new StringBuilder(prompt.Length);
        int i = 0;
        while (i < prompt.Length)
        {
            char c = prompt[i];
            if (c != '<')
            {
                result.Append(c);
                i++;
                continue;
            }
            int close = PromptTagFlattening.FindMatchingTag(prompt, i);
            if (close == -1)
            {
                result.Append(c);
                i++;
                continue;
            }
            string content = prompt[(i + 1)..close];
            (string prefix, string? predata, string data) = PromptTagFlattening.SplitTag(content);
            if (prefix is "alt" or "alternate")
            {
                string[] alternatives = PromptTagFlattening.SplitSmart(data);
                string chosen = alternatives.Length > 0 ? alternatives[step % alternatives.Length] : "";
                result.Append(ResolveAt(chosen, step, totalSteps));
                i = close + 1;
                continue;
            }
            if (prefix == "fromto" && PromptTagFlattening.TryParseWhen(predata, out float when))
            {
                string[] parts = PromptTagFlattening.SplitSmart(data);
                if (parts.Length == 2)
                {
                    string branch = step < SwitchStep(when, totalSteps) ? parts[0] : parts[1];
                    result.Append(ResolveAt(branch, step, totalSteps));
                    i = close + 1;
                    continue;
                }
            }
            // Not a scheduling tag: <weight[N]:...> has already been converted to (text:N) by
            // PromptTagFlattening upstream, and every other tag (region/segment/embed/lora/refcrop/etc.)
            // passes through untouched, unchanged across every step.
            result.Append(prompt, i, close - i + 1);
            i = close + 1;
        }
        return result.ToString();
    }

    /// <summary>Resolves all steps, deduplicates identical prompts, and returns the variant list plus a
    /// per-step index. When the prompt has no scheduling, returns a single variant for every step.</summary>
    public static PromptSchedule Resolve(string prompt, int totalSteps)
    {
        if (totalSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(totalSteps), "totalSteps must be positive.");
        }
        List<string> variants = new List<string>();
        Dictionary<string, int> seen = new Dictionary<string, int>();
        int[] stepToVariant = new int[totalSteps];
        for (int s = 0; s < totalSteps; s++)
        {
            string resolved = ResolveAt(prompt, s, totalSteps);
            if (!seen.TryGetValue(resolved, out int idx))
            {
                idx = variants.Count;
                variants.Add(resolved);
                seen[resolved] = idx;
            }
            stepToVariant[s] = idx;
        }
        return new PromptSchedule(variants, stepToVariant);
    }

    /// <summary>The (fractional) step a <c>&lt;fromto[when]:...&gt;</c> switches at, as a 1:1 port of
    /// <c>ParsedText.flatten</c> in SwarmUI's reference <c>SwarmText.py</c>: <c>when &lt; 1</c> is a fraction of
    /// <paramref name="totalSteps"/>, anything else is an absolute step index, and the comparison against the
    /// step stays in floating point. Returned rather than rounded to an int because rounding first shifts the
    /// boundary by a step for odd step counts (<c>0.5</c> of 5 steps switches after step 2, not after step 1).</summary>
    private static float SwitchStep(float when, int totalSteps) => when < 1f ? when * totalSteps : when;
}
