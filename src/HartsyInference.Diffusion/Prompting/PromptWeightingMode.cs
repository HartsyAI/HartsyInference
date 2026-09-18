namespace HartsyInference.Diffusion.Prompting;

/// <summary>Which of SwarmUI's two prompt-weighting mechanisms a model family must use for <c>(word:1.5)</c> parity.
/// SwarmUI picks between them at encode time by probing the loaded tokenizer stack —
/// <c>use_attn_token_weights = not token_batches_have_weights(clip.tokenize("(x:2)"))</c>
/// (<c>SwarmText.py:553</c>) — so a family lands on <see cref="CondScale"/> only when EVERY tokenizer arm sets
/// ComfyUI's <c>disable_weights=True</c>; one weight-keeping arm (a CLIP-L pooled tower, a T5 arm) puts the whole
/// family back on <see cref="ComfyBlend"/>.</summary>
public enum PromptWeightingMode
{
    /// <summary>No weighting: the family never passes through <c>SwarmTextEncodeAdvanced</c>, so dropping weights IS
    /// parity. Music families only (<c>WorkflowGenerator.cs:2584-2599</c>).</summary>
    None,

    /// <summary>Weights survive tokenization, so ComfyUI blends them into the encoder OUTPUT:
    /// <c>z = (z − z_empty)·w + z_empty</c> where <c>z_empty</c> is the same encoder run on <c>gen_empty_tokens</c>
    /// (<c>comfy/sd1_clip.py:54-63</c>).</summary>
    ComfyBlend,

    /// <summary>Every tokenizer arm discards weights, so SwarmUI encodes at weight 1.0 and then scales each token's
    /// cond row by its weight, right-aligned (<c>pos = condLen − len(batch) + i</c>, out-of-range skipped) —
    /// <c>multiply_cond_by_token_weights</c>, <c>SwarmText.py:256-271</c>. A prompt-wide uniform weight instead
    /// scales the whole cond plus <c>pooled_output</c>.</summary>
    CondScale,

    /// <summary><see cref="CondScale"/> plus the <c>SwarmAttnTokenWeights</c> joint-attention patch
    /// (<c>attn1_token_weight_patch</c>, <c>SwarmText.py:281-312</c>): cond slots only, <c>v[:,pos] *= w</c> for
    /// <c>w &lt; 1</c> and an attention logit bias of <c>(w − 1)·2</c> for <c>w &gt; 1</c>. SwarmUI inserts that node
    /// for Krea2 alone (<c>WorkflowGenerator.cs:965-972</c>, gated on <c>ModelSpecificEnhancements</c>), even though
    /// Flux/Chroma/Qwen-Image/HunyuanVideo expose the same <c>img_slice</c> hook.</summary>
    CondScaleWithAttention
}
