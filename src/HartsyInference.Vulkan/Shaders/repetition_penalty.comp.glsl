// repetition_penalty: HF-convention repetition penalty (divide positive logits, multiply negative ones),
// applied SEQUENTIALLY over the history buffer by a single thread — matching
// HartsyInference.LLM.Sampling.RepetitionPenaltyStep's CPU reference exactly, including its compounding
// behavior when the same token appears more than once in history (each occurrence re-reads the
// already-penalized value). A naive one-thread-per-history-entry parallelization would race on repeated
// tokens and silently diverge from that reference, so this deliberately stays single-threaded — cheap
// relative to one decode step's matmuls, and correctness against the documented HF convention matters
// far more here than parallelism.
//
// Compile:
//   glslc repetition_penalty.comp.glsl -o repetition_penalty_f32.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Logits_   { float logits_data[]; };
layout(set = 0, binding = 1) readonly buffer History_ { uint history_data[]; };
layout(set = 0, binding = 2) readonly buffer Count_   { uint count_data[]; };

layout(push_constant) uniform Push {
    uint vocabSize;
    float penalty;
} pc;

void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    uint count = count_data[0];
    for (uint i = 0u; i < count; i++) {
        uint token = history_data[i];
        if (token >= pc.vocabSize) continue;
        float logit = logits_data[token];
        logits_data[token] = logit > 0.0 ? logit / pc.penalty : logit * pc.penalty;
    }
}
