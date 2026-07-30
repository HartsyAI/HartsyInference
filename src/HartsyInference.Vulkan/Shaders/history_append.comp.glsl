// history_append: appends the device-resident token id into the repetition-penalty history buffer at
// the current count, then increments the count — a fixed 1-thread launch (matches CudaBackend's
// LaunchHistoryAppend). Guards against writing past capacity (the count buffer still increments past
// capacity so a caller can detect overflow; the CPU repetition-penalty reference already ignores
// out-of-range indices, so an over-capacity history simply stops gaining new penalized entries rather
// than corrupting memory).
//
// Compile:
//   glslc history_append.comp.glsl -o history_append_f32.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer History_  { uint history_data[]; };
layout(set = 0, binding = 1) buffer Count_    { uint count_data[]; };
layout(set = 0, binding = 2) readonly buffer TokenId_ { uint token_data[]; };

layout(push_constant) uniform Push {
    uint capacity;
} pc;

void main() {
    if (gl_GlobalInvocationID.x != 0u) return;
    uint idx = count_data[0];
    if (idx < pc.capacity) {
        history_data[idx] = token_data[0];
    }
    count_data[0] = idx + 1u;
}
