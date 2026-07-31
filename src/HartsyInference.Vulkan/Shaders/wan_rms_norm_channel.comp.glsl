// wan_rms_norm_channel: Wan2.2 VAE channel-wise RMS norm (vae.py `RMS_norm`: F.normalize over dim=1 *
// gamma * sqrt(C)). x, out: [B, C, spatial] with C on dim 1 (stride = spatial). gamma: [C], optional
// (hasGamma=0 skips it, matching a null gamma). One invocation per (b, s) position, reduces over C twice
// (sumSq pass, then scale+write pass) — mirrors CudaBackend's wan_vae_rms_norm_channel and IBackend's CPU
// reference (eps applied to the L2 not the mean). Accumulates in float, not double — shaderFloat64 isn't
// guaranteed device-wide (this backend targets AMD/Intel too) and C is small (<= a few hundred), so the
// float-vs-double delta is well below anything visible; see the regression test's tolerance.
//
// Compile:
//   glslc wan_rms_norm_channel.comp.glsl -o wan_rms_norm_channel_f32.spv

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) readonly buffer In_    { float in_data[];    };
layout(set = 0, binding = 1) readonly buffer Gamma_ { float gamma_data[]; };
layout(set = 0, binding = 2) writeonly buffer Out_  { float out_data[];   };

layout(push_constant) uniform Push {
    uint c;
    uint spatial;
    float eps;
    float sqrtC;
    uint hasGamma;
    uint numPos;
} pc;

void main() {
    uint pos = gl_GlobalInvocationID.x;
    if (pos >= pc.numPos) return;

    uint b = pos / pc.spatial;
    uint s = pos % pc.spatial;
    uint baseB = b * pc.c * pc.spatial;

    float sumSq = 0.0;
    for (uint ci = 0; ci < pc.c; ci++) {
        float v = in_data[baseB + ci * pc.spatial + s];
        sumSq += v * v;
    }
    float denom = max(sqrt(sumSq), pc.eps);
    float scale = pc.sqrtC / denom;

    for (uint ci = 0; ci < pc.c; ci++) {
        uint off = baseB + ci * pc.spatial + s;
        float gv = (pc.hasGamma != 0) ? gamma_data[ci] : 1.0;
        out_data[off] = in_data[off] * scale * gv;
    }
}
