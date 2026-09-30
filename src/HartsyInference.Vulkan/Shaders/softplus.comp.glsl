// softplus: ln(1 + e^x) over F32, threshold 20, matching SoftplusReference (torch.nn.functional.softplus).
// Evaluated in float with a hand-rolled exp and log1p so a tiny result keeps its precision (the reference gets
// that from double). out_data may alias in_data (one thread reads then writes one element).

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) buffer Out_ { float out_data[]; };
layout(set = 0, binding = 1) readonly buffer In_ { float in_data[]; };

layout(push_constant) uniform Push {
    uint n;
} pc;

// exp accurate to about one ulp on every driver: the built-in is allowed a few ulp and on some hardware lowers to
// ex2(x * log2(e)), whose product error grows with |x|. Range-reduce by hand with a split ln2, then a degree-7 series.
float accExp(float x) {
    if (x > 88.0) return uintBitsToFloat(0x7F800000u);
    if (x < -87.0) return 0.0;
    float kf = floor(x * 1.44269504089 + 0.5);
    float r = fma(-kf, 0.693145751953125, x);
    r = fma(-kf, 1.42860677e-6, r);
    float p = 1.0 + r * (1.0 + r * 0.5 * (1.0 + r * (1.0 / 3.0) * (1.0 + r * 0.25 * (1.0 + r * 0.2
        * (1.0 + r * (1.0 / 6.0) * (1.0 + r * (1.0 / 7.0)))))));
    return ldexp(p, int(kf));
}

// softplus(x) = max(x, 0) + log1p(exp(-|x|)); log1p by the atanh series, which keeps its relative precision
// for tiny arguments where log(1 + t) cannot (the built-in log is only absolutely accurate near 1).
float softplusScalar(float x) {
    if (x > 20.0) return x;
    float t = accExp(-abs(x));
    float s = t / (2.0 + t);
    float q = s * s;
    float series = 1.0 + q * (1.0 / 3.0 + q * (1.0 / 5.0 + q * (1.0 / 7.0 + q * (1.0 / 9.0 + q * (1.0 / 11.0
        + q * (1.0 / 13.0 + q * (1.0 / 15.0 + q * (1.0 / 17.0))))))));
    return max(x, 0.0) + 2.0 * s * series;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    out_data[i] = softplusScalar(in_data[i]);
}
