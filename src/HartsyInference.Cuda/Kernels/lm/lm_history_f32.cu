// Graph-capture decode: the repetition-penalty history, kept DISTINCT. Appends tokenId[0] (the token this step is about to embed)
// to history[0, *historyCount) only when it is not already there, so lm_repetition_penalty_f32, which penalizes every listed entry
// once, penalizes each distinct token once: the Hugging Face and llama.cpp convention, and RepetitionPenaltyStep's on the CPU. A
// list with repeats penalized a token once per occurrence, penalty^n after n occurrences, which wrecked long greedy outputs.
// Single thread: the scan is at most one read per distinct token so far, and a fixed 1x1x1 launch stays graph-replay-safe.
extern "C" {

__global__ void lm_history_append_distinct(
    int* __restrict__ history,
    int* __restrict__ historyCount,
    const int* __restrict__ tokenId)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int count = *historyCount;
    int token = tokenId[0];
    for (int i = 0; i < count; ++i)
    {
        if (history[i] == token) return;
    }
    history[count] = token;
    *historyCount = count + 1;
}

} // extern "C"
