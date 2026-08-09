/* Babel: C — finite gradient quorum before allreduce commit. */
#include "grad_quorum.h"
#include <math.h>

GradDecision grad_evaluate(const RankReport *ranks, int n, float min_healthy_ratio,
                           int *poison_out, int *poison_n, int *healthy_out) {
    int poison_count = 0, healthy = 0;
    for (int i = 0; i < n; i++) {
        int bad = !ranks[i].finite || !isfinite(ranks[i].grad_norm);
        if (bad) {
            if (poison_out && poison_count < n) poison_out[poison_count] = ranks[i].rank;
            poison_count++;
        } else healthy++;
    }
    if (poison_n) *poison_n = poison_count;
    if (healthy_out) *healthy_out = healthy;
    if (n <= 0 || healthy == 0) return GRAD_ABORT;
    float ratio = (float)healthy / (float)n;
    if (poison_count > 0 && ratio >= min_healthy_ratio) return GRAD_ISOLATE;
    if (ratio < min_healthy_ratio) return GRAD_ABORT;
    return GRAD_COMMIT;
}
