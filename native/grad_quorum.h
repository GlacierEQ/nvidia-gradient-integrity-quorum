#ifndef GRAD_QUORUM_H
#define GRAD_QUORUM_H
typedef struct { int rank; float grad_norm; int finite; } RankReport;
typedef enum { GRAD_COMMIT, GRAD_ISOLATE, GRAD_ABORT } GradDecision;
GradDecision grad_evaluate(const RankReport *ranks, int n, float min_healthy_ratio,
                           int *poison_out, int *poison_n, int *healthy_out);
#endif
