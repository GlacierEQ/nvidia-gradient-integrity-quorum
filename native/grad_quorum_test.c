#include "grad_quorum.h"
#include "grad_quorum.c"
#include <stdio.h>
#include <math.h>
int main(void) {
    RankReport r[3] = {{0,1.f,1},{1,NAN,0},{2,1.2f,1}};
    int poison[3], pn, healthy;
    GradDecision d = grad_evaluate(r, 3, 0.5f, poison, &pn, &healthy);
    if (d != GRAD_ISOLATE || pn != 1 || poison[0] != 1) return 1;
    RankReport clean[2] = {{0,1.f,1},{1,1.1f,1}};
    d = grad_evaluate(clean, 2, 0.5f, poison, &pn, &healthy);
    if (d != GRAD_COMMIT) return 2;
    printf("ok\n");
    return 0;
}
