#ifndef IMADUAL_H_INCLUDED
#define IMADUAL_H_INCLUDED

typedef struct datadual
{
    sample *z;
    double **K;
    double *func;
    double norm;
} datadual;

int imadual(sample *sample, double **w, double *margin, int *svs, int verbose);
int imadual_fixed_margin_perceptron(datadual *data, double gamma, int *passes, int *ctot, int *index, double max_time);

#endif // IMADUAL_H_INCLUDED
