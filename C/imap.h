#ifndef IMAP_H_INCLUDED
#define IMAP_H_INCLUDED

typedef struct data
{
    sample *z;
    double *func;
    double *w;
    double norm;
} data;

int imap(sample *sample, double **w, double *margin, int *svs, int verbose);
int imap_fixed_margin_perceptron(data *data, double gamma, int *passes, int *ctot, int *index, double q, double max_time);

#endif // IMAP_H_INCLUDED
