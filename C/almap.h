#ifndef ALMAP_H_INCLUDED
#define ALMAP_H_INCLUDED

int almap(sample *sample, double **w, double *margin, int *svs, int verbose);
int imap_fixed_margin_perceptron(data *data, double gamma, int *passes, int *ctot, int *index, double q, double max_time);

#endif // ALMAP_H_INCLUDED
