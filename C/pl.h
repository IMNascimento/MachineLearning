#ifndef PL_H_INCLUDED
#define PL_H_INCLUDED

//typedef struct datapl
//{
//    sample *z;
//    double *w;
//    double norm;
//} data;

int linear_programming(sample *sample, double **w, double *margin, int *svs, int verbose);

#endif // PL_H_INCLUDED
