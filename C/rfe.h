/*****************************************************
 * recursive feature elimination lib                 *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *****************************************************/
#ifndef RFE_H_INCLUDED
#define RFE_H_INCLUDED

struct rfe_select_weight
{
    double w;
    double val;
    int fname;
};
typedef struct rfe_select_weight rfe_select_weight;

sample*
rfe_select_features(char *filename, sample *sample,
        int (*train)(struct sample*,double**,double*,int*,int),
        int depth, int jump, int leave_one_out, int skip, crossvalidation *cv, int verbose);

int
rfe_select_compare_weight_greater(const void *a, const void *b);

#endif //_RFE_H_INCLUDED
