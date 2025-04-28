/*****************************************************
 * golub feature selection                           *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *****************************************************/
#ifndef GOLUB_H_INCLUDED
#define GOLUB_H_INCLUDED

struct golub_select_score
{
    int fname;
    double score;
};
typedef struct golub_select_score golub_select_score;

sample* golub_select_features(char *filename, sample *sample, int (*train)(struct sample*,double**,double*,int*,int), int number, int verbose);
int golub_select_compare_score_greater(const void *a, const void *b);

#endif // GOLUB_H_INCLUDED
