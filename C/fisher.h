/*****************************************************
 * fisher feature selection                          *
 *                                                   *
 * Saulo Moraes Villela <saulomv@gmail.com>          *
 * sep 08, 2011                                      *
 *****************************************************/
#ifndef FISHER_H_INCLUDED
#define FISHER_H_INCLUDED

struct fisher_select_score
{
    int fname;
    double score;
};
typedef struct fisher_select_score fisher_select_score;

sample* fisher_select_features(char *filename, sample *sample, int (*train)(struct sample*,double**,double*,int*,int), int number, int verbose);
int fisher_select_compare_score_greater(const void *a, const void *b);

#endif // FISHER_H_INCLUDED
