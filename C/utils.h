/*****************************************************
 * utils lib                                         *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *****************************************************/
#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include "data.h"
//#include "kernel.h"

/*double linked list of integers*/
struct int_dll
{
    int index;
    struct int_dll *prev;
    struct int_dll *next;
};
typedef struct int_dll int_dll;

/*parametros do k-fold cross-validation*/
typedef struct crossvalidation
{
    int fold;
    int qtde;
    int jump;
    int *seed;
    double erro_inicial;
    double erro_atual;
    double erro_limite;
} crossvalidation;

/*int dll functions*/
int_dll* utils_int_dll_create();
int_dll* utils_int_dll_remove(int_dll **node);
int_dll* utils_int_dll_append(int_dll *list);
void utils_int_dll_free(int_dll **head);

/*random stuff*/
void utils_initialize_random();
int* utils_initialize_random_list(int size);
void utils_shuffle(int **list, int size);

/*some util functions for every one*/
double utils_max(double one, double two);
double utils_min(double one, double two);

/*algortim testing functions*/
double utils_leave_one_out(sample *sample, int (*train)(struct sample*,double**,double*,int*,int), int skip, int verbose);
double utils_k_fold(sample *sample, int (*train)(struct sample*,double**,double*,int*,int), int fold, int seed, int verbose);
void utils_validation(sample *train_sample, sample *test_sample, int (*train)(struct sample*,double**,double*,int*,int), int fold, int qtde, int verbose);

double utils_leave_one_out_matrix(sample *sample, int (*train)(struct sample*,double**,double*,int*,int), double **o_matrix, int skip, int verbose);

double* utils_get_weight(sample *sample);
double* utils_get_dualweight(sample *sample);
double* utils_get_dualweight_prodint(sample *sample);
double utils_norm(double *v, int dim, double q);

void utils_plot_results(sample *sample, double shade);

void utils_plot_2d(sample *sample, double *w, char *base_entrada, char *train);
void utils_plot_3d(sample *sample, double *w, char *base_entrada, char *train);
#endif // UTILS_H_INCLUDED
