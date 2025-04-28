/*****************************************************
 * golub feature selection                           *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *****************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data.h"
#include "utils.h"
#include "golub.h"

sample*
golub_select_features(char *filename, sample *sample, int (*train)(struct sample*,double**,double*,int*,int), int number, int verbose)
{
    int i = 0;
    int j = 0;
    int num_pos = 0, num_neg = 0;
    int dim  = sample->dim;
    int size = sample->size;
    int *remove = NULL;
    double *avg_neg = NULL;
    double *avg_pos = NULL;
    double *sd_neg  = NULL;
    double *sd_pos  = NULL;
    golub_select_score *scores = NULL;
    struct sample *stmp = NULL;
    struct sample *stmp_parcial = NULL;
    int parcial = 0;

    double *w = NULL;
    double margin = 0;
    int svs = 0;

    /*alloc memory*/
    avg_pos = (double*) malloc(dim*sizeof(double));
    if(avg_pos == NULL) { printf("Out of mem!!\n"); exit(1); }
    avg_neg = (double*) malloc(dim*sizeof(double));
    if(avg_neg == NULL) { printf("Out of mem!!\n"); exit(1); }
    sd_pos =  (double*) malloc(dim*sizeof(double));
    if(sd_pos  == NULL) { printf("Out of mem!!\n"); exit(1); }
    sd_neg =  (double*) malloc(dim*sizeof(double));
    if(sd_neg  == NULL) { printf("Out of mem!!\n"); exit(1); }

    /*calc average*/
    for(i = 0; i < dim; ++i)
    {
        num_neg = 0;
        num_pos = 0;
        avg_neg[i] = 0;
        avg_pos[i] = 0;
        for(j = 0; j < size; ++j)
        {
            if(sample->points[j].y == -1)
            {
                avg_neg[i] += sample->points[j].x[i];
                ++num_neg;
            }
            else
            {
                avg_pos[i] += sample->points[j].x[i];
                ++num_pos;
            }
        }
        avg_neg[i] /= num_neg;
        avg_pos[i] /= num_pos;
    }

    /*calc standard deviation*/
    for(i = 0; i < dim; ++i)
    {
        sd_neg[i] = 0;
        sd_pos[i] = 0;
        for(j = 0; j < size; ++j)
        {
            if(sample->points[j].y == -1) sd_neg[i] += pow(sample->points[j].x[i]-avg_neg[i], 2);
            else                          sd_pos[i] += pow(sample->points[j].x[i]-avg_pos[i], 2);
        }
        sd_neg[i] = sqrt(sd_neg[i]/(num_neg-1));
        sd_pos[i] = sqrt(sd_pos[i]/(num_pos-1));
    }

    /*alloc scores*/
    scores = (golub_select_score*) malloc(dim*sizeof(golub_select_score));
    if(scores == NULL) { printf("Out of mem!!\n"); exit(1); }

    /*calc scores*/
    for(i = 0; i < dim; ++i)
    {
        scores[i].score = fabs(avg_pos[i]-avg_neg[i])/(sd_pos[i]+sd_neg[i]);
        scores[i].fname = sample->fnames[i];
        if(verbose)
            printf("Score: %lf, Fname: %d\n", scores[i].score, scores[i].fname);
    }
    if(verbose) printf("----------------------------\n");

    if(verbose) printf("Dim: %d -- ", dim);

    /*training sample*/
    if(!train(sample, &w, &margin, &svs, 0))
    {
        free(w); w = NULL;
        if(verbose) printf("Treinamento falhou!\n");
        //break;
    }
    else
    {
        printf("Treinamento com sucesso...\n");
        printf("Margem = %lf, Vetores Suporte = %d\n", margin, svs);
        printf("----------------------------\n");
    }

    free(avg_pos);
    free(avg_neg);
    free(sd_pos);
    free(sd_neg);

    qsort(scores, dim, sizeof(golub_select_score), golub_select_compare_score_greater);

    stmp_parcial = data_copy_sample(sample);

    /*alloc remove*/
    remove = (int*) malloc((dim-number)*sizeof(int));
    if(remove == NULL) { printf("Out of mem!!\n"); exit(1); }
    for(i = 0; i < (dim-number); ++i)
    {
        if(verbose) printf("Score: %lf, Fname: %d\n", scores[i].score, scores[i].fname);
        remove[i] = scores[i].fname;

        stmp = data_remove_features(sample, remove, i+1, 0);

        if(verbose)
            printf("Dim: %d -- ", dim-i-1);

        /*training sample*/
        w = NULL;
        if(!train(stmp, &w, &margin, &svs, 0))
        {
            free(w); w = NULL;
            if(verbose) printf("Treinamento falhou!\n");
            parcial = 1;
            //data_free_sample(&stmp);
            break;
        }
        else
        {
            printf("Treinamento com sucesso...\n");
            printf("Margem = %lf, Vetores Suporte = %d\n", margin, svs);
            printf("----------------------------\n");
        }
        data_free_sample(&stmp_parcial);
        stmp_parcial = data_copy_sample(stmp);
    }

    /*save info*/
    //stmp = data_remove_features(sample, remove, (dim-number), 0);
    free(remove);
    free(scores);
    if(w) free(w);
    data_free_sample(&sample);
    if(parcial)
    {
        data_write(filename, stmp_parcial, 0);
        data_free_sample(&stmp);
        return stmp_parcial;
    }
    else
    {
        data_write(filename, stmp, 0);
        data_free_sample(&stmp_parcial);
        return stmp;
    }
}

/*----------------------------------------------------------*
 * Returns 1 for a > b, -1 a < b, 0 if a = b                *
 *----------------------------------------------------------*/
int
golub_select_compare_score_greater(const void *a, const void *b)
{
    const golub_select_score *ia = (const golub_select_score*) a;
    const golub_select_score *ib = (const golub_select_score*) b;

    /*                V (greater)*/
    return (ia->score > ib->score) - (ia->score < ib->score);
}

