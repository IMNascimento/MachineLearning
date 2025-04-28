/*****************************************************
 * recursive feature elimination lib                 *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *****************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "utils.h"
#include "rfe.h"

#define primeiro_decaimento 0.25

/*----------------------------------------------------------*
 * RFE feature selection                                    *
 *----------------------------------------------------------*/
sample*
rfe_select_features(char *filename, sample *sample,
        int (*train)(struct sample*,double**,double*,int*,int),
        int depth, int jump, int leave_one_out, int skip, crossvalidation *cv, int verbose)
{
    register int i = 0, j = 0;
    int svcount = 0;
    int *features = NULL;
    int dim = sample->dim;
    int level = 0;
    double margin = 0;
    //double max_time = sample->max_time;
    double *w = NULL;
    double *novo_w = NULL;
    double leave_oo = 0;
    struct sample *stmp = sample;
    rfe_select_weight *weight = NULL;
    int leveljump = 0;
    double errokfold = 0;

    double START_TIME = 100.0f*clock()/CLOCKS_PER_SEC;

    double tempo_parcial = 0;
    double margem_parcial = 0;
    int dim_parcial = 0;
    int svs_parcial = 0;
    int *features_parcial = NULL;
    struct sample *stmp_parcial = NULL;
    int parcial = 0; //verifica se última solução é uma solução recuperada (parcial)
    //int ftime = 1;

    /*error check*/
    if(depth < 1 || depth >= dim)
    {
        printf("Profundidade invalida!\n"); exit(1);
    }

    /*creating an array to place selected features*/
    features = (int*) malloc(depth*sizeof(int));
    if(features == NULL) { printf("Out of mem!!%d\n", depth); exit(1); }

    /*inicializando o cross-validation*/
    if(cv->qtde > 0)
    {
        //utils_initialize_random();
        cv->seed = (int*) malloc(cv->qtde*sizeof(int));
        if(cv->seed == NULL) { printf("Out of memory\n"); exit(1); }
        for(i = 0; i < cv->qtde; i++)
            cv->seed[i] = i; //rand();
        cv->erro_inicial = 0;
        cv->erro_atual   = 0;
    }
    double n0 = 1;

    while(1)
    {
        svcount = 0;
        margin  = 0;

        //if(level != 0) // || level == depth) //else stmp->max_time = max_time;
        if(level == 1)
            n0 = stmp->max_time *= primeiro_decaimento;
        else if(level > 1)
            stmp->max_time = n0 * exp(-stmp->mult_tempo * ((double)dim/(dim-level)));

        /*training sample*/
        if(!train(stmp, &w, &margin, &svcount, 0))
        {
            free(w); w = NULL;
            if(verbose) printf("Treinamento falhou!\n");
            if(level > 0)
            {
                printf("---------------\n :: FINAL :: \n---------------\n");

                printf("Features Escolhidas: ");
                for(i = 0; i < stmp_parcial->dim-1; ++i) printf("%3d,", stmp_parcial->fnames[i]);
                printf("%3d\n", stmp_parcial->fnames[i]);

                if(cv->qtde > 0)
                {
                    if((dim-dim_parcial) % cv->jump != 0)
                    {
                        for(cv->erro_atual = 0, i = 0; i < cv->qtde; i++)
                            cv->erro_atual += utils_k_fold(stmp, train, cv->fold, cv->seed[i], 0);
                        errokfold = cv->erro_atual / cv->qtde;
                    }
                    printf("Dim: %d, Margem: %lf, SVs: %d, Erro %d-fold: %.2lf%%\n", dim_parcial, margem_parcial, svs_parcial, cv->fold, errokfold);
                }
                else
                    printf("Dim: %d, Margem: %lf, SVs: %d\n", dim_parcial, margem_parcial, svs_parcial);
                printf("---------------\nTempo total: %.3lfs\n\n", tempo_parcial);
                parcial = 1;
                data_write(filename, stmp_parcial, 0);
                //free(weight);
            }
            break;
        }

        margem_parcial = margin;
        svs_parcial = svcount;
        tempo_parcial = (100.0f*clock()/CLOCKS_PER_SEC-START_TIME)/100.0f;
        dim_parcial = dim-level;

        data_free_sample(&stmp_parcial);
        stmp_parcial = data_copy_sample(stmp);

        free(features_parcial);
        if(level-jump > 0)
        {
            features_parcial = (int*) malloc((level-jump)*sizeof(int));
            if(features_parcial == NULL) { printf("Out of mem!!%d\n", depth); exit(1); }
        }
        for(i = 0; i < level-jump; ++i)
            features_parcial[i] = features[i];

        if(cv->qtde > 0)
        {
            if(level == 0)
            {
                for(i = 0; i < cv->qtde; i++)
                    cv->erro_inicial += utils_k_fold(stmp, train, cv->fold, cv->seed[i], 1);
                errokfold = cv->erro_inicial / cv->qtde;
            }
            else if(level % cv->jump == 0)
            {
                for(cv->erro_atual = 0, i = 0; i < cv->qtde; i++)
                    cv->erro_atual += utils_k_fold(stmp, train, cv->fold, cv->seed[i], 0);
                errokfold = cv->erro_atual / cv->qtde;
            }
        }

        /*leave one out*/
        if(leave_one_out)
        {
            leave_oo = utils_leave_one_out(stmp, train, skip, 0);
            printf("LeaveOO -- Dim: %d, Margem: %lf, LeaveOO: %lf, SVs: %d\n", (dim-level), margin, leave_oo, svcount);
        }
        else if(verbose)
        {
            if(cv->qtde > 0 && level % cv->jump == 0)
                printf("Dim: %d, Margem: %lf, SVs: %d, Erro %d-fold: %.2lf%%\n", (dim-level), margin, svcount, cv->fold, errokfold);
            else
                printf("Dim: %d, Margem: %lf, SVs: %d\n", (dim-level), margin, svcount);
                //printf("Dim: %d, Margem: %lf, Distancia entre os centros: %f, SVs: %d\n", (dim-level), data_get_dist_centers(stmp), margin, svcount);
        }

        /*allocating space for w/feature*/
        weight = (rfe_select_weight*) malloc(stmp->dim*sizeof(rfe_select_weight));
        if(weight == NULL) { printf("Error: Out of memory\n"); exit(1); }

        /*copying elements of array*/
        for(i = 0; i < stmp->dim; ++i)
        {
            weight[i].w = w[i];
            weight[i].fname = stmp->fnames[i];
        }

        /*sorting*/
        qsort(weight, stmp->dim, sizeof(rfe_select_weight), rfe_select_compare_weight_greater);

        printf("---------------------\n");
        if(verbose > 1)
        {
            for(i = 0; i < stmp->dim; ++i)
                printf("%d: %lf\n", weight[i].fname, weight[i].w);
            printf("---------------------\n");
        }

        /*stopping criterion*/
        if(level >= depth || (cv->qtde > 0 && cv->erro_atual-cv->erro_inicial > cv->erro_limite))
        {
            printf("---------------\n :: FINAL :: \n---------------\n");
            //if(stmp->dim < 50)
            printf("Features Escolhidas: ");
            for(i = 0; i < stmp->dim-1; ++i) printf("%3d,", stmp->fnames[i]);
            printf("%3d\n", stmp->fnames[i]);

            printf("---------------\nFeatures Eliminadas: ");
            for(i = 0; i < leveljump-1; ++i) printf("%3d,", features[i]);
            printf("%3d\n", features[i]);

            if(cv->qtde > 0)
            {
                if(level % cv->jump != 0)
                {
                    for(cv->erro_atual = 0, i = 0; i < cv->qtde; i++)
                        cv->erro_atual += utils_k_fold(stmp, train, cv->fold, cv->seed[i], 0);
                    errokfold = cv->erro_atual / cv->qtde;
                }
                printf("Dim: %d, Margem: %lf, SVs: %d, Erro %d-fold: %.2lf%%\n", dim-level, margin, svcount, cv->fold, errokfold);
            }
            else
                printf("Dim: %d, Margem: %lf, SVs: %d\n", dim-level, margin, svcount);

            printf("---------------\nTempo total: %.3lfs\n\n", (100.0f*clock()/CLOCKS_PER_SEC-START_TIME)/100.0f);
            data_write(filename, stmp, 0);
            free(weight);
            break;
        }

        if(level+jump > depth)
            leveljump = depth;
        else
            leveljump = level+jump;

        /*manutencao do w do pai para o IMA Primal*/
        if(stmp->kernel_type == 9)
        {
            for(j = 0; j < stmp->dim; ++j)
                for(i = level; i < leveljump; ++i)
                    if(weight[i-level].w == w[j])
                        w[j] = 0;

            novo_w = (double*) malloc(((sample->dim)-leveljump)*sizeof(double));
            if(novo_w == NULL) { printf("Error: Out of memory\n"); exit(1); }

            for(i = 0, j = 0; j < stmp->dim; ++j)
                if(w[j] != 0)
                    novo_w[i++] = w[j];
            //novo_w[i] = w[j]; //bias nao copia mais
            free(w);
            w = novo_w;
            novo_w = NULL;
        }
        else //IMA Dual e SMO
        {
            free(w); w = NULL;
        }

        if(stmp != sample) data_free_sample(&stmp);

        /*saving removed feature name*/
        for(i = level; i < leveljump; ++i)
        {
            printf("Removendo w = %lf\n", weight[i-level].w);
            features[i] = weight[i-level].fname;
        }
        printf("---------------------\n");
        free(weight);

        /*increment*/
        if(level+jump > depth)
        {
            level = depth;
            jump  = 0;
        }
        else
            level += jump;
        /*get temp data struct*/
        stmp = data_remove_features(sample, features, level, verbose);
    }
    /*free stuff*/
    free(features);
    free(features_parcial);
    if(cv->qtde > 0) free(cv->seed);
    free(w);
    data_free_sample(&sample);
    if(parcial)
    {
        data_free_sample(&stmp);
        return stmp_parcial;
    }
    else
    {
        data_free_sample(&stmp_parcial);
        return stmp;
    }
}

/*----------------------------------------------------------*
 * Returns 1 for a > b, -1 a < b, 0 if a = b                *
 *----------------------------------------------------------*/
int
rfe_select_compare_weight_greater(const void *a, const void *b)
{
    const rfe_select_weight *ia = (const rfe_select_weight*) a;
    const rfe_select_weight *ib = (const rfe_select_weight*) b;

    /*                 V (greater)*/
    return (fabs(ia->w) > fabs(ib->w)) - (fabs(ia->w) < fabs(ib->w));
}
