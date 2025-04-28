/*****************************************************
 * Admissible Ordered Search                         *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *                                                   *
 * Saulo Moraes Villela <saulomv@gmail.com>          *
 * 2009, 2011                                        *
 *****************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "data.h"
#include "utils.h"
#include "kernel.h"
#include "aos.h"
#include "pl.h"
#include "smo.h"

#define HASH_SIZE 161387
//#define HASH_SIZE 319993
//#define HASH_SIZE 999983
//#define HASH_SIZE 1686049
#define HASH_WIDTH 100
//#define HASH_WIDTH 200
//#define HASH_WIDTH 500
//#define HASH_WIDTH 2000
#define MAX_BREATH 99999
#define MAX_DEPTH 99999
#define MAX_HEAP 500000
//#define MAX_HEAP 1000000
//#define MAX_HEAP 30000000
//#define MAX_HEAP 50000000
#define NUM_ERROR_EPS 0.05
#define primeiro_decaimento 0.5

int contheap         = 0;
int contheapreins    = 0;
int conthash         = 0;
int contprooning     = 0;
int maxheapsize      = 0;
int contnaoheap      = 0;
int conthashnaoheap  = 0;
int contexpandidos   = 0;
int contprojetados   = 0;
int contprojtreinados= 0;

double START_TIME, tempo_inicial, max_time_orig;

struct sample *stmp_parcial = NULL;
double tempo_parcial;
double margem_parcial;
int dim_parcial;
int svs_parcial;

int contheap_parcial;
int contheapreins_parcial;
int conthash_parcial;
int contprooning_parcial;
int maxheapsize_parcial;
int contnaoheap_parcial;
int conthashnaoheap_parcial;
int contexpandidos_parcial;
int contprojetados_parcial;
int contprojtreinados_parcial;
int sobraprojecoes_parcial;

double errokfold = 0;

double n0 = 1;
int dim_orig = 1;

/*----------------------------------------------------------*
 * Run A* feature selection main loop                       *
 *----------------------------------------------------------*/
void
aos_select_feature_main_loop(sample *sample, aos_select_heap **heapp,
        aos_select_hash **hashp,
        int (*train)(struct sample*,double**,double*,int*,int),
        int breadth, double bonus, int *lool, int startdim,
        int look_ahead_depth, double *g_margin, int cut, int skip,
        int startover, int doleave_oo, int depth, int forma_ordenacao,
        int forma_escolha, int ftime, crossvalidation *cv, int verbose)
{
    register int i = 0, j = 0, k = 0;
    int level = 0;
    int *ofnames = NULL;
    int svcount = 0;
    //int trained = 0;
    double margin = 0;
    double omargin = 0;
    double wnorm = 0;
    double *w = NULL;
    double *w_manut = NULL; //vetor w de manutencao p/ look ahead
    double tpmargin = 0;
    double sumnorm  = 0;
    double leave_oo = -1;
    aos_select_gamma *gtmp = NULL;
    aos_select_weight *weight = NULL;
    aos_select_hash *hash = *hashp;
    aos_select_heap *heap = *heapp;

    int dim  = sample->dim;
    int size = sample->size;
    double q = sample->q;

    int loolflag = 0; //fechar uma dimensao

    if(heap->size == 0)
    {
        if(ftime && sample->kernel_type == 9) //primeira dimensão -- solução exata primal
        {
            if(sample->q == 2)
            {
                sample->kernel_type = 0;
                if(!smo_train(sample, &w, &margin, &svcount, 0))
                {
                    free(w);
                    if(verbose > 1) printf("Treinamento falhou!\n");
                    return;
                }
                sample->kernel_type = 9;
            }
            else if(sample->q == 1)
            {
                if(!linear_programming(sample, &w, &margin, &svcount, 0))
                {
                    free(w);
                    if(verbose > 1) printf("Treinamento falhou!\n");
                    return;
                }
            }
        }
        else if(ftime && sample->mult_tempo == 2) //primeira dimensão -- solução exata dual -- "gambiarra"
        {
            if(!smo_train(sample, &w, &margin, &svcount, 0))
            {
                free(w);
                if(verbose > 1) printf("Treinamento falhou!\n");
                return;
            }
        }
        else //não primeira dimensão ou não IMA
        {
            /*Training*/
            if(!train(sample, &w, &margin, &svcount, 0))
            {
                free(w);
                if(verbose > 1) printf("Treinamento falhou!\n");
                return;
            }
            //if(ftime) sample->max_time /= sample->mult_tempo;
        }
    }
    else //if(heap->size > 0)
    {
        gtmp = aos_select_heap_pop(heap);

        if(gtmp->level > depth)
        {   /*eliminar nodo com nivel maior que a profundidade desejada*/
            //aos_select_hash_set_null(hash, gtmp);
            //free(gtmp->fnames);
            //free(gtmp->w);
            //free(gtmp);
            gtmp = NULL;
            return;
        }

        w = gtmp->w;
        if(gtmp->rgamma > 0)
        {
            //trained = heap->elements[1]->train;
            margin  = gtmp->rgamma;
            svcount = gtmp->sv;
        }
        else
        {   /*Training*/
            margin = gtmp->pgamma;
            sample->bias = gtmp->bias;
            if(!train(sample, &w, &margin, &svcount, 0))
            {   /*training failed, remove this option*/
                free(w);
                //free(gtmp->fnames);
                //free(gtmp->w);
                //free(gtmp);
                gtmp = NULL;
                if(verbose > 1) printf("Treinamento falhou!\n");
                return;
            }

            if(forma_escolha == 2)
                gtmp->value = margin * data_get_dist_centers(sample, -1);
            else
                gtmp->value = margin;
            gtmp->rgamma = margin;
            gtmp->train  = 1;

            contprojtreinados++;

            /*verifica se realmente eh o melhor ou se jah chegou na profundidade desejada*/
            if(gtmp->value < heap->elements[1]->value || gtmp->level == depth)
            {   /*reinsert the node into heap*/
                gtmp->w  = w;
                gtmp->sv = svcount;
                aos_select_heap_insert(heap, gtmp, 0);
                return;
            }
        }
        /*this is the best solution, continue*/
        contexpandidos++;
        //ofnames = gtmp->fnames;
        ofnames = (int*) malloc(gtmp->level*sizeof(int));
        memcpy(ofnames, gtmp->fnames, gtmp->level*sizeof(int));
        level   = gtmp->level;
        omargin = gtmp->pgamma;
        //aos_select_hash_set_null(hash, gtmp);
        //gtmp->fnames = NULL;
        //free(gtmp->w);
        //gtmp->w = NULL;
        //free(gtmp);
        gtmp = NULL;
        if(level > *lool)
        {
            (*lool) = level;
            loolflag = 1; //fechou a dimensao
        }
    }

    /*some verbose*/
    if(verbose)
        if(level > 0)
        {
            printf("-> Expandindo as features (");
            for(i = 0; i < level-1; ++i)
                printf("%d,", ofnames[i]);
            printf("%d", ofnames[i]);
            printf(") -- Margem: %lf, pMargem: %lf, Nivel: %d\n", margin, omargin, level);
        }

    /*calculating leave one out, if it's the first to hit this level*/
    if((*lool) == level && (loolflag == 1 || level == 0))
    {
        loolflag = 0;
        if(cv->qtde > 0)
        {
            if(level == 0)
            {
                for(cv->erro_inicial = 0, i = 0; i < cv->qtde; i++)
                    cv->erro_inicial += utils_k_fold(sample, train, cv->fold, cv->seed[i], 1);
                errokfold = cv->erro_inicial / cv->qtde;
            }
            else if(level % cv->jump == 0)
            {
                for(cv->erro_atual = 0, i = 0; i < cv->qtde; i++)
                    cv->erro_atual += utils_k_fold(sample, train, cv->fold, cv->seed[i], 0);
                errokfold = cv->erro_atual / cv->qtde;
            }
        }

        if(doleave_oo)
        {
            leave_oo = utils_leave_one_out(sample, train, skip, 0);
            printf("Leave One Out -- Dim: %d, Margem: %lf, LeaveOO: %lf, SVs: %d, Tempo: %.3lfs\n", (startdim-level), margin, leave_oo, svcount, ((100.0f*clock()/CLOCKS_PER_SEC)-tempo_inicial)/100.0f);
        }
        else
        {
            leave_oo = -1;
            printf("--- --- --- --- --- --- --- ---\n");
            if(cv->qtde > 0 && level % cv->jump == 0)
                printf("Dim: %d, Margem: %lf, SVs: %d, Erro %d-fold: %.2lf%%, Tempo: %.3lfs", (startdim-level), margin, svcount, cv->fold, errokfold, ((100.0f*clock()/CLOCKS_PER_SEC)-tempo_inicial)/100.0f);
            else
                printf("Dim: %d, Margem: %lf, SVs: %d, Tempo: %.3lfs", (startdim-level), margin, svcount, ((100.0f*clock()/CLOCKS_PER_SEC)-tempo_inicial)/100.0f);
            if(level > 0)
            {
                printf(" - Features eliminadas: ");
                for(j = 0; j < level-1; j++)
                    printf("%d,", ofnames[j]);
                printf("%d", ofnames[j]);
            }
            printf("\nIns: %d / ", contheap);
            printf("ReIns: %d / ", contheapreins);
            printf("Max: %d / ", maxheapsize);
            printf("Podas: %d / ", contprooning);
            printf("Hash: %d / ", conthash);
            printf("Expandidos: %d / ", contexpandidos);
            printf("Nao Treinados: %d", contprojetados-contprojtreinados);
            printf("\n--- --- --- --- --- --- --- ---\n");
        }

        /*salvando os dados da ultima dimensao fechada*/
        margem_parcial = margin;
        svs_parcial = svcount;
        tempo_parcial = ((100.0f*clock()/CLOCKS_PER_SEC)-tempo_inicial)/100.0f;
        dim_parcial = startdim-level;
        contheap_parcial = contheap;
        contheapreins_parcial = contheapreins;
        conthash_parcial = conthash;
        contprooning_parcial = contprooning;
        maxheapsize_parcial = maxheapsize;
        contnaoheap_parcial = contnaoheap;
        conthashnaoheap_parcial = conthashnaoheap;
        contexpandidos_parcial = contexpandidos;
        contprojetados_parcial = contprojetados;
        contprojtreinados_parcial = contprojtreinados;
        //sobraprojecoes_parcial = aos_select_heap_projected(heap);
        data_free_sample(&stmp_parcial);
        stmp_parcial = data_copy_sample(sample);

        if(look_ahead_depth > 0)
        {
            if(sample->kernel_type == 9)
            {
                /*criar um w de manutencao pra volta do look_ahead*/
                w_manut = (double*) malloc(dim*sizeof(double));
                if(w_manut == NULL) { printf("Error: Out of memory 1\n"); exit(1); }
                memcpy(w_manut, w, dim*sizeof(double));//for(i = 0; i < dim; i++) w_manut[i] = w[i];
            }

            /*get new look ahead margin*/
            (*g_margin) = aos_select_look_ahead(sample, heap, hash, ofnames, w, train, depth, level, look_ahead_depth, bonus, forma_escolha, verbose);

            if(sample->kernel_type == 9)
            {
                w = NULL;
                //free(w);
                w = (double*) malloc(dim*sizeof(double));
                if(w == NULL) { printf("Error: Out of memory 2\n"); exit(1); }
                memcpy(w, w_manut, dim*sizeof(double));//for(i = 0; i < dim; i++) w[i] = w_manut[i];
                free(w_manut);
            }
        }
        /*cut heap based on its level and look ahead margin*/
        //printf("Look Ahead = %d\n", look_ahead_depth);
        //if(look_ahead_depth > 0 || cut < startdim-level)
        aos_select_heap_cut(heap, hash, *lool, cut, *g_margin, verbose);

	    /*check for startover level*/
	    /*if((level != 0) && (level % startover == 0))
	    {
            //cleaning up heap and hash
            aos_select_hash_free(hashp);
            aos_select_heap_free(heapp);

            *hashp = aos_select_hash_create(HASH_SIZE, HASH_WIDTH);
            *heapp = aos_select_heap_create();
            hash = *hashp;
            heap = *heapp;
	    }*/

        if(verbose > 2)
        {
            printf(" (%d)\t", heap->size);
            for(j = 0; j < level ; j++)
                printf("%d ", ofnames[j]);
            printf("\n");
        }
    }

    /*allocating space for w/feature*/
    weight = (aos_select_weight*) malloc(dim*sizeof(aos_select_weight));
    if(weight == NULL) { printf("Error: Out of memory 3\n"); exit(1); }

    /*copying elements of array*/
    for(i = 0; i < dim; i++)
    {
        weight[i].w      = w[i];
        weight[i].indice = i;
        weight[i].fname  = sample->fnames[i];
        weight[i].raio   = -1;
        weight[i].dcents = -1;
        weight[i].golub  = -1;
        weight[i].fisher = -1;
    }

    if(forma_ordenacao == 2)
        for(i = 0; i < dim; i++)
            weight[i].dcents = data_get_dist_centers(sample, weight[i].fname);
    else if(forma_ordenacao == 3)
        for(i = 0; i < dim; i++)
        {
            weight[i].raio   = data_get_radius(sample, weight[i].fname, q);
            weight[i].dcents = data_get_dist_centers(sample, weight[i].fname);
        }
    else if(forma_ordenacao == 4)
        for(i = 0; i < dim; i++)
            weight[i].raio   = data_get_radius(sample, weight[i].fname, q);
    else if(forma_ordenacao == 5 || forma_ordenacao == 6)
    {
        int num_pos = 0, num_neg = 0;
        /*alloc memory*/
        double *avg_pos = (double*) malloc(dim*sizeof(double));
        if(avg_pos == NULL) { printf("Out of mem!! AVG_POS\n"); exit(1); }
        double *avg_neg = (double*) malloc(dim*sizeof(double));
        if(avg_neg == NULL) { printf("Out of mem!! AVG_NEG\n"); exit(1); }
        double *sd_pos =  (double*) malloc(dim*sizeof(double));
        if(sd_pos  == NULL) { printf("Out of mem!! SD_POS\n"); exit(1); }
        double *sd_neg =  (double*) malloc(dim*sizeof(double));
        if(sd_neg  == NULL) { printf("Out of mem!! SD_NEG\n"); exit(1); }

        /*calc average*/
        for(i = 0; i < dim; ++i)
        {
            num_neg    = 0; num_pos    = 0;
            avg_neg[i] = 0; avg_pos[i] = 0;
            for(j = 0; j < size; ++j)
                if(sample->points[j].y == -1)
                { avg_neg[i] += sample->points[j].x[i]; num_neg++; }
                else
                { avg_pos[i] += sample->points[j].x[i]; num_pos++; }
            avg_neg[i] /= num_neg;
            avg_pos[i] /= num_pos;
        }

        /*calc variance*/
        for(i = 0; i < dim; ++i)
        {
            sd_neg[i] = 0; sd_pos[i] = 0;
            for(j = 0; j < size; ++j)
                if(sample->points[j].y == -1)
                    sd_neg[i] += pow(sample->points[j].x[i]-avg_neg[i], 2);
                else
                    sd_pos[i] += pow(sample->points[j].x[i]-avg_pos[i], 2);
        }

        for(i = 0; i < dim; i++)
        {
            weight[i].golub  = fabs(avg_pos[i]-avg_neg[i])/(sqrt(sd_pos[i]/(num_pos-1))+sqrt(sd_neg[i]/(num_neg-1)));
            weight[i].fisher = pow(avg_pos[i]-avg_neg[i], 2)/(sd_pos[i]+sd_neg[i]);
        }
        free(avg_pos);
        free(avg_neg);
        free(sd_pos);
        free(sd_neg);
    }

    /*forma de ordenacao*/
    if(forma_ordenacao == 6) // w * fisher
        qsort(weight, dim, sizeof(aos_select_weight), aos_select_compare_weightfisher_greater);
    else if(forma_ordenacao == 5) // w * golub
        qsort(weight, dim, sizeof(aos_select_weight), aos_select_compare_weightgolub_greater);
    else if(forma_ordenacao == 4) // w * raio
        qsort(weight, dim, sizeof(aos_select_weight), aos_select_compare_weightradius_greater);
    else if(forma_ordenacao == 3) // w * raio / distcenter
        qsort(weight, dim, sizeof(aos_select_weight), aos_select_compare_weightradiuscenter_greater);
    else if(forma_ordenacao == 2) // w / distcenter
        qsort(weight, dim, sizeof(aos_select_weight), aos_select_compare_weightcenter_greater);
    else // w
        qsort(weight, dim, sizeof(aos_select_weight), aos_select_compare_weight_greater);

    /*getting norm*/
    if(sample->kernel_type == 9)
        if(q == 1)
        {
            for(wnorm = 0, i = 0; i < dim; ++i)
                wnorm += fabs(w[i]);
        }
        else
            wnorm = utils_norm(w, dim, q);
    else
    {
        double **matrix = kernel_generate_matrix(sample);
        wnorm = kernel_feature_space_norm(sample, matrix);
        kernel_free_matrix(matrix, size);
    }

    /*ramificando os nodos*/
    for(i = 0; i < breadth; ++i)
    {
        if(sample->kernel_type == 9)
        {   /*calculo da margem projetada no IMA Primal*/
            if(dim > 1)
            {   // margem projetada calculada pela norma q
                if(q == 1)
                {
                    for(sumnorm = 0, j = 0; j < dim; ++j)
                        if(i != j)
                            sumnorm += fabs(weight[j].w);
                }
                else
                {
                    for(sumnorm = 0, j = 0; j < dim; ++j)
                        if(i != j)
                            sumnorm += pow(fabs(weight[j].w), q);
                    sumnorm = pow(sumnorm, 1.0/q);
                }
                tpmargin = sumnorm/wnorm*margin;
            }
            else
            {
                sumnorm = pow(fabs(weight[0].w), q);
                sumnorm = pow(sumnorm, 1.0/q);
                tpmargin = sumnorm/wnorm*margin;
            }
        }
        else
        {   /*calculo da margem projetada no IMA Dual e SMO*/
            double **Hk = kernel_generate_matrix_H_without_dim(sample, weight[i].indice);
            double *alphaaux = (double*) malloc(size*sizeof(double));
            if(alphaaux == NULL) { printf("Error: Out of memory 4\n"); exit(1); }

            for(k = 0; k < size; ++k)
                for(alphaaux[k] = 0, j = 0; j < size; ++j)
                    alphaaux[k] += sample->points[j].alpha * Hk[k][j];

            for(sumnorm = 0, k = 0; k < size; ++k)
                sumnorm += alphaaux[k] * sample->points[k].alpha;
            sumnorm = sqrt(sumnorm);
            tpmargin = sumnorm/wnorm*margin;

            free(alphaaux);
            kernel_free_matrix(Hk, size);
        }

        /*creating new nodes for heap*/
        gtmp = (aos_select_gamma*) malloc(sizeof(aos_select_gamma));
        if(gtmp == NULL) { printf("Error: Out of memory 5\n"); exit(1); }

        /*setting values*/
        if(forma_escolha == 2)
        {
            if(weight[i].w == 0)
                gtmp->value  = margin*weight[i].dcents;
            else
                gtmp->value  = tpmargin*weight[i].dcents;
        }
        else
        {
            if(weight[i].w == 0)
                gtmp->value  = margin;
            else
                gtmp->value  = tpmargin;
        }
        gtmp->pgamma = tpmargin;
        if(weight[i].w == 0)
        {
            gtmp->rgamma = margin;
            gtmp->train  = 1;
        }
        else
        {
            gtmp->rgamma = -1;
            gtmp->train  = 0;
        }
        gtmp->w      = NULL;
        gtmp->bias   = 0;
        gtmp->sv     = 0;
        gtmp->level  = level+1;
        gtmp->raio   = weight[i].raio;
        gtmp->dcents = weight[i].dcents;
        gtmp->golub  = weight[i].golub;
        gtmp->fisher = weight[i].fisher;

        /*manutencao do W do pai*/
        if(sample->kernel_type == 9)
        {
            gtmp->w = (double*) malloc(dim*sizeof(double));
            if(gtmp->w == NULL) { printf("Error: Out of memory 6\n"); exit(1); }

            for(k = 0, j = 0; j < dim; j++)
                if(sample->fnames[j] != weight[i].fname)
                    gtmp->w[k++] = w[j];
            //gtmp->w[k] = w[j]; //bias nao copia mais
            gtmp->bias = sample->bias;
        }

        /*creating new fnames array*/
        gtmp->fnames = (int*) malloc((level+1)*sizeof(int));
        if(gtmp->fnames == NULL) { printf("Error: Out of memory 7\n"); exit(1); }

        for(j = 0; j < level; j++)
            gtmp->fnames[j] = ofnames[j];
        gtmp->fnames[j] = weight[i].fname;

        /*sorting new feature array*/
        qsort(gtmp->fnames, level+1, sizeof(int), aos_select_compare_int_greater);

        if(forma_escolha == 2)
        {
            if(i != 0 && tpmargin*weight[i].dcents < (1-NUM_ERROR_EPS)*(*g_margin))
            {
                aos_select_hash_add(hash, gtmp);
                contnaoheap++; continue;
            }
        }
        else
        {
            if(i != 0 && tpmargin < (1-NUM_ERROR_EPS)*(*g_margin))
            {
                aos_select_hash_add(hash, gtmp);
                contnaoheap++; continue;
            }
        }

        /*some verbose*/
        if(verbose)
        {
            if(forma_ordenacao == 5 || forma_ordenacao == 6)
                printf("  -- Novo nodo - Feature: %d, Value: %lf, pMargem: %lf, DCent: %lf, Raio: %lf, Golub: %lf, Fisher: %lf, Nivel: %d\n", weight[i].fname, gtmp->value, gtmp->pgamma, gtmp->dcents, gtmp->raio, gtmp->golub, gtmp->fisher, gtmp->level);
            else
                //printf("  -- Novo nodo - Feature: %d, Value: %lf, pMargem: %lf, DCent: %lf, Raio: %lf, Nivel: %d\n", weight[i].fname, gtmp->value, gtmp->pgamma, gtmp->dcents, gtmp->raio, gtmp->level);
                printf("  -- Novo nodo - Feature: %d, Value: %lf, rMargem: %lf, pMargem: %lf, DCent: %lf, Raio: %lf, Nivel: %d\n", weight[i].fname, gtmp->value, gtmp->rgamma, gtmp->pgamma, gtmp->dcents, gtmp->raio, gtmp->level);
        }

        /*push node into heap if it is not redundant in hash*/
        if(aos_select_hash_add(hash, gtmp))
        {
            aos_select_heap_insert(heap, gtmp, 1);
            contprojetados++;
        }
        else
            conthashnaoheap++;
    }

    /*freeing stuff*/
    //free(w);
    free(weight);
    free(ofnames);
}

/*----------------------------------------------------------*
 * look ahead for pruning value                             *
 *----------------------------------------------------------*/
double
aos_select_look_ahead(sample *sample, aos_select_heap *heap,
        aos_select_hash *hash, int *fnames_orig, double *w_orig,
        int (*train)(struct sample*,double**,double*,int*,int),
        int depth, int level_orig, int look_ahead_depth,
        double bonus, int forma_escolha, int verbose)
{
    register int i = 0, j = 0;
    int level = level_orig;
    int svcount = 0;
    int count = 0;
    int feat = 0;
    int *features = NULL;
    double min = 0;
    double margin = 0;
    double g_margin = 0;
    double *w = w_orig;
    double *novo_w = NULL;
    struct sample *stmp = sample;
    aos_select_gamma *gtmp = NULL;
    double distcents = 0;

    /*creating an array to place selected features*/
    features = (int*) malloc((look_ahead_depth+1)*sizeof(int));
    if(features == NULL) { printf("Out of mem!! Feats Look-ahead\n"); exit(1); }

    while(1)
    {
        /*stopping criterion*/
        if(count == look_ahead_depth || count == sample->dim-1 || stmp->dim == sample->dim-depth || level == depth)
            break;

        if(forma_escolha == 2)
        {   /*selecting one feature with least w / dist. centers*/
            min  = fabs(w[0])/data_get_dist_centers(stmp, stmp->fnames[0]);
            feat = stmp->fnames[0];
            for(i = 1; i < stmp->dim; i++)
            {
                distcents = data_get_dist_centers(stmp, stmp->fnames[i]);
                if(fabs(w[i])/distcents < min)
                {
                    min = fabs(w[i])/distcents;
                    feat = stmp->fnames[i];
                }
            }
        }
        else
        {   /*selecting one feature with least w*/
            min  = fabs(w[0]);
            feat = stmp->fnames[0];
            for(i = 1; i < stmp->dim; i++)
                if(fabs(w[i]) < min)
                {
                    min = fabs(w[i]);
                    feat = stmp->fnames[i];
                }
        }

        /*manutencao do w do pai para o IMA Primal*/
        if(stmp->kernel_type == 9)
        {
            novo_w = (double*) malloc((stmp->dim-1)*sizeof(double));
            if(novo_w == NULL) { printf("Error: Out of memory 8\n"); exit(1); }
            for(i = 0, j = 0; j < stmp->dim; ++j)
                if(stmp->fnames[j] != feat)
                    novo_w[i++] = w[j];
        }

        /*saving removed feature name*/
        features[count] = feat;

        /*removing old data sample*/
        if(stmp != sample) data_free_sample(&stmp);

        /*get temp data struct*/
        stmp = data_remove_features(sample, features, (count+1), 0);

        if(level == 0)
            n0 = stmp->max_time = max_time_orig * primeiro_decaimento;
        else
            stmp->max_time = n0 * exp(-stmp->mult_tempo * ((double)dim_orig/(dim_orig-level-1)));

        /*creating new nodes for heap*/
        gtmp = (aos_select_gamma*) malloc(sizeof(aos_select_gamma));
        if(gtmp == NULL) { printf("Error: Out of memory 9\n"); exit(1); }

        /*setting values*/
        gtmp->value  = -1;
        gtmp->pgamma = stmp->margin;
        gtmp->rgamma = -1;
        gtmp->train  =  0;
        gtmp->sv     =  0;
        gtmp->w      = NULL;
        gtmp->bias   = 0;
        gtmp->level  = level+1;
        gtmp->raio   = -1; //data_get_radius(stmp, -1, stmp->q);
        if(forma_escolha == 2)
            gtmp->dcents = data_get_dist_centers(stmp, -1);
        else
            gtmp->dcents = -1;
        gtmp->fisher = -1;
        gtmp->golub  = -1;

        /*creating new fnames array*/
        gtmp->fnames = (int*) malloc((level+1)*sizeof(int));
        if(gtmp->fnames == NULL) { printf("Error: Out of memory 10\n"); exit(1); }

        for(j = 0; j < level_orig; ++j) gtmp->fnames[j] = fnames_orig[j];
        for(j = 0; j < count+1   ; ++j) gtmp->fnames[level_orig+j] = features[j];

        /*sorting new feature array*/
        qsort(gtmp->fnames, level+1, sizeof(int), aos_select_compare_int_greater);

        if(verbose)
        {
            printf("  -- Novo look-ahead nodo - Features: ");
            for(i = 0; i < count; i++)
                printf("%d, ", features[i]);
            printf("%d\n", features[i]);
        }

        /*push node into heap if it is not redundant in hash*/
        if(aos_select_hash_add(hash, gtmp))
        {
            /*training sample*/
            svcount = 0;
            margin = gtmp->pgamma;
            stmp->bias = gtmp->bias;
            if(!train(stmp, &novo_w, &margin, &svcount, 0))
            {
                free(novo_w);
                if(verbose) printf("Treinamento falhou!\n");
                break;
            }
            gtmp->w      = novo_w;
            gtmp->bias   = stmp->bias;
            gtmp->sv     = svcount;
            gtmp->rgamma = margin;
            gtmp->train  = 1;
            if(forma_escolha == 2)
                gtmp->value = margin * gtmp->dcents;
            else
                gtmp->value = margin;
            g_margin = gtmp->value;
            aos_select_heap_insert(heap, gtmp, 1);
        }
        if(stmp->kernel_type == 9)
        {
            w = (double*) malloc((stmp->dim)*sizeof(double));
            if(w == NULL) { printf("Error: Out of memory 11\n"); exit(1); }
            memcpy(w, novo_w, (stmp->dim)*sizeof(double));
        }

        /*increment*/
        level++;
        count++;
    }

    w_orig = w;

    /*free stuff*/
    free(features);
    if(stmp != sample) data_free_sample(&stmp);

    /*return gamma*/
    return g_margin;
}

/*-----------------------------------------------------------*
 * Run A* feature selection                                  *
 *                                                           *
 * breadth = maximum search breadth.                         *
 * depth   = stop when depth is reached                      *
 * bonus   = bonus value in formula margin*(1+bonus(level+1))*
 * look_ahead_depth = depth of puring value search           *
 *-----------------------------------------------------------*/
sample*
aos_select_features(char *filename, sample *sample,
        int (*train)(struct sample*,double**,double*,int*,int),
        int breadth, int depth, double bonus, int cut, int look_ahead_depth,
        int skip, int startover, int doleave_oo, int forma_ordenacao, int forma_escolha,
        crossvalidation *cv, int verbose)
{
    register int i = 0;
    int tbreadth = 0;
    int level    = 0;
    int dim      = sample->dim;
    int startdim = dim_orig = dim;
    int *fnames  = NULL;
    int ftime    = 1;
    int lool     = 0;    /*leave one out level*/
    double g_margin = 0;
    struct sample* stmp   = NULL;
    aos_select_heap* heap = NULL;
    aos_select_hash* hash = NULL;
    max_time_orig = sample->max_time;

    int parcial = 0; //verifica se última solução é uma solução recuperada (parcial)

    /*inicializacao dos contadores*/
    contheap         = 0;
    contheapreins    = 0;
    conthash         = 0;
    contprooning     = 0;
    maxheapsize      = 0;
    contnaoheap      = 0;
    conthashnaoheap  = 0;
    contexpandidos   = 0;
    contprojetados   = 0;
    contprojtreinados= 0;

    START_TIME = 100.0f*clock()/CLOCKS_PER_SEC;
    tempo_inicial = START_TIME;

    /*checking arguments*/
    if(breadth > MAX_BREATH) { breadth = MAX_BREATH; }
    if(depth   > dim - 1   ) { depth   = dim - 1;    }
    if(depth   > MAX_DEPTH ) { depth   = MAX_DEPTH;  }

    /*create a hash*/
    hash = aos_select_hash_create(HASH_SIZE, HASH_WIDTH);
    heap = aos_select_heap_create();

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

    /*do this while my depth permits*/
    while(1)
    {
        /*first problem to solve, when heap is empty*/
        if(heap->size == 0)
        {
            /*quit end of heap found*/
            if(ftime == 0) /*first time = false -- nao eh a primeira vez, o processo falhou!*/
            {
                printf("Final do Heap, recuperando ultima dimensao...\n\n");

                /*pegar os dados do ultimo lool, uma vez que foi a ultima dimensao fechada*/
                printf("---------------\n :: FINAL :: \n---------------\n");
                printf("Features Escolhidas: ");
                for(i = 0; i < stmp_parcial->dim-1; ++i)
                    printf("%3d,", stmp_parcial->fnames[i]);
                printf("%3d\n", stmp_parcial->fnames[i]);

                if(cv->qtde > 0)
                {
                    if(level % cv->jump != 0)
                    {
                        for(cv->erro_atual = 0, i = 0; i < cv->qtde; i++)
                            cv->erro_atual += utils_k_fold(stmp_parcial, train, cv->fold, cv->seed[i], 0);
                        errokfold = cv->erro_atual / cv->qtde;
                    }
                    printf("Dim: %d, Margem: %lf, SVs: %d, Erro %d-fold: %.2lf%%\n", dim_parcial, margem_parcial, svs_parcial, cv->fold, errokfold);
                }
                else
                    printf("Dim: %d, Margem: %lf, SVs: %d\n", dim_parcial, margem_parcial, svs_parcial);
                printf("Total de insercoes no Heap: %d\n", contheap_parcial);
                printf("Total de reinsercoes no Heap: %d\n", contheapreins_parcial);
                printf("Tamanho maximo do Heap: %d\n", maxheapsize_parcial);
                printf("Total de podas no Heap: %d\n", contprooning_parcial);
                printf("Nodos expandidos: %d\n", contexpandidos_parcial);
                printf("Nao inseridos no Heap: %d\n", contnaoheap_parcial);
                printf("Total de projecoes: %d\n", contprojetados_parcial);
                printf("Total de projecoes treinadas: %d\n", contprojtreinados_parcial);
                printf("Total de projecoes nao treinadas: %d\n", contprojetados_parcial-contprojtreinados_parcial);
                //printf("Sobra de projecoes no Heap: %d\n", sobraprojecoes_parcial);
                printf("Nodos iguais no Hash que nao entraram no Heap: %d\n", conthashnaoheap_parcial);
                printf("Tamanho do Hash: %d\n", conthash_parcial);
                printf("Tempo total: %.3lfs\n", tempo_parcial);
                parcial = 1;
                /*Save data to file*/
                data_write(filename, stmp_parcial, 0);
                break;
            }

            /*check breadth*/
            tbreadth = breadth;
            if(tbreadth > dim) tbreadth = dim;

            /*run select*/
            aos_select_feature_main_loop(sample, &heap, &hash, train,
                    tbreadth, bonus, &lool, startdim, look_ahead_depth,
                    &g_margin, cut, skip, startover, doleave_oo, depth,
                    forma_ordenacao, forma_escolha, ftime, cv, verbose);
            if(heap->size == 0)
            {
                printf("Treinamento inicial falhou!!!\n\n");
                break;
            }
            ftime = 0;
        }
        /*subsequent problems (heap not empty)*/
        else
        {
            /*create new data struct*/
            level  = heap->elements[1]->level;
            fnames = heap->elements[1]->fnames;
            stmp   = data_remove_features(sample, fnames, level, 0);

            if(level == 1)
                n0 = stmp->max_time *= primeiro_decaimento;
            else if(level > 1)
                stmp->max_time = n0 * exp(-stmp->mult_tempo * ((double)startdim/(startdim-level)));

            /*stop criterium*/
            if(stmp->dim == dim-depth && heap->elements[1]->rgamma > 0)
            {
                printf("---------------\n :: FINAL :: \n---------------\n");
                printf("Features Escolhidas: ");
                for(i = 0; i < stmp->dim-1; ++i)
                    printf("%3d,", stmp->fnames[i]);
                printf("%3d\n", stmp->fnames[i]);

                printf("---------------\nFeatures Eliminadas: ");
                for(i = 0; i < sample->dim-stmp->dim-1; ++i)
                    printf("%3d,", heap->elements[1]->fnames[i]);
                printf("%3d\n", heap->elements[1]->fnames[i]);

                if(cv->qtde > 0)
                {
                    for(cv->erro_atual = 0, i = 0; i < cv->qtde; i++)
                        cv->erro_atual += utils_k_fold(stmp, train, cv->fold, cv->seed[i], 0);
                    errokfold = cv->erro_atual / cv->qtde;
                    printf("Dim: %d, Margem: %lf, SVs: %d, Erro %d-fold: %.2lf%%\n", stmp->dim, heap->elements[1]->rgamma, heap->elements[1]->sv, cv->fold, errokfold);
                }
                else
                    printf("Dim: %d, Margem: %lf, SVs: %d\n", stmp->dim, heap->elements[1]->rgamma, heap->elements[1]->sv);
                printf("Total de insercoes no Heap: %d\n", contheap);
                printf("Total de reinsercoes no Heap: %d\n", contheapreins);
                printf("Tamanho maximo do Heap: %d\n", maxheapsize);
                printf("Total de podas no Heap: %d\n", contprooning);
                printf("Nodos expandidos: %d\n", contexpandidos);
                printf("Nao inseridos no Heap: %d\n", contnaoheap);
                printf("Total de projecoes: %d\n", contprojetados);
                printf("Total de projecoes treinadas: %d\n", contprojtreinados);
                printf("Total de projecoes nao treinadas: %d\n", contprojetados-contprojtreinados);
                //printf("Sobra de projecoes no Heap: %d\n", aos_select_heap_projected(heap));
                printf("Nodos iguais no Hash que nao entraram no Heap: %d\n", conthashnaoheap);
                printf("Tamanho do Hash: %d\n", conthash);
                printf("Tempo total: %.3lfs\n", ((100.0f*clock()/CLOCKS_PER_SEC)-tempo_inicial)/100.0f);

                /*Save data to file*/
                data_write(filename, stmp, 0);
                break;
            }

            /*some verbose*/
            if(verbose && level <= 50)
            {
                printf("-- Testando o nodo - Features (");
                for(i = 0; i < level-1; ++i)
                    printf("%d,", fnames[i]);
                printf("%d)\n", fnames[i]);
                printf("------------------------------------------\n");
            }

            /*check breadth*/
            tbreadth = breadth;
            if(tbreadth > stmp->dim) tbreadth = stmp->dim;

            /*run select*/
            aos_select_feature_main_loop(stmp, &heap, &hash, train,
                    tbreadth, bonus, &lool, startdim, look_ahead_depth,
                    &g_margin, cut, skip, startover, doleave_oo, depth,
                    forma_ordenacao, forma_escolha, ftime, cv, verbose);

            /*free stuff*/
            data_free_sample(&stmp);
        }

        /*verbose*/
        if(verbose)
        {
            if(verbose > 1)
            {
                aos_select_heap_print(heap);
                fflush(stdout);
            }
            else
                printf("-- Heap Size: %d\n", heap->size);
            printf("--------\n");
        }
    }

    //aos_select_hash_print(hash, sample->dim);

    /*free stuff*/
    aos_select_hash_free(&hash);
    aos_select_heap_free(&heap);
    data_free_sample(&sample);
    if(cv->qtde > 0) free(cv->seed);
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

/***********************************************************
 *              HASH FUNCTIONS                             *
 ***********************************************************/

/*----------------------------------------------------------*
 * erase an element from my hash                            *
 *----------------------------------------------------------*/
void
aos_select_hash_set_null(aos_select_hash *hash, aos_select_gamma *elmt)
{
    register int i = 0, j = 0;
    int index = 0;
    register double func = 0;

    /*error check*/
    if(elmt == NULL || hash == NULL) return;

    /*hashing function*/
    for(i = 0; i < elmt->level; ++i)
        func += pow(elmt->fnames[i], 2);

    index = fmod(func, hash->length);

    /*finding it*/
    i = 0;
    while(i < hash->width && hash->elements[index][i] != NULL)
    {
        /*check equality between nodes*/
        if(aos_select_node_equal(elmt, hash->elements[index][i]))
        {
            /*shift elements*/
            j = i+1;
            while(j < hash->width && hash->elements[index][j] != NULL)
            {
                hash->elements[index][j-1] = hash->elements[index][j];
                j++;
            }
            /*setting last element as null*/
            hash->elements[index][j-1] = NULL;

            break;
        }
        i++;
    }
}

/*----------------------------------------------------------*
 * insert an element into my hash                           *
 *----------------------------------------------------------*/
int
aos_select_hash_add(aos_select_hash *hash, aos_select_gamma *elmt)
{
    register int i = 0;
    int index = 0;
    register double func = 0;

    /*error check*/
    if(elmt == NULL || hash == NULL)
        return 0;

    /*hasing function*/
    func = 0;
    for(i = 0; i < elmt->level; ++i)
        func += pow(elmt->fnames[i], 2);

    index = fmod(func, hash->length);

    /*skiping equals*/
    i = 0;
    while(i < hash->width && hash->elements[index][i] != NULL)
    {
        /*check equality between nodes*/
        if(aos_select_node_equal(elmt, hash->elements[index][i]))
        {
            /*this node is identical to some other node*/
            /*check if this node has real gamma*/
            if(hash->elements[index][i]->rgamma < 0)
            {
                /*keep node with highest projected value*/
                if(hash->elements[index][i]->value < elmt->value)
                {
                    hash->elements[index][i]->pgamma = elmt->pgamma;
                    hash->elements[index][i]->value  = elmt->value;
                }
            }

            /*printf("IDENTICAL! (%d) ",index);
            printf("["); int j;
            for(j = 0; j< hash->elements[index][i]->level; ++j)
                printf("%d,",hash->elements[index][i]->fnames[j]);
            printf("] = ");

            printf("[");
            for(j = 0; j< elmt->level; ++j)
                printf("%d,",elmt->fnames[j]);
            printf("]\n");
            */

            /*free stuff*/
            free(elmt->fnames);
            free(elmt->w);
            free(elmt);
            return 0;
        }
        else
        {
            /*
            printf("CRASH! (%d) ",index);
            printf("[");
            for(j = 0; j< hash->elements[index][i]->level; ++j)
                printf("%d,",hash->elements[index][i]->fnames[j]);
            printf("]\n");
            */
        }
        /*increment*/
        i++;
    }

    /*adding element*/
    if(i >= hash->width)
    {
        int filled = 0;
        for(i = 0; i < hash->length; ++i)
            if(hash->elements[i][0] == NULL) filled++;

        printf("NEED RE-HASH! (just failed) (%d/%d) = %lf%c\n", filled, hash->length, ((double)filled)/(hash->length)*100.0, '%');
        exit(1);
    }
    else
    {
        hash->elements[index][i] = elmt;
        conthash++;
    }
    return 1;
}

/*----------------------------------------------------------*
 * creates a fresh new hash                                 *
q *----------------------------------------------------------*/
aos_select_hash*
aos_select_hash_create(int length, int width)
{
    register int i = 0, j = 0;
    aos_select_hash *hash = NULL;

    /*alloc space for head node*/
    hash = (aos_select_hash*) malloc(sizeof(aos_select_hash));
    if(hash == NULL) return NULL;
    hash->length = length;
    hash->width  = width;

    /*allocating space for new hash*/
    hash->elements = (aos_select_gamma***) malloc(length*sizeof(aos_select_gamma**));

    /*error check*/
    if(hash->elements == NULL)
    {
        free(hash);
        return NULL;
    }

    /*allocating space for crashes*/
    for(i = 0; i < length; ++i)
    {
        /*array for crashes*/
        hash->elements[i] = (aos_select_gamma**) malloc(width*sizeof(aos_select_gamma*));

        /*error check*/
        if(hash == NULL)
        {
            for(j = 0; j < i; ++j) free(hash->elements[j]);
            free(hash);
            return NULL;
        }

        /*set elements as NULLS*/
        for(j = 0; j < width; ++j)
            hash->elements[i][j] = NULL;
    }
    return hash;
}

/*---------------------------------q-------------------------*
 * frees hash table                                         *
 *----------------------------------------------------------*/
void
aos_select_hash_free(aos_select_hash **hash)
{
    register int i = 0;//, j = 0;
    for(i = 0; i < (*hash)->length; ++i)
    {
        //for(j = 0; j < (*hash)->width; ++j)
        //{
        //    free((*hash)->elements[i][j]->w);
        //    free((*hash)->elements[i][j]->fnames);
        //}
        free((*hash)->elements[i]);
    }

    free((*hash)->elements);
    free((*hash));
    (*hash) = NULL;
}

/*---------------------------------q-------------------------*
 * print hash table                                         *
 *----------------------------------------------------------*/
void
aos_select_hash_print(aos_select_hash *hash, int dim)
{

    register int i = 0, j = 0, k = 0, d = 0;
    int cont = 0;
    //for(d = 0; d < dim; d++)
    //{
        for(i = 0; i < hash->length; ++i)
            for(j = 0; j < hash->width; ++j)
                if(hash->elements[i][j] != NULL)
                    //if(hash->elements[i][j]->level == d)
                    {
                        cont++;
                        //printf("Hash[%d][%d] --", i, j);
                        //printf(" (");
                        //for(k = 0; k < (hash->elements[i][j]->level)-1; ++k)
                        //    printf("%3d,", hash->elements[i][j]->fnames[k]);
                        //printf("%3d\n", hash->elements[i][j]->fnames[k]);
                    }
        //printf("\n");
    //}
    printf("Cont = %d\n", cont);
    if(dim <= 10)
        for(d = 0; d < dim; d++)
        {
            for(i = 0; i < hash->length; ++i)
                for(j = 0; j < hash->width; ++j)
                    if(hash->elements[i][j] != NULL)
                        if(hash->elements[i][j]->level == d)
                        {
                            for(k = 0; k < (hash->elements[i][j]->level)-1; ++k)
                                printf("%3d,", hash->elements[i][j]->fnames[k]);
                            printf("%3d\n", hash->elements[i][j]->fnames[k]);
                        }
            printf("\n");
        }
}

/***********************************************************
 *               HEAP FUNCTIONS                            *
 ***********************************************************/
/*----------------------------------------------------------*
 * creates a heap                                           *
 *----------------------------------------------------------*/
aos_select_heap*
aos_select_heap_create()
{
    register int i = 0;
    aos_select_heap *heap = NULL;

    heap = (aos_select_heap*) malloc(sizeof(aos_select_heap));
    if(heap == NULL)
    {
        printf("Out of mem! Heap 1\n");
        exit(1);
    }

    heap->elements = (aos_select_gamma**) malloc((MAX_HEAP+1)*sizeof(aos_select_gamma*));

    if(heap->elements==NULL)
    {
        printf("Out of mem! Heap 2\n");
        exit(1);
    }

    for(i = 0; i < MAX_HEAP+1; ++i) heap->elements[i] = NULL;
    heap->size = 0;
    return heap;
}

/*----------------------------------------------------------*
 * Insert into heap                                         *
 *----------------------------------------------------------*/
int
aos_select_heap_insert(aos_select_heap* heap, aos_select_gamma* tok, int cont)
{
    register int i = 0;
    double val = 0;

    if(heap->size == MAX_HEAP)
    {
        if(tok->value > heap->elements[MAX_HEAP]->value)
            i = MAX_HEAP;
        else return 0;
    }
    else i = ++(heap->size);

    val = (heap->elements[i/2] != NULL) ? heap->elements[i/2]->value : 0;

    while(i > 1 && val < tok->value)
    {
        heap->elements[i] = heap->elements[i/2];
        i /= 2;
        val = (heap->elements[i/2] != NULL) ? heap->elements[i/2]->value : 0;
    }
    heap->elements[i] = tok;

    if(heap->size > maxheapsize)
        maxheapsize = heap->size;

    if(cont) contheap++;
    else contheapreins++;
    return 1;
}

/*----------------------------------------------------------*
 * Returns top element                                      *
 *----------------------------------------------------------*/
aos_select_gamma*
aos_select_heap_pop(aos_select_heap* heap)
{
    aos_select_gamma *min_element = NULL;

    if(heap->size == 0)
    {
        printf("Tried to pop an empty heap!\n");
        return NULL;
    }

    min_element = heap->elements[1];

    heap->size--;
    aos_select_heap_percolate(heap, 1);

    return min_element;
}

/*----------------------------------------------------------*
 * Frees heap                                               *
 *----------------------------------------------------------*/
void
aos_select_heap_free(aos_select_heap** heap)
{
    register int i = 0;
    for(i = 1; i <= (*heap)->size; ++i)
    {
        if((*heap)->elements[i] != NULL)
        {
            free((*heap)->elements[i]->fnames);
            free((*heap)->elements[i]->w);
        }
        free((*heap)->elements[i]);
    }
    free((*heap)->elements);
    free(*heap);
    *heap = NULL;
}

/*----------------------------------------------------------*
 * Prints heap                                              *
 *----------------------------------------------------------*/
void
aos_select_heap_print(aos_select_heap *heap)
{
    register int i = 0, j = 0;
    int *fnames = NULL;
    aos_select_gamma *curr = NULL;

    if(heap == NULL) return;

    for(i = 1; i <= heap->size; ++i)
    {
        curr = heap->elements[i];
        fnames = curr->fnames;
        printf("Heap[%2d] --", i);
        if(curr->level <= 50)
        {
            printf(" (");
            for(j = 0; j < curr->level-1; ++j)
                printf("%d,", fnames[j]);
            printf("%d)", fnames[j]);
        }
        printf(" Value: %lf,", curr->value);
        printf(" rMargem: %lf,", curr->rgamma);
        printf(" pMargem: %lf,", curr->pgamma);
        //printf(" Raio: %lf,", curr->raio);
        //printf(" Golub: %lf,", curr->golub);
        //printf(" Fisher: %lf,", curr->fisher);
        printf(" Nivel: %d\n", curr->level);
    }
}

/*----------------------------------------------------------*
 * Cont projected margin nodes in heap                      *
 *----------------------------------------------------------*/
int
aos_select_heap_projected(aos_select_heap *heap)
{
    register int i = 0;
    int projected = 0;
    aos_select_gamma *curr = NULL;

    if(heap == NULL) return 0;

    for(i = 1; i <= heap->size; ++i)
    {
        curr = heap->elements[i];
        if(curr->rgamma < 0)
            projected++;
    }
    return projected;
}

/*----------------------------------------------------------*
 * removes old levels                                       *
 *----------------------------------------------------------*/
void
aos_select_heap_cut(aos_select_heap *heap, aos_select_hash *hash, int levelat, int cut, double g_margin, int verbose)
{
    register int i = 0;
    int count = 0;
    aos_select_gamma *curr = NULL;
    //printf("Oiiiii!\n");
    for(i = heap->size; i > 0; --i)
    {
        curr = heap->elements[i];
        if (curr->level < levelat)
            if(curr->value < (1-NUM_ERROR_EPS)*g_margin || levelat-curr->level >= cut)
            {
                /*errase it from hash*/
                //if(levelat-curr->level >= cut) aos_select_hash_set_null(hash, curr);

                free(curr->fnames);
                free(curr->w);
                free(curr);

                /*percolate heap*/
                heap->size--;
                aos_select_heap_percolate(heap, i);
                count++;
            }
    }
    if(verbose) printf("  [Nodos removidos com a poda: %d]\n", count);

    contprooning += count;
}

/*----------------------------------------------------------*
 * percolates                                               *
 *----------------------------------------------------------*/
void
aos_select_heap_percolate(aos_select_heap *heap, int i)
{
    int child = i*2;
    aos_select_gamma *last_element = NULL;
    last_element = heap->elements[heap->size+1];

    if(child > heap->size)
    {
        heap->elements[i] = last_element;
        return;
    }

    if(child != heap->size && heap->elements[child+1]->value > heap->elements[child]->value)
        child++;

    /*percolate one level*/
    if(last_element->value < heap->elements[child]->value)
    {
        heap->elements[i] = heap->elements[child];
        aos_select_heap_percolate(heap,child);
    }
    else heap->elements[i] = last_element;
}



/***********************************************************
 *              OTHER FUNCTIONS                            *
 ***********************************************************/

/*----------------------------------------------------------*
 * checks if two nodes are the same                         *
 *----------------------------------------------------------*/
int
aos_select_node_equal(aos_select_gamma *one, aos_select_gamma *two)
{
    register int i  = 0;
    int eq = 0;

    if(one->level != two->level)
        return 0;

    for(i = 0; i < one->level; ++i)
    {
        if(one->fnames[i] == two->fnames[i]) eq++;
        else break;
    }
    return (eq == one->level);
}

/*----------------------------------------------------------*
 * Returns 1 for a > b, -1 a < b, 0 if a = b                *
 *----------------------------------------------------------*/
int
aos_select_compare_int_greater(const void *a, const void *b)
{
    const int *ia = (const int*) a;
    const int *ib = (const int*) b;

    /*          V (greater)*/
    return (*ia > *ib) - (*ia < *ib);
}

int
aos_select_compare_weight_greater(const void *a, const void *b)
{
    const aos_select_weight *ia = (const aos_select_weight*) a;
    const aos_select_weight *ib = (const aos_select_weight*) b;

    /*                  V (greater)*/
    return (fabs(ia->w) > fabs(ib->w)) - (fabs(ia->w) < fabs(ib->w));
}

int
aos_select_compare_weightradius_greater(const void *a, const void *b)
{
    const aos_select_weight *ia = (const aos_select_weight*) a;
    const aos_select_weight *ib = (const aos_select_weight*) b;

    /*                  V (greater)*/
    return (fabs(ia->w*ia->raio) > fabs(ib->w*ib->raio)) - (fabs(ia->w*ia->raio) < fabs(ib->w*ib->raio));
}

int
aos_select_compare_weightcenter_greater(const void *a, const void *b)
{
    const aos_select_weight *ia = (const aos_select_weight*) a;
    const aos_select_weight *ib = (const aos_select_weight*) b;

    /*                  V (greater)*/
    return (fabs(ia->w/ia->dcents) > fabs(ib->w/ib->dcents)) - (fabs(ia->w/ia->dcents) < fabs(ib->w/ib->dcents));
}

int
aos_select_compare_weightradiuscenter_greater(const void *a, const void *b)
{
    const aos_select_weight *ia = (const aos_select_weight*) a;
    const aos_select_weight *ib = (const aos_select_weight*) b;

    /*                  V (greater)*/
    return (fabs(ia->w*ia->raio/ia->dcents) > fabs(ib->w*ib->raio/ib->dcents)) - (fabs(ia->w*ia->raio/ia->dcents) < fabs(ib->w*ib->raio/ib->dcents));
}

int
aos_select_compare_weightfisher_greater(const void *a, const void *b)
{
    const aos_select_weight *ia = (const aos_select_weight*) a;
    const aos_select_weight *ib = (const aos_select_weight*) b;

    /*                  V (greater)*/
    return (fabs(ia->w*ia->fisher) > fabs(ib->w*ib->fisher)) - (fabs(ia->w*ia->fisher) < fabs(ib->w*ib->fisher));
}

int
aos_select_compare_weightgolub_greater(const void *a, const void *b)
{
    const aos_select_weight *ia = (const aos_select_weight*) a;
    const aos_select_weight *ib = (const aos_select_weight*) b;

    /*                  V (greater)*/
    return (fabs(ia->w*ia->golub) > fabs(ib->w*ib->golub)) - (fabs(ia->w*ia->golub) < fabs(ib->w*ib->golub));
}
