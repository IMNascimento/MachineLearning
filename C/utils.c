/*****************************************************
 * Utils lib                                         *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * Saulo Moraes <saulomv@gmail.com>                  *
 * 2004 / 2011                                       *
 *****************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "data.h"
#include "kernel.h"
#include "utils.h"

extern double kernel_param;
extern int kernel_type;

/*----------------------------------------------------------*
 * Returns the norm of a given vector                       *
 *----------------------------------------------------------*/
double
utils_norm(double* v, int dim, double q)
{
    register double sumnorm = 0.0;
    register int i = 0;

    for(i = 0; i < dim; ++i)
        sumnorm += pow(fabs(v[i]), q);
    return pow(sumnorm, 1.0/q);
}

/*----------------------------------------------------------*
 * Initializes random seed and random array                 *
 *----------------------------------------------------------*/
void
utils_initialize_random()
{
    //struct timeval tv;
    //gettimeofday(&tv, NULL);
    //srand(tv.tv_usec);
    //srand(time(NULL));
    srand(0);
}

/*----------------------------------------------------------*
 * Initializes random seed and random array                 *
 *----------------------------------------------------------*/
int*
utils_initialize_random_list(int size)
{
    register int i = 0;
    int *list = NULL;

    utils_initialize_random();

    list = (int*) malloc(size*sizeof(int));
    if(list == NULL) { printf("Error: Out of memory\n"); exit(1); }

    for(i = 0; i < size; ++i) list[i] = i;

    return list;
}

/*----------------------------------------------------------*
 * Returns shuffled array                                   *
 *----------------------------------------------------------*/
void
utils_shuffle(int **list, int size)
{
    register int i = 0;
    int r = 0, t = 0;

    for(i = 0; i < size; ++i)
    {
        r = rand()%size;
        t = (*list)[i];
        (*list)[i] = (*list)[r];
        (*list)[r] = t;
    }
}

/*----------------------------------------------------------*
 * Created a double linked list of ints                     *
 *----------------------------------------------------------*/
int_dll*
utils_int_dll_create()
{
    int_dll *head = NULL;

    /*Creating head node*/
    head = (int_dll*) malloc(sizeof(int_dll));
    if(head == NULL) { printf("Error: Out of memory\n"); return NULL; }
    head->next  = NULL;
    head->index = -1;
    head->prev  = NULL;

    return head;
}

/*----------------------------------------------------------*
 * Removes an elemente from a double linked list            *
 *----------------------------------------------------------*/
int_dll*
utils_int_dll_remove(int_dll **node)
{
    int_dll* ret = NULL;

    if((*node) == NULL) return NULL;

    /*remove items from list*/
    ret = (*node)->prev;

    /*fix reference one*/
    if((*node)->prev != NULL)
        (*node)->prev->next = (*node)->next;

    /*fix reference two*/
    if((*node)->next != NULL)
        (*node)->next->prev = (*node)->prev;

    free(*node);
    (*node) = NULL;

    return ret;
}

/*----------------------------------------------------------*
 * Appends a new node after list                            *
 *----------------------------------------------------------*/
int_dll*
utils_int_dll_append(int_dll *list)
{
    int_dll *tmp = NULL;

    /*error check*/
    if(list == NULL)
    { printf("Error in utils int linked list\n"); return NULL; }

    /*save old next*/
    tmp = list->next;

    /*new node*/
    list->next = (int_dll*) malloc(sizeof(int_dll));
    if(list == NULL) { printf("Error: Out of memory\n"); return NULL; }

    /*reference fixing*/
    list->next->prev = list;
    list->next->next = tmp;
    if(tmp != NULL) tmp->prev = list->next;

    /*finishing up*/
    list = list->next;
    list->index = -1;

    /*returning*/
    return list;
}

/*----------------------------------------------------------*
 * Clears linked list                                       *
 *----------------------------------------------------------*/
void
utils_int_dll_free(int_dll **head)
{
    int_dll *list = NULL;
    int_dll *tmpl = NULL;

    list = *head;
    while(list != NULL)
    {
        tmpl = list;
        list = list->next;
        free(tmpl);
    }
    *head = NULL;
}

/*----------------------------------------------------------*
 * Returns minimum value                                    *
 *----------------------------------------------------------*/
double
utils_min(double one, double two)
{
    double min = one;
    if(two < min) min = two;
    return min;
}

/*----------------------------------------------------------*
 * Returns maximum value                                    *
 *----------------------------------------------------------*/
double
utils_max(double one, double two)
{
    double max = one;
    if(two > max) max = two;
    return max;
}

/*----------------------------------------------------------*
 * Returns weight on feature space with dot product kernel  *
 *----------------------------------------------------------*/
double*
utils_get_weight(sample *sample)
{
	register int i = 0, j = 0;
	double* w = NULL;

	w = (double*) malloc((sample->dim)*sizeof(double));
	if(w == NULL) { printf("Error: Out of memory\n"); exit(1); }

	for(j = 0; j < sample->dim; ++j)
	    for(w[j] = 0, i = 0; i < sample->size; ++i)
            w[j] += sample->points[i].alpha * sample->points[i].y * sample->points[i].x[j];

    //for(i = 0; i < sample->size; ++i) //bias
    //    w[j] += sample->points[i].alpha * sample->points[i].y;

	return w;
}

/*----------------------------------------------------------*
 * Returns weight on feature space with otheres kernels     *
 *----------------------------------------------------------*/
double*
utils_get_dualweight(sample *sample)
{
    register int i = 0, j = 0, k = 0;
    int size = sample->size, dim = sample->dim;
    double **H = NULL, **Hk = NULL, **matrixdif = NULL;

    H = kernel_generate_matrix_H(sample);

	double *w = (double*) malloc(dim*sizeof(double));
	if(w == NULL) { printf("Error: Out of memory\n"); exit(1); }

    matrixdif = (double**) malloc(size*sizeof(double*));
    if(matrixdif == NULL) { printf("Error: Out of memory\n"); exit(1); }
    for(i = 0; i < size; ++i)
    {
        matrixdif[i] = (double*) malloc(size*sizeof(double));
        if(matrixdif[i] == NULL) { printf("Error: Out of memory\n"); exit(1); }
    }

	double *alphaaux = (double*) malloc(size*sizeof(double));
	if(alphaaux == NULL) { printf("Error: Out of memory\n"); exit(1); }

	for(k = 0; k < dim; ++k)
	{
	    Hk = kernel_generate_matrix_H_without_dim(sample, k);

	    for(i = 0; i < size; ++i)
            for(j = 0; j < size; ++j)
                matrixdif[i][j] = H[i][j] - Hk[i][j];

        for(i = 0; i < size; ++i)
            for(alphaaux[i] = 0, j = 0; j < size; ++j)
                alphaaux[i] += sample->points[j].alpha * matrixdif[i][j];

        for(w[k] = 0, i = 0; i < size; ++i)
            w[k] += alphaaux[i] * sample->points[i].alpha;

        kernel_free_matrix(Hk, size);
	}
	kernel_free_matrix(H, size);
	kernel_free_matrix(matrixdif, size);
	free(alphaaux);
	return w;
}

/*----------------------------------------------------------*
 * Returns weight on feature space with dot product kernel  *
 *----------------------------------------------------------*/
double*
utils_get_dualweight_prodint(sample *sample)
{
    register int i = 0, j = 0, k = 0;
    int size = sample->size, dim = sample->dim;

	double* w = (double*) malloc(dim*sizeof(double));
	if(w == NULL) { printf("Error: Out of memory\n"); exit(1); }

	double* alphaaux = (double*) malloc(size*sizeof(double));
	if(alphaaux == NULL) { printf("Error: Out of memory\n"); exit(1); }

    double** H = (double**) malloc(size*sizeof(double*));
    if(H == NULL) { printf("Error: Out of memory\n"); exit(1); }
    for(i = 0; i < size; ++i)
    {
        H[i] = (double*) malloc(size*sizeof(double));
        if(H[i] == NULL) { printf("Error: Out of memory\n"); exit(1); }
    }

	for(k = 0; k < dim; ++k)
	{
        for(i = 0; i < size; ++i)
            for(j = 0; j < size; ++j)
                H[i][j] = sample->points[i].x[k] * sample->points[j].x[k] * sample->points[i].y * sample->points[j].y;

        for(i = 0; i < size; ++i)
            for(alphaaux[i] = 0, j = 0; j < size; ++j)
                alphaaux[i] += sample->points[j].alpha * H[i][j];

        for(w[k] = 0, i = 0; i < size; ++i)
            w[k] += alphaaux[i] * sample->points[i].alpha;
	}
	return w;
}

/*----------------------------------------------------------*
 * Calculates Leave One Out Error Estimate                  *
 *----------------------------------------------------------*/
double
utils_leave_one_out_matrix(sample* sample, int (*train)(struct sample*,double**,double*,int*,int), double **o_matrix, int skip, int verbose)
{
	register int i = 0, j = 0, k = 0;
	int error = 0;
	register double func = 0.0;
    struct sample *tsample = NULL;
    double *w = NULL;
    double **matrix = NULL;
    double margin = 0;
    int svcount = 0;
    //kernel_print_matrix(o_matrix,sample->size);

	/*start leave one out*/
    for(i = 0; i < sample->size; ++i)
	{
        if(skip == +1 && sample->points[i].alpha == 0) continue;

		/*Creating temporary data array*/
        tsample = data_remove_point(sample, i);
        matrix = kernel_remove_element(o_matrix, sample->size, i);
        //kernel_print_matrix(matrix,sample->size-1);

		/*training*/
        if(!train(tsample, matrix, &margin, &svcount, 0))
        {
		    if(verbose) printf("LeaveOneOut error: On sample %d out, convergence was not achieved!\n", i);
        }
        free(w);

		/*testing*/
        func = tsample->bias;
        for(k = 0; k < tsample->size; ++k)
        {
            j = (k >= i) ? k+1 : k;
            func += tsample->points[k].alpha * tsample->points[k].y * o_matrix[j][i];
        }

        if(sample->points[i].y*func < 0)
        {
            if(verbose) printf("[%2dx]", i);
            error++;
        }
        else if(verbose) { printf("[%2d ]", i); fflush(stdout); }

        if(verbose > 1)
            printf("Sample %d (%d), function=%lf\n", i, sample->points[i].y, func);

		/*freeing data*/
        data_free_sample(&tsample);
        kernel_free_matrix(matrix, sample->size-1);
	}
	if(verbose) { printf("\n"); }

	return (((double)error)/((double)sample->size)*100.0);

}

/*----------------------------------------------------------*
 * Calculates Leave One Out Error Estimate                  *
 *----------------------------------------------------------*/
double
utils_leave_one_out(sample* sample, int (*train)(struct sample*,double**,double*,int*,int), int skip, int verbose)
{
	register int i = 0, j = 0, k = 0;
	int error = 0;
	double func = 0.0;
    double **matrix = NULL;
    struct sample *tsample = NULL;
    double *w = NULL;
    double margin = 0;
    int svcount = 0;

    matrix = kernel_generate_matrix(sample);

	/*start leave one out*/
    for(i = 0; i< sample->size; ++i)
	{
        if(skip == +1 && sample->points[i].alpha == 0) continue;

		/*Creating temporary data array*/
        tsample = data_remove_point(sample, i);

		/*training*/
        if(!train(tsample, &w, &margin, &svcount, 0))
        {
		    if(verbose) printf("LeaveOneOut error: On sample %d out, convergence was not achieved!\n", i);
        }
        free(w);

		/*testing*/
        func = tsample->bias;
        for(k = 0; k < tsample->size; ++k)
        {
            j = (k >= i) ? k+1 : k;
            func += tsample->points[k].alpha * tsample->points[k].y * matrix[j][i];
        }

        if(sample->points[i].y * func < 0)
        {
            if(verbose) printf("[%2dx]", i);
            error++;
        }
        else if(verbose)
        {
             printf("[%2d ]", i);
             fflush(stdout);
        }

        if(verbose > 1) printf("Sample %d (%d), function=%lf\n", i, sample->points[i].y, func);

		/*freeing data*/
        for(j = 0; j < tsample->size; ++j)
            free(tsample->points[j].x);
        free(tsample->points);
        free(tsample->fnames);
        free(tsample);
        tsample = NULL;
	}
	if(verbose) printf("\n");

    /*free stuff*/
    kernel_free_matrix(matrix, sample->size);

	return (((double)error)/((double)sample->size)*100.0f);
}

/*----------------------------------------------------------*
 * Calcula o Erro com um K-Fold Cross-Validation            *
 *----------------------------------------------------------*/
double
utils_k_fold(sample *sample, int (*train)(struct sample*,double**,double*,int*,int), int fold, int seed, int verbose)
{
	register int i = 0, j = 0, k = 0;
	double error = 0;
	int qtdpos = 0, qtdneg = 0;
    int gasto_pos = 0, gasto_neg = 0;
	int *veterro;
	register double func = 0.0;
    struct sample *sample_pos = NULL, *sample_neg = NULL;
    struct sample *train_sample = NULL, *test_sample = NULL, *traintest_sample = NULL;
    struct sample **vet_sample_pos = NULL, **vet_sample_neg = NULL, **vet_sample_final = NULL;
    double *w = NULL;
    double margin = 0;
    int svcount = 0;
    double **matrix = NULL;
    //double norm = 0;

    //double q_anterior = sample->q;
    //sample->q = 2;

    kernel_type  = sample->kernel_type;
    kernel_param = sample->kernel_param;

    veterro = (int*) malloc(fold*sizeof(int));
    if(veterro == NULL) { printf("Error: Out of memory\n"); exit(1); }
    for(i = 0; i < fold; i++) veterro[i] = 0;

    sample_pos = data_copy_sample_zero(sample);
    sample_neg = data_copy_sample_zero(sample);

    for(i = 0; i < sample->size; i++)
        if(sample->points[i].y == 1)
            sample_pos = data_insert_point(sample_pos, sample, i);
        else
            sample_neg = data_insert_point(sample_neg, sample, i);

    qtdpos = sample_pos->size;
    qtdneg = sample_neg->size;

    if(verbose > 1)
    {
        printf("\nTotal de pontos: %d\n", sample->size);
        printf("Qtde Pos.: %d\n", qtdpos);
        printf("Qtde Neg.: %d\n\n", qtdneg);
    }

    srand(seed);
    /*randomize*/
    for(i = 0; i < sample_pos->size; i++)
    {
        struct point aux;
        j = rand()%(sample_pos->size);
        aux = sample_pos->points[i];
        sample_pos->points[i] = sample_pos->points[j];
        sample_pos->points[j] = aux;
    }
    for(i = 0; i < sample_neg->size; i++)
    {
        struct point aux;
        j = rand()%(sample_neg->size);
        aux = sample_neg->points[i];
        sample_neg->points[i] = sample_neg->points[j];
        sample_neg->points[j] = aux;
    }

    vet_sample_pos   = (struct sample**) malloc(fold*sizeof(struct sample*));
    vet_sample_neg   = (struct sample**) malloc(fold*sizeof(struct sample*));
    vet_sample_final = (struct sample**) malloc(fold*sizeof(struct sample*));

    for(i = 0; i < fold; i++)
    {
        vet_sample_pos[i]   = data_copy_sample_zero(sample);
        vet_sample_neg[i]   = data_copy_sample_zero(sample);
        vet_sample_final[i] = data_copy_sample_zero(sample);
    }

    for(i = 0, j = 0; i < fold-1; i++)
    {
        for(; j < ((sample_pos->size)-gasto_pos)/(fold-i)+gasto_pos; j++)
            vet_sample_pos[i] = data_insert_point(vet_sample_pos[i], sample_pos, j);
        gasto_pos = j;
    }
    for(; j < sample_pos->size; j++)
        vet_sample_pos[i] = data_insert_point(vet_sample_pos[i], sample_pos, j);

    for(i = 0, j = 0; i < fold-1; i++)
    {
        for(; j < ((sample_neg->size)-gasto_neg)/(fold-i)+gasto_neg; j++)
            vet_sample_neg[fold-1-i] = data_insert_point(vet_sample_neg[fold-1-i], sample_neg, j);
        gasto_neg = j;
    }
    for(; j < sample_neg->size; j++)
        vet_sample_neg[fold-1-i] = data_insert_point(vet_sample_neg[fold-1-i], sample_neg, j);

    data_free_sample(&sample_pos);
    data_free_sample(&sample_neg);

    for(i = 0; i < fold; i++)
    {
        for(j = 0; j < vet_sample_pos[i]->size; j++)
            vet_sample_final[i] = data_insert_point(vet_sample_final[i], vet_sample_pos[i], j);
        for(; j < vet_sample_pos[i]->size + vet_sample_neg[i]->size; j++)
            vet_sample_final[i] = data_insert_point(vet_sample_final[i], vet_sample_neg[i], j-vet_sample_pos[i]->size);
    }

    for(i = 0; i < fold; i++)
    {
        data_free_sample(&vet_sample_pos[i]);
        data_free_sample(&vet_sample_neg[i]);
    }
    free(vet_sample_pos);
    free(vet_sample_neg);

    /*start cross-validation*/
    for(j = 0; j < fold; ++j)
    {
        test_sample = data_copy_sample(vet_sample_final[j]);

        train_sample = data_copy_sample_zero(sample);
        for(i = 0; i < fold; ++i)
            if(i != j)
                for(k = 0; k < vet_sample_final[i]->size; k++)
                    train_sample = data_insert_point(train_sample, vet_sample_final[i], k);

        if(verbose)
        {
            printf("Cross-Validation %d:\n", j+1);
            printf("Pts de Treino: %d\n", train_sample->size);
            printf("Pts de Teste:  %d\n", test_sample->size);
        }

        /*training*/
        w = NULL;
        margin = 0;
        svcount = 0;
        if(!train(train_sample, &w, &margin, &svcount, 0))
        {
            if(verbose) printf("Erro no %d-fold: No conjunto %d, a convergencia nao foi alcancada!\n", fold, j+1);
            continue;
        }

        //printf("Margin: %lf\n", margin);

        if(sample->kernel_type == 9)
        {   /*testing imap*/
            for(i = 0; i < test_sample->size; ++i)
            {
                for(func = train_sample->bias, k = 0; k < train_sample->dim; ++k)
                    func += w[k] * test_sample->points[i].x[k];

                if(test_sample->points[i].y * func <= 0)
                {
                    if(verbose > 1) printf("[%2dx] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
                    veterro[j]++;
                }
                else
                {
                    if(verbose > 1) printf("[%2d ] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
                }
                if(verbose) fflush(stdout);
            }
        }
        else
        {   /*testing imadual and smo*/
            traintest_sample = data_join_samples(test_sample, train_sample);
            matrix = kernel_generate_matrix(traintest_sample);

            for(i = 0; i < test_sample->size; ++i)
            {
                for(func = train_sample->bias, k = 0; k < train_sample->size; ++k)
                    func += train_sample->points[k].alpha * train_sample->points[k].y * matrix[k+test_sample->size][i];

                if(test_sample->points[i].y * func <= 0)
                {
                    if(verbose > 1) printf("[%2dx] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
                    veterro[j]++;
                }
                else
                {
                    if(verbose > 1) printf("[%2d ] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
                }
                if(verbose) fflush(stdout);
            }
            kernel_free_matrix(matrix, traintest_sample->size);
        }

        if(verbose) printf("Erro %2d: %d -- %.2lf%%\n", j+1, veterro[j], ((double)veterro[j]/(double)vet_sample_final[j]->size)*100.0f);
        error += ((double)veterro[j]/(double)vet_sample_final[j]->size)*100.0f;

        /*freeing data*/
        free(w);
        data_free_sample(&train_sample);
        data_free_sample(&test_sample);
        data_free_sample(&traintest_sample);
    }

    /*freeing data*/
    for(i = 0; i < fold; i++)
        data_free_sample(&vet_sample_final[i]);
    free(vet_sample_final);
    free(veterro);

    //sample->q = q_anterior;

	return (((double)error)/(double)fold);
}

/*----------------------------------------------------------*
 * Calcula o Erro de Validacao, alem de chamar o k-fold     *
 *----------------------------------------------------------*/
void
utils_validation(sample *train_sample, sample *test_sample, int (*train)(struct sample*,double**,double*,int*,int), int fold, int qtde, int verbose)
{
	register  int i = 0, k = 0;
	double error = 0;
	int erro = 0;
	double errocross = 0;
	register double func = 0.0;
    sample *traintest_sample = NULL;
    double *w = NULL;
    double margin = 0;
    int svcount = 0;
    double **matrix = NULL;

    //double q_anterior = train_sample->q;
    //train_sample->q = 2;

    kernel_type  = train_sample->kernel_type;
    kernel_param = train_sample->kernel_param;

    /*cross-validation*/
    if(qtde > 0)
    {
//        for(errocross = 0, i = 0; i < qtde; i++)
//        {
//            if(verbose) printf("\nExecucao %d / %d:\n", i+1, qtde);
//            errocross += utils_k_fold(train_sample, train, fold, i, verbose);
//        }
//        printf("\nErro %d-Fold Cross Validation: %2.2lf%%\n", fold, errocross/qtde);
        double *errocrossVet = (double*) malloc(qtde*sizeof(double));
        if(!errocrossVet) return;

        for(errocross = 0, i = 0; i < qtde; i++)
        {
            if(verbose) printf("\nExecucao %d / %d:\n", i+1, qtde);
            errocrossVet[i] = utils_k_fold(train_sample, train, fold, i, verbose);
            errocross += errocrossVet[i];
            printf("Erro Execucao %d / %d: %.2lf%%\n", i+1, qtde, errocrossVet[i]);
        }
        printf("\nErro Medio %d-Fold Cross Validation: %.2lf %c %.2lf\n", fold, errocross/qtde, 241, data_standard_deviation(errocrossVet, qtde));
        free(errocrossVet);
    }

    /*start final validation*/
    if(verbose)
    {
        printf("\nFinal Validation:\n");
        printf("Pts de Treino: %d\n", train_sample->size);
        printf("Pts de Teste:  %d\n", test_sample->size);
    }

    /*training*/
    if(!train(train_sample, &w, &margin, &svcount, 0))
        if(verbose) printf("Erro na validacao: a convergencia nao foi alcancada no conjunto de treinamento!\n");

    if(train_sample->kernel_type == 9)
    {   /*testing imap*/
        for(i = 0; i < test_sample->size; ++i)
        {
            for(func = train_sample->bias, k = 0; k < train_sample->dim; ++k)
                func += w[k] * test_sample->points[i].x[k];

            if(test_sample->points[i].y * func <= 0)
            {
                if(verbose > 1) printf("[%2dx] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
                erro++;
            }
            else
            {
                if(verbose > 1) printf("[%2d ] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
            }
            if(verbose) fflush(stdout);
        }
    }
    else
    {   /*testing imadual and smo*/
        traintest_sample = data_join_samples(test_sample, train_sample);
        matrix = kernel_generate_matrix(traintest_sample);

        for(i = 0; i < test_sample->size; ++i)
        {
            for(func = train_sample->bias, k = 0; k < train_sample->size; ++k)
                func += train_sample->points[k].alpha * train_sample->points[k].y * matrix[k+test_sample->size][i];

            if(test_sample->points[i].y * func <= 0)
            {
                if(verbose > 1) printf("[%2dx] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
                erro++;
            }
            else
            {
                if(verbose > 1) printf("[%2d ] function: %lf, y: %d\n", i+1, func, test_sample->points[i].y);
            }
            if(verbose) fflush(stdout);
        }
        kernel_free_matrix(matrix, traintest_sample->size);
    }

    printf("Erro de Validacao: %d -- %.2lf%%\n", erro, (double)erro/(double)test_sample->size*100.0f);
    error += ((double)erro/(double)test_sample->size)*100.0f;

    /*freeing data*/
    free(w);
    w = NULL;
    data_free_sample(&traintest_sample);

    //train_sample->q = q_anterior;
}

/*----------------------------------------------------------*
 * Plots results                                            *
 *----------------------------------------------------------*/
void
utils_plot_results(sample *sample, double shade)
{
    register int i = 0, j = 0, k = 0;
    int max_step = 500;
    double max[2];
    double min[2];
    double step;
    double step2;
    if(sample->dim != 2)
    {
        printf("Can only plot dim = 2!\n");
        return;
    }

    /*open file*/
    FILE* file1 = fopen("temp_1", "w");
    FILE* file2 = fopen("temp_2", "w");
    FILE* file3 = fopen("temp_3", "w");
    FILE* file4 = fopen("temp_4", "w");
    if(file1 == NULL) return;
    if(file2 == NULL) return;
    if(file3 == NULL) return;
    if(file4 == NULL) return;

    for(i = 0; i < sample->size; ++i)
    {
        for(j = 0; j < sample->dim; ++j)
        {
            if(i == 0 || min[j] > sample->points[i].x[j])
                min[j] = sample->points[i].x[j];
            if(i == 0 || max[j] < sample->points[i].x[j])
                max[j] = sample->points[i].x[j];

            if(sample->points[i].y == +1)
                fprintf(file1, "%lf ", sample->points[i].x[j]);
            else
                fprintf(file2, "%lf ", sample->points[i].x[j]);
        }
        if(sample->points[i].y == +1)
            fprintf(file1, "\n");
        else
            fprintf(file2, "\n");
    }
    fclose(file1);
    fclose(file2);

    step  = 3*(max[0]-min[0])/max_step;
    step2 = 3*(max[1]-min[1])/max_step;
    for(i = 0; i < max_step; ++i)
    {
        double x0 = 3*min[0]+i*step;
        for(j = 0; j < max_step; ++j)
        {
            double func = sample->bias;
            double x1 = 3*min[1]+j*step2;
            point two;
            two.alpha = 0;
            two.y     = 1;
            two.x     = (double*) malloc(2*sizeof(double));
            two.x[0]  = x0;
            two.x[1]  = x1;

            for(k = 0; k < sample->size; ++k)
            {
                if(sample->points[k].alpha > 0)
                    func += sample->points[k].alpha * sample->points[k].y * kernel_function(&sample->points[k], &two, sample->dim);
            }
            //printf("%lf\n",func);
            if(fabs(func) <= shade)
                fprintf(file3, "%lf %lf\n", x0, x1);

            free(two.x);
        }
    }
    fclose(file3);
    fclose(file4);

    system("gnuplot data_plot.gplot\n");
}

void
utils_gnuplot(const char *gnucommand)
{
  char syscommand[1024];
  sprintf(syscommand, "echo %s | gnuplot -persist", gnucommand);
  system(syscommand);
}
/*----------------------------------------------------------*
 * Plot points and w in 2D                                  *
 *----------------------------------------------------------*/
void
utils_plot_2d(sample *sample, double *w, char *base_entrada, char *train)
{
    if(sample->dim != 2)
    {
        printf("Dimensao diferente de 2. Impossivel plotar!\n");
        return;
    }
    register int i;
	FILE *arquivo_pos, *arquivo_neg;
	arquivo_pos = fopen("pos.plt", "w");
	arquivo_neg = fopen("neg.plt", "w");
	for(i = 0; i < sample->size; i++)
		if(sample->points[i].y == 1)
			fprintf(arquivo_pos, "%lf\t%lf\t%d\n", sample->points[i].x[0], sample->points[i].x[1], sample->points[i].y);
		else
			fprintf(arquivo_neg, "%lf\t%lf\t%d\n", sample->points[i].x[0], sample->points[i].x[1], sample->points[i].y);

	fclose(arquivo_pos);
	fclose(arquivo_neg);
	char textoGrafico[200];
	sprintf(textoGrafico, "reset; set term postscript enhanced; set output '2d_%s_%s.eps'; f(x) = %lf*x + %lf; g(x) = %lf*x + %lf; h(x) = %lf*x + %lf; plot 'pos.plt' using 1:2 title '+1' with points, 'neg.plt' using 1:2 title '-1' with points, f(x) notitle with lines ls 1, g(x) notitle with lines ls 2, h(x) notitle with lines ls 2", base_entrada, train, w[0]/-w[1], sample->bias/-w[1], w[0]/-w[1], (sample->bias + sample->margin*sample->norm)/-w[1], w[0]/-w[1], (sample->bias - sample->margin*sample->norm)/-w[1]);
	//printf("%s\n\n", textoGrafico);
	utils_gnuplot(textoGrafico);
	//remove("pos.plt");
	//remove("neg.plt");
}

/*----------------------------------------------------------*
 * Plot points and w in 3D                                  *
 *----------------------------------------------------------*/
void
utils_plot_3d(sample *sample, double *w, char *base_entrada, char *train)
{
    if(sample->dim != 3)
    {
        printf("Dimensao diferente de 3. Impossivel plotar!\n");
        return;
    }
    register int i;
	FILE *arquivo_pos, *arquivo_neg;
	arquivo_pos = fopen("pos.plt", "w");
	arquivo_neg = fopen("neg.plt", "w");
	for(i = 0; i < sample->size; i++)
		if(sample->points[i].y == 1)
			fprintf(arquivo_pos, "%lf\t%lf\t%lf\t%d\n", sample->points[i].x[0], sample->points[i].x[1], sample->points[i].x[2], sample->points[i].y);
		else
			fprintf(arquivo_neg, "%lf\t%lf\t%lf\t%d\n", sample->points[i].x[0], sample->points[i].x[1], sample->points[i].x[2], sample->points[i].y);

	fclose(arquivo_pos);
	fclose(arquivo_neg);
	char textoGrafico[200];
	sprintf(textoGrafico, "reset; set term postscript enhanced; set output '3d_%s_%s.eps'; f(x,y) = %f*x + %f*y + %f; splot 'pos.plt' using 1:2:3 title '+1' with points, 'neg.plt' using 1:2:3 title '-1' with points, f(x,y) notitle with lines ls 1", base_entrada, train, w[0]/-w[2], w[1]/-w[2], sample->bias/-w[2]);
	//printf("%s\n\n", textoGrafico);
	utils_gnuplot(textoGrafico);
	//remove("pos.plt");
	//remove("neg.plt");
}
