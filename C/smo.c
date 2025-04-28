/*****************************************************
 * SMO classifier lib                                *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *****************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include "kernel.h"
#include "data.h"
#include "imadual.h"
#include "smo.h"

#define C 9999 //0.05
#define EPS 0.0000001
#define TOL 0.0001
#define MAX_EPOCH 9999

extern double kernel_param;
extern int kernel_type;

/*----------------------------------------------------------*
 * Fuction to find a second training example to change      *
 *----------------------------------------------------------*/
int
smo_examine_example(sample *sample, smo_learning_data *l_data, double **matrix, int_dll *head, int i1, int verbose)
{
    register int i = 0;
    double y1      = 0;
    double e1      = 0;
    double r1      = 0;
    double alpha1  = 0;

    /*clearning up done list*/
    for(i = 0; i < sample->size; ++i) l_data[i].done = 0;
    l_data[i1].done = 1;

    /*reading stuff from array*/
    y1     = sample->points[i1].y;
    alpha1 = sample->points[i1].alpha;
    if(alpha1 > 0 && alpha1 < C) e1 = l_data[i1].error;
    else                         e1 = smo_function(sample, matrix, head, i1) - y1;

    /*calculating r1*/
    r1 = y1 * e1;

    /*try to find next example by 3 different ways*/
    if((r1 < -TOL && alpha1 < C) || (r1 > TOL && alpha1 > 0))
    {
             if(smo_max_errors(sample,l_data,matrix,head,i1,e1,verbose)    ) return 1;
        else if(smo_iterate_non_bound(sample,l_data,matrix,head,i1,verbose)) return 1;
        else if(smo_iterate_all_set(sample,l_data,matrix,head,i1,verbose)  ) return 1;
    }
    else if(verbose > 2) printf("Return0 -1\n");

    return 0;
}

/*----------------------------------------------------------*
 * Fuction to find second example based on max error e1-e2  *
 *----------------------------------------------------------*/
int
smo_max_errors(sample *sample, smo_learning_data *l_data, double **matrix, int_dll *head, int i1, double e1, int verbose )
{
    int k       = 0;
    int i2      =-1;
    double tmax = 0;
    double e2   = 0;
    double temp = 0;
    int_dll *list = NULL;

    if(verbose>2) printf("  Max errors iterations\n");

    /*iterate through the non-bond examples*/
    list = head->next;
    while(list != NULL)
    {
        k = list->index;
        if(l_data[k].done == 0 && sample->points[k].alpha < C)
        {
            e2 = l_data[k].error;
            temp = fabs(e1-e2);

            if(temp > tmax){ tmax = temp; i2 = k; }
        }
        list = list->next;
    }
    if(i2 >= 0 && smo_take_step(sample,l_data,matrix,head,i1,i2,verbose)) return 1;

    return 0;
}

/*----------------------------------------------------------*
 * Find second example, look at the non-bound examples      *
 *----------------------------------------------------------*/
int
smo_iterate_non_bound(sample *sample, smo_learning_data *l_data, double **matrix, int_dll *head, int i1, int verbose)
{
    int k    = 0;
    int_dll *list = NULL;

    if(verbose>2) printf("  Non-bound iteration\n");

    /* look through all non-bound examples*/
    list = head->next;
    while(list != NULL)
    {
        k = list->index;
        if(l_data[k].done == 0 && sample->points[k].alpha < C)
            if(smo_take_step(sample,l_data,matrix,head,i1,k,verbose)) return 1;
        list = list->next;
    }
    return 0;
}

/*----------------------------------------------------------*
 * Find second example, look at the entire set              *
 *----------------------------------------------------------*/
int
smo_iterate_all_set(sample *sample, smo_learning_data *l_data, double **matrix, int_dll *head, int i1, int verbose)
{
    int k0 = 0;
    register int k  = 0;
    int i2 = 0;
    int size = sample->size;

    if(verbose>2) printf("  All-set iteration\n");

    srand(0);
    /*random starting point*/
    //k0 = 0;
    k0 = rand()%size;

    for(k = k0; k < size+k0; ++k)
    {
        i2 = k%size;
        if(l_data[i2].done == 0 && smo_take_step(sample,l_data,matrix,head,i1,i2,verbose))
            return 1;
    }
    return 0;
}

/*----------------------------------------------------------*
 * Change two alphas in the training set                    *
 *----------------------------------------------------------*/
int
smo_take_step(sample *sample, smo_learning_data *l_data, double **matrix, int_dll *head, int i1, int i2, int verbose)
{
    int i=0, y1=0, y2=0, s=0; //, size=0;
    double alpha1=0, alpha2=0, new_alpha1=0, new_alpha2=0;
    double e1=0, e2=0, min_val=0, max_val=0, eta=0;
    double max_val_f=0, min_val_f=0;
    double bnew=0, b=0; //delta_b=0 , b=0;
    double t1=0, t2=0, error_tot=0;
    int_dll *list = NULL;

    /*this sample is done*/
    l_data[i2].done = 1;

    /*get info from sample struct*/
    b      = -sample->bias;
    y1     = sample->points[i1].y;
    y2     = sample->points[i2].y;
    //size   = sample->size;
    alpha1 = sample->points[i1].alpha;
    alpha2 = sample->points[i2].alpha;

    /*get error values for i1*/
    if(alpha1 > 0 && alpha1 < C) e1 = l_data[i1].error;
    else                         e1 = smo_function(sample,matrix,head,i1) - y1;

    /*get error values for i2*/
    if(alpha2 > 0 && alpha2 < C) e2 = l_data[i2].error;
    else                         e2 = smo_function(sample,matrix,head,i2) - y2;

    /*calculate s*/
    s = y1*y2;

    /*compute min and max*/
	if(s == -1)
    {
		min_val = utils_max(0, alpha2 - alpha1);
		max_val = utils_min(C, C + alpha2 - alpha1);
	}
	else
    {
		min_val = utils_max(0, alpha2 + alpha1 - C);
		max_val = utils_min(C, alpha1 + alpha2);
	}
    if(min_val == max_val){ if(verbose>2) printf("return0 2\n"); return 0;}

    /*compute eta*/
    eta = 2.0 * matrix[i1][i2] - matrix[i1][i1] - matrix[i2][i2];

    /*compute new alpha2*/
    if(eta < 0)
    {
        new_alpha2 = alpha2 + y2*(e2-e1)/eta;

             if(new_alpha2 < min_val) new_alpha2 = min_val;
        else if(new_alpha2 > max_val) new_alpha2 = max_val;
    }
    else
    {
        /*computing min and max functions*/
        double c1 = eta/2.0;
        double c2 = y2 * (e1-e2) - eta*alpha2;
        min_val_f = c1 * min_val * min_val + c2*min_val;
        max_val_f = c1 * max_val * max_val + c2*min_val;

             if(min_val_f > max_val_f + EPS) new_alpha2 = min_val;
        else if(min_val_f < max_val_f - EPS) new_alpha2 = max_val;
        else new_alpha2 = alpha2;
    }

    /*exit if no change made*/
    if(fabs(new_alpha2-alpha2) < EPS*(new_alpha2+alpha2+EPS))
    {
        if(verbose>2)printf("return0 3\n");
        return 0;
    }

    /*calculate new alpha1*/
    new_alpha1 = alpha1 - s*(new_alpha2-alpha2);
    if(new_alpha1 < 0)
    {
        new_alpha2+= s*new_alpha1;
        new_alpha1 = 0;
    }
    else if(new_alpha1 > C)
    {
        new_alpha2+= s * (new_alpha1-C);
        new_alpha1 = C;
    }
    /*saving new alphas*/
    sample->points[i1].alpha = new_alpha1;
    sample->points[i2].alpha = new_alpha2;

    /*saving new stuff into sv list*/
    if(new_alpha1 > 0 && l_data[i1].sv == NULL)
    {
        int_dll *list = utils_int_dll_append(head);
        list->index = i1;
        l_data[i1].sv = list;
    }
    else if(new_alpha1 == 0 && l_data[i1].sv !=NULL)
        utils_int_dll_remove(&(l_data[i1].sv));

    if(new_alpha2 > 0 && l_data[i2].sv == NULL)
    {
        int_dll *list = utils_int_dll_append(head);
        list->index = i2;
        l_data[i2].sv = list;
    }
    else if(new_alpha2 == 0 && l_data[i2].sv !=NULL)
        utils_int_dll_remove(&(l_data[i2].sv));

    /*update bias*/
    t1 = y1 * (new_alpha1 - alpha1);
    t2 = y2 * (new_alpha2 - alpha2);

    if(new_alpha1 > 0 && new_alpha1 < C)
        bnew = b + e1 + t1*matrix[i1][i1] + t2*matrix[i1][i2];
    else
    {
        if(new_alpha2 > 0 && new_alpha2 < C)
            bnew = b + e2 + t1*matrix[i1][i2] + t2*matrix[i2][i2];
        else
        {
            double b1 = 0, b2 = 0;
            b2 = b + e1 + t1*matrix[i1][i1] + t2*matrix[i1][i2];
            b1 = b + e2 + t1*matrix[i1][i2] + t2*matrix[i2][i2];
            bnew = (b1+b2)/2.0;
        }
    }
    //delta_b = bnew - b;
    b = bnew;
    sample->bias = -b;

    /*updating error cache*/
    error_tot = 0;
    list = head->next;
    while(list != NULL)
    {
        i = list->index;
        if((i != i1 && i !=i2) && sample->points[i].alpha < C)
        {
            l_data[i].error = smo_function(sample,matrix,head,i) - sample->points[i].y;
            error_tot += l_data[i].error;
        }
        list = list->next;
    }
    l_data[i1].error = 0.0;
    l_data[i2].error = 0.0;

    if(verbose>1)
        printf("Total error= %lf, alpha(%d)= %lf, alpha(%d)= %lf\n", error_tot, i1, new_alpha1, i2, new_alpha2);

    return 1;
}

/*----------------------------------------------------------*
 * Prints the result of training                            *
 *----------------------------------------------------------*/
void
smo_test_learning(sample *sample, double **matrix, smo_learning_data *l_data, int_dll *head)
{
    register int i = 0;
    for(i = 0; i < sample->size; ++i)
        printf("%d -> %lf (erro=%lf) (alpha=%1.10lf)\n", i+1, smo_function(sample, matrix, head, i), l_data[i].error, sample->points[i].alpha);
}

/*----------------------------------------------------------*
 * Returns function evaluation ate point "index"            *
 *----------------------------------------------------------*/
double
smo_function(sample *sample, double **matrix, int_dll *head, int index)
{
    register int i = 0;
    register double sum = 0;
    int_dll *list = head->next;

    while(list != NULL)
    {
        i = list->index;
        if(sample->points[i].alpha > 0)
            sum += sample->points[i].alpha * sample->points[i].y * matrix[i][index];
        list = list->next;
    }
    sum += sample->bias;

    return sum;
}

/*----------------------------------------------------------*
 * Training function                                         *
 *----------------------------------------------------------*/
int
smo_train_matrix(sample *sample, double **matrix, double *margin, int *svs, int verbose)
{
    register int i = 0;
    int ret = 1;
    double norm = 1;
    smo_learning_data *l_data = NULL;
    int_dll *head = NULL;

    //srand(0);

    /*creating support vector linked list*/
    head = utils_int_dll_create();

    /*allocating array for l_data*/
    l_data = (smo_learning_data*) malloc((sample->size)*sizeof(smo_learning_data));
    if(l_data == NULL) { return 0; }

    /*clear data*/
    sample->bias = 0;
    for(i = 0; i < sample->size; i++)
        sample->points[i].alpha = 0;

    /*run training algorithm*/
    ret = smo_training_routine(sample, l_data, matrix, head, verbose);

    norm = kernel_feature_space_norm(sample, matrix);
    (*margin) = 1.0/norm;

    *svs = 0;
    for(i = 0; i < sample->size; ++i)
    {
        if(sample->points[i].alpha > 0) ++(*svs);
        if(sample->points[i].alpha > C) ret = 0;
    }

    /*free stuff*/
    utils_int_dll_free(&head);
    free(l_data);

    return ret;
}

/*----------------------------------------------------------*
 * Training function                                         *
 *----------------------------------------------------------*/
int
smo_train(sample *sample, double **w, double *margin, int *svs, int verbose)
{
    register int i = 0;
    int ret = 1;
    double **matrix = NULL;
    double norm = 1;
    smo_learning_data *l_data = NULL;
    int_dll *head = NULL;
    double *w_saved = NULL;

    kernel_type  = sample->kernel_type;
    kernel_param = sample->kernel_param;

    //srand(0);
    //srand(time(NULL));

    /*creating support vector linked list*/
    head = utils_int_dll_create();

    /*allocating array for l_data*/
    l_data = (smo_learning_data*) malloc((sample->size)*sizeof(smo_learning_data));
    if(l_data == NULL) { printf("!!BUG- data is empty\n"); return 0; }

    /*clear data*/
    sample->bias = 0;
    for(i = 0; i < sample->size; i++)
        sample->points[i].alpha = 0;

    /*run training algorithm*/
    matrix = kernel_generate_matrix(sample);
    ret = smo_training_routine(sample, l_data, matrix, head, verbose);

    kernel_print_matrix(matrix, sample->size);

    norm = kernel_feature_space_norm(sample, matrix);
    if(kernel_type == 0)
        w_saved = utils_get_weight(sample);
    else
    {
        if(kernel_type == 1 && kernel_param == 1)
            w_saved = utils_get_dualweight_prodint(sample);
        else
            w_saved = utils_get_dualweight(sample);
    }

    *w = w_saved;

    (*margin) = 1.0/norm;

    *svs = 0;
    for(i = 0; i < sample->size; ++i)
    {
        if(sample->points[i].alpha > 0) ++(*svs);
        if(sample->points[i].alpha > C) ret = 0;
    }

    if(verbose)
    {
        printf("Numero de Vetores Suporte: %d\n", *svs);
        printf("Margem encontrada: %lf\n\n", *margin);
        if(verbose > 1)
        {
            for(i = 0; i < sample->dim; i++)
                printf("W[%d]: %lf\n", sample->fnames[i], w_saved[i]);
            printf("Bias: %lf\n\n", sample->bias);
        }
    }

    /*free stuff*/
    kernel_free_matrix(matrix, sample->size);
    utils_int_dll_free(&head);
    free(l_data);
    //ret = 1;
    return ret;
}

/*----------------------------------------------------------*
 * Training function                                         *
 *----------------------------------------------------------*/
int
smo_training_routine(sample *sample, smo_learning_data *l_data, double **matrix, int_dll *head, int verbose)
{
    int epoch       = 0;
    int k           = 0;
    int num_changed = 0;
    int tot_changed = 0;
    int examine_all = 1;

    /*initialize variables*/
    sample->bias = 0;
    for(k = 0; k < sample->size; ++k)
    {
        sample->points[k].alpha = 0;
        l_data[k].error = 0;
        l_data[k].done  = 0;
        l_data[k].sv    = NULL;
    }

    /*training*/
    while(num_changed > 0 || examine_all)
    {
        /*stop if iterated too much!*/
        if(epoch > MAX_EPOCH) return 0;

        num_changed = 0;
        if(examine_all)
            for(k = 0; k < sample->size; ++k)
                num_changed += smo_examine_example(sample, l_data, matrix, head, k, verbose);
        else
            for(k = 0; k < sample->size; ++k)
                if(sample->points[k].alpha > 0 && sample->points[k].alpha < C)
                    num_changed += smo_examine_example(sample, l_data, matrix, head, k, verbose);

             if(examine_all == 1) examine_all = 0;
        else if(num_changed == 0) examine_all = 1;
        tot_changed += num_changed;
        ++epoch;
    }

    /*final verbose*/
    if(verbose)
    {
        smo_test_learning(sample, matrix, l_data, head);
        //printf("Margin = %lf, number of changes %d\n",1.0/norm,tot_changed);
        //free(w);
    }
    return 1;
}
