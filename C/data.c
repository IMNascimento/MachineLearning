/* data.c:  Data input/output lib
            Data features manipulate
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include "data.h"

/*----------------------------------------------------------*
 * Reads data from file (svm-light format).                 *
 *----------------------------------------------------------*/
int
data_load(char* fname, sample **sample, int verbose)
{
    int ret = 0;
    FILE *file = fopen(fname, "r");
    if(file == NULL)
    {
        if(verbose) printf("Error: Could not read file!\n");
        return 0;
    }
    (*sample) = (struct sample*) malloc(sizeof(struct sample));
    if((*sample) == NULL)
    {
        if(verbose) printf("Error: No memory!\n");
        return 0;
    }
    ret = data_load_routine(file, *sample, verbose);
    if(!ret) { free(*sample); }
    fclose(file);
    return ret;
}

/*----------------------------------------------------------*
 * Reads data from file.                                    *
 *----------------------------------------------------------*/
int
data_load_routine(FILE* file, sample *sample, int verbose)
{
    if(!file) { if(verbose) printf("Error: Could not read file!\n"); return 0; }

    int lines  = 0;
    int dim    = 0;
    int i      = 0;
    int ret    = 1;
    int fflag  = 1; /*indicates if it needs to save feature names*/
    char c     = '\0';

    pnt_ll* dhead = NULL;
    pnt_ll* dlist = NULL;
    pnt_ll* dtmp  = NULL;

    sample->fnames       = NULL;
    sample->points       = NULL;
    sample->index        = NULL;
    sample->bias         = 0;
    sample->q            = 2;
    sample->p            = 2;
    sample->max_time     = 1;
    sample->mult_tempo   = 1;
    sample->kernel_param = 1;
    sample->kernel_type  = 9;
    sample->flexivel     = 0;
    sample->alpha_aprox  = 0;
    sample->normalized   = 0;

    /*reading in bias if exists*/
    if(fscanf(file, "b%lf", &(sample->bias)) == 1)
    { if (verbose) printf("Got bias...ata_wr%lf \n", sample->bias); }

    /* reding data file */
    while(!feof(file))
    {
        int label    = 0;
        int pos      = 0;
        int lastpos  = 0;
        double val   = 0;
        int count    = 1;
        x_ll* head   = NULL;
        x_ll* list   = NULL;
        x_ll* tmp    = NULL;
        double *x    = NULL;
        double alpha = 0.0;

        /*Scanning label*/
        if(fscanf(file, "%d", &label) != 1) break;
        if(abs(label) != 1)
        {
            if(verbose) printf("Label must be -1 or +1, found:%d, at line %d\n", label, i);
            ret = 0;
            break;
        }

        /*allocating space for this data*/
        dtmp = (pnt_ll*) malloc(sizeof(pnt_ll));
        if(dtmp == NULL) { if(verbose) printf("Out of memory3\n"); ret=0; break; }
        if(dlist) dlist->next = dtmp;
        if(!dhead) dhead = dtmp;
        dtmp->next = NULL;
        dlist = dtmp;

        /*verbose output*/
        if(verbose > 1) printf("%d\t", label);

        /*Reading x values*/
        c = fgetc(file);
        while(c != EOF && c != '\n')
        {
            if(isdigit(c))
            {
                /*Scanning x value*/
                ungetc(c, file);
                if(fscanf(file, "%d:%lf", &pos, &val) != 2)
                { if(verbose) printf("Error: File format is invalid\n"); ret=0; break; }

                /*Error check*/
                if(sample->fnames != NULL && sample->fnames[count-1] != pos)
                { if(verbose) printf("Error: File format is invalid\n"); ret=0; break; }

                /*Error check*/
                if(lastpos >= pos)
                { if(verbose) printf("Error: Feature names must be in order.\n"); ret=0; break; }
                lastpos = pos;

                /*allocating space for x value*/
                tmp = (x_ll*) malloc(sizeof(x_ll));
                if(tmp == NULL) { if(verbose) printf("Out of memory!\n"); ret=0; break; }

                /*setting vales */
                tmp->val   = val;
                tmp->fname = pos;
                tmp->next  = NULL;

                if(list) list->next = tmp;
                if(!head) head = tmp;
                list = tmp;

                count++;
            }
            /*reading alpha value if exists*/
            else if(c == 'A')
            {
                if(fscanf(file, ":%lf", &alpha) != 1)
                { if(verbose) printf("Error: File format is invalid\n"); ret=0; break; }
            }
            c = fgetc(file);
        }
        /*error check */
        if(dim > 0 && count-1 != dim)
        { if(verbose) printf("Error: Samples do not all have the same dimension\n"); ret=0; break; }

        /*panic exit, clean up!*/
        if(!ret)
        {
            list = head;
            while(list != NULL) { tmp = list; list = list->next; free(tmp); }
            break;
        }
        dim = count-1;

        /*allocating an array for x*/
        x = (double*) malloc(dim*sizeof(double));
        if(x == NULL) { if(verbose) printf("Out of memory.\n"); ret=0; break; }

        /*allocating space for feature name array*/
        if(sample->fnames == NULL) sample->fnames = (int*) malloc(dim*sizeof(int));
        if(sample->fnames == NULL) { if(verbose) printf("Out of memory.\n"); ret=0; break; }

        /*Creating new structure data*/
        list = head; i = 0;
        while(list != NULL)
        {
            x_ll* tmp = list;
            if(fflag) sample->fnames[i] = list->fname;
            x[i++] = list->val;
            list   = list->next;
            free(tmp);
            if(verbose > 1) printf("%lf,", x[i-1]);
        }
        dlist->data.x     = x;
        dlist->data.alpha = alpha;
        dlist->data.y     = label;
        lines++;
        fflag = 0;

        /*verbose output*/
        if(verbose > 1) printf("\n");
    }
    if(lines == 0) ret=0;

    /*Emergency exit, clean up*/
    if(!ret)
    {
        dlist = dhead;
        while(dlist != NULL) { dtmp = dlist; dlist = dlist->next; free(dtmp); }
        return ret;
    }

    /*allocating data array*/
    sample->points = (point*) malloc(lines*sizeof(point));
    if(sample->points == NULL) { if(verbose) printf("Out of memory.\n"); return 0; }
    i = 0;
    dlist = dhead;
    while(dlist != NULL)
    {
        pnt_ll* tmp = dlist;
        sample->points[i++] = dlist->data;
        dlist = dlist->next;
        free(tmp);
    }

    /*verbose*/
    if(verbose) printf("Tamanho  = %d\nDimensao = %d\n\n", i, dim);

    /*set dimension and size*/
    sample->size = i;
    sample->dim  = dim;

    return 1;
}

/*----------------------------------------------------------*
 * Save support vectors and alpha                           *
 *----------------------------------------------------------*/
int
data_write(char *fname, sample *sample, int sv_only)
{
    int ret = 0;

//    char arquivo_saida_temp[100] = "db/out/";
//    strcat(arquivo_saida_temp, fname);
//
//    srand(time(NULL));
//    int timestamp = rand()%100000 * rand()%100000;
//    char stimestamp[6];
//    srand(0);
//
//    printf("Timestamp: %d\n\n", timestamp);
//
//    register int i, j, k;
//    int digito;
//    char temp[6];
//
//    for(i = 0; timestamp != 0; i++)
//    {
//        digito = timestamp % 10;
//        temp[i] = (char)(digito + 48); //48 = (int)'0';
//        timestamp /= 10;
//    }
//
//    for(j = i-1, k = 0; j >= 0; j--, k++)
//        stimestamp[k] = temp[j];
//    stimestamp[i] = '\0';
//
//    strcat(arquivo_saida_temp, "_");
//    strcat(arquivo_saida_temp, stimestamp);
//    strcat(arquivo_saida_temp, ".txt");

    /*open file*/
    //FILE* file = fopen(arquivo_saida_temp, "w");
    FILE* file = fopen(fname, "w");
    if(file == NULL) return 0;

    ret = data_write_routine(file, sample, sv_only);
    if(file != stdout) fclose(file);
    return ret;
}

/*----------------------------------------------------------*
 * Save weight vector                                       *
 *----------------------------------------------------------*/
int
data_write_weights(char *fname, double *w, int dim)
{
    register int i;

    /*open file*/
    FILE* file = fopen(fname, "w");
    if( file == NULL) return 0;

    fprintf(file, "%d\n", dim);
    for(i = 0; i < dim+1; ++i) fprintf(file, "%lf\n", w[i]);

    return 1;
}

/*----------------------------------------------------------*
 * load weight vector                                       *
 *----------------------------------------------------------*/
int
data_load_weights(char *fname, double **w, int dim)
{
    register int i;
    int d;

    /*open file*/
    FILE* file = fopen(fname, "r");
    if(file == NULL) return 0;

    fscanf(file, "%d", &d);
    if(d != dim) return 0;

    (*w) = (double*) malloc(sizeof(double)*dim);
    if(!(*w)) return 0;

    for(i = 0; i < dim; ++i)
        fscanf(file, "%lf", &(*w)[i]);

    return 1;
}

/*----------------------------------------------------------*
 * Save support vectors and alpha                           *
 *----------------------------------------------------------*/
int
data_write_routine(FILE *file, sample* sample, int sv_only)
{
    register int i = 0, j = 0;
    if(file == NULL) return 0;

    /*print bias*/
    //fprintf(file, "b%.15f\n", sample->bias);

    /*print data and their alphas*/
    for(i = 0; i < sample->size; ++i)
    {
        if((sv_only && sample->points[i].alpha > 0) || !sv_only)
        {
            fprintf(file, "%d ", sample->points[i].y);

            for(j = 0; j < sample->dim; ++j)
                //fprintf(file,"%d:%d ", sample->fnames[j], (int)sample->points[i].x[j]);
                fprintf(file,"%d:%.15lf ", sample->fnames[j], sample->points[i].x[j]);

            //fprintf(file, "A:%.15f", sample->points[i].alpha);
            fprintf(file, "\n");
        }
    }
    return 1;
}

/*----------------------------------------------------------*
 * Frees samples                                            *
 *----------------------------------------------------------*/
void
data_free_sample(sample **sample)
{
    register int i = 0;
    if(sample == NULL || (*sample) == NULL) return;
    /*freeing data*/
    for(i = 0; i < (*sample)->size; ++i)
        free((*sample)->points[i].x);
    free((*sample)->points);
    free((*sample)->fnames);
    if((*sample)->index) free((*sample)->index);
    free((*sample));
    *sample = NULL;
}

/*----------------------------------------------------------*
 * joins two samples in a new sample                        *
 *----------------------------------------------------------*/
sample*
data_join_samples(sample *sample1, sample *sample2)
{
    register int i = 0, j = 0;
    int index     = 0;
    int s         = 0;
    int dim       = sample1->dim;
    sample *smout = NULL;

    if(sample1->dim > sample2->dim)
    {
        printf("Error, sample1 dimension must be less or equal to sample2\n");
        exit(1);
    }

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory 1\n"); exit(1); }

    /*allocating space for new array*/
    smout->size   = sample1->size + sample2->size;
    smout->points = (point*) malloc(smout->size*sizeof(point));
    if(smout->points == NULL) { printf("Error: Out of memory 2\n"); exit(1); }

    /*allocating space for index array (if exists)*/
    if(sample1->index && sample2->index)
    {
        smout->index = (int*) malloc(smout->size*sizeof(int));
        if(smout->index == NULL) { printf("Error: Out of memory 3\n"); exit(1); }
        for(i = 0; i < sample1->size; ++i) smout->index[i] = sample1->index[i];
        for(i = 0; i < sample2->size; ++i) smout->index[i+sample1->size] = i+sample1->size;
    }
    else
        smout->index = NULL;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc(dim*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory 4\n"); exit(1); }

    /*copying fnames*/
    for(j = 0; j < dim; ++j) smout->fnames[j] = sample1->fnames[j];

    /*copying bias*/
    smout->bias = sample1->bias;
    smout->dim  = sample1->dim;

    /*copying norms p and q, max_time and mult_tempo*/
    smout->q          = sample1->q;
    smout->p          = sample1->p;
    smout->max_time   = sample1->max_time;
    smout->mult_tempo = sample1->mult_tempo;

    /*copying kernel params*/
    smout->kernel_type  = sample1->kernel_type;
    smout->kernel_param = sample1->kernel_param;

    /*copying flexiblity, norm, margin, alpha prox. and normalized*/
    smout->flexivel    = sample1->flexivel;
    smout->norm        = sample1->norm;
    smout->margin      = sample1->margin;
    smout->alpha_aprox = sample1->alpha_aprox;
    smout->normalized  = sample1->normalized;

    /*copying sample1 information to new data array*/
    for(i = 0; i < sample1->size; ++i)
    {
        /*initializing*/
        smout->points[i].y     = sample1->points[i].y;
        smout->points[i].alpha = sample1->points[i].alpha;

        /*allocating space for vector x*/
        smout->points[i].x = (double*) malloc(dim*sizeof(double));
        if(smout->points[i].x == NULL) { printf("Error: Out of memory 5\n"); exit(1); }

        /* copying features */
        for(j = 0; j < dim; ++j) smout->points[i].x[j] = sample1->points[i].x[j];
    }

    /*copying sample2 information to new data array*/
    for(i = 0; i < sample2->size; ++i)
    {
        /*initializing*/
        index = i + sample1->size;
        smout->points[index].y     = sample2->points[i].y;
        smout->points[index].alpha = sample2->points[i].alpha;

        /*allocating space for vector x*/
        smout->points[index].x = (double*) malloc(dim*sizeof(double));
        if(smout->points[index].x == NULL) { printf("Error: Out of memory 6\n"); exit(1); }

        /* copying features */
        s = 0;
        for(j = 0; j < sample2->dim; ++j)
        {
            if(sample1->fnames[s] == sample2->fnames[j])
            {
                smout->points[index].x[s] = sample2->points[i].x[j];
                s++;
            }
        }
        if(s != dim) { printf("Error, s and dim dont match, (%d, %d)\n", s, dim); exit(0); }
    }

    return smout;
}

/*----------------------------------------------------------*
 * Computes norm in dual variables                          *
 *----------------------------------------------------------*/
double
data_calculate_norm(sample *data, double **K)
{
    register int i, j;
    register double sum, sum1;
    point *points = data->points;

    for(sum = 0, i = 0; i < data->size; ++i)
    {
        for(sum1 = 0, j = 0; j < data->size; ++j)
            sum1 += points[j].alpha * points[j].y * K[i][j];

        sum += points[i].y * points[i].alpha * sum1;
    }
    return sqrt(sum);
}

/*----------------------------------------------------------*
 * Returns sample with database normalized                  *
 *----------------------------------------------------------*/
sample*
data_normalize_database(sample *samp)
{
    register int i = 0, j = 0;
    int dim       = samp->dim;
    int size      = samp->size;
    sample *smout = NULL;
    double norm;

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory\n"); exit(1); }

    /*copying data*/
    smout->size         = samp->size;
    smout->dim          = samp->dim+1;
    smout->bias         = samp->bias;
    smout->p            = samp->p;
    smout->q            = samp->q;
    smout->kernel_type  = samp->kernel_type;
    smout->kernel_param = samp->kernel_param;
    smout->max_time     = samp->max_time;
    smout->mult_tempo   = samp->mult_tempo;
    smout->alpha_aprox  = samp->alpha_aprox;
    smout->flexivel     = samp->flexivel;
    smout->norm         = samp->norm;
    smout->margin       = samp->margin;
    smout->normalized   = 1;

    /*allocating space for new array*/
    smout->points = (point*) malloc(size*sizeof(point));
    if(smout->points == NULL) { printf("Error: Out of memory\n"); exit(1); }

    /*allocating space for index array (if exists)*/
    if(samp->index)
    {
        smout->index = (int*) malloc(size*sizeof(int));
        if(smout->index == NULL) { printf("Error: Out of memory\n"); exit(1); }
        for(i = 0; i < size; ++i) smout->index[i] = samp->index[i];
    }
    else
        smout->index = NULL;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc((dim+1)*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory\n"); exit(1); }

    /*copying information to new data array*/
    for(i = 0; i < size; ++i)
    {
        /*initializing*/
        smout->points[i].y     = samp->points[i].y;
        smout->points[i].alpha = samp->points[i].alpha;

        /*allocating space for vector x*/
        smout->points[i].x = (double*) malloc((dim+1)*sizeof(double));
        if(smout->points[i].x == NULL) { printf("Error: Out of memory\n"); exit(1); }

        /* copying features */
        for(norm = 0, j = 0; j < dim; ++j)
        {
            smout->points[i].x[j] = samp->points[i].x[j];
            smout->fnames[j]      = samp->fnames[j];
            norm += pow(fabs(smout->points[i].x[j]), smout->p);
        }
        smout->points[i].x[j] = 1;
        smout->fnames[j]      = j+1;
        norm += pow(fabs(smout->points[i].x[j]), smout->p);
        norm = pow(norm, 1.0/smout->p);
        for(j = 0; j < dim+1; ++j)
            smout->points[i].x[j] /= norm;
    }
    data_free_sample(&samp);
    samp = NULL;
    return smout;
}

/*----------------------------------------------------------*
 * Returns data normalized to max x norm                    *
 *----------------------------------------------------------*/
void
data_norm(sample* sample, double p)
{
    register int i = 0, j = 0;
    register double norm = 0.0;

    for(i = 0; i < sample->size; ++i)
    {
        for(norm = 0, j = 0; j < sample->dim; ++j)
            norm += pow(fabs(sample->points[i].x[j]), p);
        norm = pow(norm, 1.0/p);
        for(j = 0; j < sample->dim; ++j)
            sample->points[i].x[j] /= norm;
    }
}

/*----------------------------------------------------------*
 * Returns data normalized                                  *
 *----------------------------------------------------------*/
void
data_normalized(double *w, int dim, double q)
{
    register int i = 0;
    register double norm = 0.0;
    for(i = 0; i < dim; ++i)
        norm += pow(fabs(w[i]), q);
    norm = pow(norm, 1.0/q);
    for(i = 0; i < dim; ++i)
        w[i] /= norm;
}

/*----------------------------------------------------------*
 * Returns variance of the data.                            *
 *----------------------------------------------------------*/
double
data_get_variance(sample* sample, int index)
{
    register int i = 0, j = 0;
    register double norm = 0.0;
    register double sum  = 0.0;
    double *avg = NULL;

    avg = (double*) malloc(sample->dim*sizeof(double));
    if(avg == NULL) { printf("Error: Out of memory 7\n"); exit(1); }

    for(j = 0; j < sample->dim; ++j)
    {
        if(index < 0 || sample->fnames[j] != index)
        {
            avg[j] = 0.0;
            for(i = 0; i < sample->size; ++i)
                avg[j] += sample->points[i].x[j];
            avg[j] = avg[j] / sample->size;
        }
    }
    sum = 0;
    for(i = 0; i < sample->size; ++i)
    {
        norm = 0;
        for(j = 0; j < sample->dim; ++j)
            if(index < 0 || sample->fnames[j] != index)
                norm += pow(avg[j] - sample->points[i].x[j], 2);
        sum += norm;
    }
    sum = sum/sample->size;

    free(avg);
    return sum;
}

/*----------------------------------------------------------*
 * Returns radius of the ball that circ. the data.          *
 *----------------------------------------------------------*/
double
data_get_radius(sample* sample, int index, double q)
{
    register int i = 0, j = 0;
    register double norm = 0.0;
    double max  = 1.0;
    double *avg = NULL;

    if(q == 2)
    {
        avg = (double*) malloc(sample->dim*sizeof(double));
        if(avg == NULL) { printf("Error: Out of memory 8\n"); exit(1); }
        for(j = 0; j < sample->dim; ++j)
        {
            if(index < 0 || sample->fnames[j] != index)
            {
                avg[j] = 0.0;
                for(i = 0; i < sample->size; ++i)
                    avg[j] += sample->points[i].x[j];
                avg[j] = avg[j] / sample->size;
            }
        }
        for(max = 0, i = 0; i < sample->size; ++i)
        {
            for(norm = 0, j = 0; j < sample->dim; ++j)
                if(index < 0 || sample->fnames[j] != index)
                    norm += pow(avg[j] - sample->points[i].x[j], 2);
            norm = sqrt(norm);
            if(max < norm)
                max = norm;
        }
        free(avg);
    }
    else if(q == 1)
    {
        for(max = 0, i = 0; i < sample->size; ++i)
            for(j = 0; j < sample->dim; ++j)
                if(index < 0 || sample->fnames[j] != index)
                    if(max < fabs(sample->points[i].x[j]))
                        max = fabs(sample->points[i].x[j]);
    }
    return max;
}

/*----------------------------------------------------------*
 * Returns distance of centers of classes                   *
 *----------------------------------------------------------*/
double
data_get_dist_centers(sample *sample, int index)
{
    register int i = 0, j = 0;
    double *avg_pos = NULL;
    double *avg_neg = NULL;
    register double dist = 0.0;
    register int size_pos = 0, size_neg = 0;

    for(size_pos = 0, size_neg = 0, i = 0; i < sample->size; i++)
        if(sample->points[i].y == 1) size_pos++;
        else                         size_neg++;

    avg_pos = (double*) malloc(sample->dim*sizeof(double));
    if(avg_pos == NULL) { printf("Error: Out of memory 9\n"); exit(1); }
    avg_neg = (double*) malloc(sample->dim*sizeof(double));
    if(avg_neg == NULL) { printf("Error: Out of memory 10\n"); exit(1); }

    for(j = 0; j < sample->dim; ++j)
    {
        avg_pos[j] = 0.0;
        avg_neg[j] = 0.0;
        for(i = 0; i < sample->size; ++i)
            if(sample->points[i].y == 1)
                avg_pos[j] += sample->points[i].x[j];
            else
                avg_neg[j] += sample->points[i].x[j];
        avg_pos[j] /= (double)size_pos;
        avg_neg[j] /= (double)size_neg;
    }

    for(dist = 0.0, j = 0; j < sample->dim; ++j)
        if(index < 0 || sample->fnames[j] != index)
            dist += pow(avg_pos[j]-avg_neg[j], 2);

    return sqrt(dist);
}

/*----------------------------------------------------------*
 * Returns distance of centers of classes without feats     *
 *----------------------------------------------------------*/
double
data_get_dist_centers_without_feats(sample *sample, int *feats, int fsize, int index)
{
    register int i = 0, j = 0;
    double *avg_pos = NULL;
    double *avg_neg = NULL;
    register double dist = 0.0;
    register int size_pos = 0, size_neg = 0;

    for(size_pos = 0, size_neg = 0, i = 0; i < sample->size; i++)
        if(sample->points[i].y == 1) size_pos++;
        else                         size_neg++;

    avg_pos = (double*) malloc(sample->dim*sizeof(double));
    if(avg_pos == NULL) { printf("Error: Out of memory 11\n"); exit(1); }
    avg_neg = (double*) malloc(sample->dim*sizeof(double));
    if(avg_neg == NULL) { printf("Error: Out of memory 12\n"); exit(1); }

    for(j = 0; j < sample->dim; ++j)
    {
        avg_pos[j] = 0.0;
        avg_neg[j] = 0.0;
        for(i = 0; i < sample->size; ++i)
            if(sample->points[i].y == 1)
                avg_pos[j] += sample->points[i].x[j];
            else
                avg_neg[j] += sample->points[i].x[j];
        avg_pos[j] /= (double)size_pos;
        avg_neg[j] /= (double)size_neg;
    }

    for(dist = 0.0, j = 0; j < sample->dim; ++j)
        if(index < 0 || sample->fnames[j] != index)
            dist += pow(avg_pos[j]-avg_neg[j], 2);

    for(j = 0; j < sample->dim; ++j)
        for(i = 0; i < fsize; i++)
            if(sample->fnames[j] == feats[i])
                dist -= pow(avg_pos[j]-avg_neg[j], 2);

    return sqrt(dist);
}

/*----------------------------------------------------------*
 * add a point into a sample                                *
 *----------------------------------------------------------*/
sample*
data_insert_point(sample *samp, sample *samp_in, int index)
{
    register int i = 0, j = 0;
    int dim        = samp->dim;
    sample *smout  = NULL;

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory 14\n"); exit(1); }

    /*allocating space for new array*/
    smout->size   = samp->size+1;
    smout->points = (point*) malloc(smout->size*sizeof(point));
    if(smout->points == NULL) { printf("Error: Out of memory 15\n"); exit(1); }

    /*allocating space for index array (if exists)*/
    if(samp->index)
    {
        smout->index = (int*) malloc(smout->size*sizeof(int));
        if(smout->index == NULL) { printf("Error: Out of memory 16\n"); exit(1); }
        for(i = 0; i < samp->size; ++i) smout->index[i] = samp->index[i];
        smout->index[i] = index;
    }
    else
        smout->index = NULL;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc(dim*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory 17\n"); exit(1); }

    /*copying fnames*/
    for(j = 0; j < dim; ++j)
        smout->fnames[j] = samp->fnames[j];

    /*copying data*/
    smout->dim          = samp->dim;
    smout->bias         = samp->bias;
    smout->q            = samp->q;
    smout->max_time     = samp->max_time;
    smout->mult_tempo   = samp->mult_tempo;
    smout->kernel_type  = samp->kernel_type;
    smout->kernel_param = samp->kernel_param;
    smout->alpha_aprox  = samp->alpha_aprox;
    smout->flexivel     = samp->flexivel;
    smout->p            = samp->p;
    smout->normalized   = samp->normalized;
    smout->norm         = samp->norm;
    smout->margin       = samp->margin;

    /*copying samp information to new data array*/
    for(i = 0; i < samp->size; ++i)
    {
        /*initializing*/
        smout->points[i].y     = samp->points[i].y;
        smout->points[i].alpha = samp->points[i].alpha;

        /*allocating space for vector x*/
        smout->points[i].x = (double*) malloc(dim*sizeof(double));
        if(smout->points[i].x == NULL) { printf("Error: Out of memory 18\n"); exit(1); }

        /* copying points */
        for(j = 0; j < dim; ++j)
            smout->points[i].x[j] = samp->points[i].x[j];
    }

    /*insert a new point into new data array*/
    smout->points[i].y     = samp_in->points[index].y;
    smout->points[i].alpha = samp_in->points[index].alpha;

    if(smout->index) smout->index[i] = i;

    /*allocating space for vector x*/
    smout->points[i].x = (double*) malloc(dim*sizeof(double));
    if(smout->points[i].x == NULL) { printf("Error: Out of memory 19\n"); exit(1); }

    /* copying features */
    for(j = 0; j < dim; ++j)
        smout->points[i].x[j] = samp_in->points[index].x[j];

    data_free_sample(&samp);

    return smout;
}

/*----------------------------------------------------------*
 * copy sample with one less data point                     *
 *----------------------------------------------------------*/
sample*
data_remove_point(sample *samp, int index)
{
    register int i = 0, j = 0;
    int dim       = samp->dim;
    int size      = samp->size;
    int offset    = 0;
    sample *smout = NULL;

    /*error check*/
    if(size == 1) { printf("Error: RemovePoint, only one point left\n"); return NULL; }

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory 20\n"); exit(1); }

    /*allocating space for new array*/
    smout->size   = size-1;
    smout->dim    = dim;
    smout->points = (point*) malloc(size*sizeof(point));
    if(smout->points == NULL) { printf("Error: Out of memory 21\n"); exit(1); }

    /*allocating space for index array (if exists)*/
    if(samp->index)
    {
        smout->index = (int*) malloc(smout->size*sizeof(int));
        if(smout->index == NULL) { printf("Error: Out of memory 22\n"); exit(1); }
        //for(i = 0; i < smout->size; ++i) smout->index[i] = samp->index[i];
    }
    else
        smout->index = NULL;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc(dim*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory 23\n"); exit(1); }

    /*copying data*/
    smout->bias         = samp->bias;
    smout->q            = samp->q;
    smout->max_time     = samp->max_time;
    smout->mult_tempo   = samp->mult_tempo;
    smout->kernel_type  = samp->kernel_type;
    smout->kernel_param = samp->kernel_param;
    smout->alpha_aprox  = samp->alpha_aprox;
    smout->flexivel     = samp->flexivel;
    smout->p            = samp->p;
    smout->normalized   = samp->normalized;
    smout->norm         = samp->norm;
    smout->margin       = samp->margin;

    /*copying information to new data array*/
    for(i = 0; i < size; ++i)
    {
        if(i == index)
        {
            offset = 1;
            continue;
        }

        /*initializing*/
        smout->points[i-offset].y     = samp->points[i].y;
        smout->points[i-offset].alpha = samp->points[i].alpha;
        if(smout->index)
        {
            smout->index[i-offset] = samp->index[i];
            if(i >= index) smout->index[i-offset]--;
        }

        /*allocating space for vector x*/
        smout->points[i-offset].x = (double*) malloc(dim*sizeof(double));
        if(smout->points[i-offset].x == NULL) { printf("Error: Out of memory 24\n"); exit(1); }

        /* copying features */
        for(j = 0; j < dim; ++j)
        {
            smout->points[i-offset].x[j] = samp->points[i].x[j];
            if(i == 0) smout->fnames[j]  = samp->fnames[j];
        }
    }
    return smout;
}

/*----------------------------------------------------------*
 * Returns data array without features in array             *
 *----------------------------------------------------------*/
sample*
data_remove_features(sample *samp, int *rem_feat, int fsize, int verbose)
{
    register int i = 0, j = 0;
    int s         = 0;
    int dim       = samp->dim;
    int size      = samp->size;
    int saveflag  = 0;
    int offset    = 0;
    sample *smout = NULL;

    /*sorting data*/
    qsort(rem_feat, fsize, sizeof(int), data_compare_int_greater);

    if(verbose>1)
    {
        for(i = 0; i < fsize; ++i) printf("--> %d\n", rem_feat[i]);
        printf("-------------------------------------\n");
    }

    /*error check*/
    if(fsize >= dim) { printf("Error: RemoveFeature, fsize(%d)>=dim(%d)\n", fsize, dim); return NULL; }

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory 25\n"); exit(1); }

    /*allocating space for new array*/
    smout->size   = size;
    smout->points = (point*) malloc(size*sizeof(point));
    if(smout->points == NULL) { printf("Error: Out of memory 26\n"); exit(1); }

    /*allocating space for index array (if exists)*/
    if(samp->index)
    {
        smout->index = (int*) malloc(size*sizeof(int));
        if(smout->index == NULL) { printf("Error: Out of memory 27\n"); exit(1); }
        for(i = 0; i < size; ++i) smout->index[i] = samp->index[i];
    }
    else
        smout->index = NULL;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc((dim-fsize)*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory 28\n"); exit(1); }

    /*copying data*/
    smout->bias         = samp->bias;
    smout->q            = samp->q;
    smout->max_time     = samp->max_time;
    smout->mult_tempo   = samp->mult_tempo;
    smout->kernel_type  = samp->kernel_type;
    smout->kernel_param = samp->kernel_param;
    smout->alpha_aprox  = samp->alpha_aprox;
    smout->flexivel     = samp->flexivel;
    smout->p            = samp->p;
    smout->normalized   = samp->normalized;
    smout->norm         = samp->norm;
    smout->margin       = samp->margin;

    /*copying information to new data array*/
    for(i = 0; i < size; ++i)
    {
        /*initializing*/
        smout->points[i].y     = samp->points[i].y;
        smout->points[i].alpha = samp->points[i].alpha;

        /*allocating space for vector x*/
        smout->points[i].x = (double*) malloc((dim-fsize)*sizeof(double));
        if(smout->points[i].x == NULL) { printf("Error: Out of memory 29\n"); exit(1); }

        /* copying features */
        s = 0; offset = 0;
        for(j = 0; j < dim; ++j)
        {
            saveflag = 1;
            if(offset < fsize && samp->fnames[j] == rem_feat[offset])
            {
                saveflag = 0;
                offset++;
            }
            if(saveflag)
            {
                smout->points[i].x[s] = samp->points[i].x[j];
                smout->fnames[s]      = samp->fnames[j];
                s++;
            }
        }
        /*error check*/
        if(s != (dim - fsize))
        {
            printf("Error: Something went wrong on RemoveFeature\n");
            printf("s = %d, dim = %d, fsize = %d\n", s, dim, fsize);
            exit(1);
        }
    }
    /*setting up dimension*/
    smout->dim = dim-fsize;

    return smout;
}

/*----------------------------------------------------------*
 * Returns data array with only features in array           *
 *----------------------------------------------------------*/
sample*
data_insert_features(sample *samp, int *ins_feat, int fsize, int verbose)
{
    register int i = 0, j = 0;
    int s         = 0;
    int dim       = samp->dim;
    int size      = samp->size;
    int saveflag  = 0;
    int offset    = 0;
    sample *smout = NULL;

    /*sorting data*/
    qsort(ins_feat, fsize, sizeof(int), data_compare_int_greater);

    if(verbose>1)
    {
        for(i = 0; i < fsize; ++i) printf("--> %d\n", ins_feat[i]);
        printf("-------------------------------------\n");
    }

    /*error check*/
    if(fsize > dim) { printf("Error: InsertFeature, fsize(%d)>dim(%d)\n", fsize, dim); return NULL; }

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory 30\n"); exit(1); }

    /*allocating space for new array*/
    smout->size   = size;
    smout->points = (point*) malloc(size*sizeof(point));
    if(smout->points == NULL) { printf("Error: Out of memory 31\n"); exit(1); }

    /*allocating space for index array (if exists)*/
    if(samp->index)
    {
        smout->index = (int*) malloc(size*sizeof(int));
        if(smout->index == NULL) { printf("Error: Out of memory 32\n"); exit(1); }
        for(i = 0; i < size; ++i) smout->index[i] = samp->index[i];
    }
    else
        smout->index = NULL;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc(fsize*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory 33\n"); exit(1); }

    /*copying data*/
    smout->bias         = samp->bias;
    smout->q            = samp->q;
    smout->max_time     = samp->max_time;
    smout->mult_tempo   = samp->mult_tempo;
    smout->kernel_type  = samp->kernel_type;
    smout->kernel_param = samp->kernel_param;
    smout->alpha_aprox  = samp->alpha_aprox;
    smout->flexivel     = samp->flexivel;
    smout->p            = samp->p;
    smout->normalized   = samp->normalized;
    smout->norm         = samp->norm;
    smout->margin       = samp->margin;

    /*copying information to new data array*/
    for(i = 0; i < size; ++i)
    {
        /*initializing*/
        smout->points[i].y     = samp->points[i].y;
        smout->points[i].alpha = samp->points[i].alpha;

        /*allocating space for vector x*/
        smout->points[i].x = (double*) malloc(fsize*sizeof(double));
        if(smout->points[i].x == NULL) { printf("Error: Out of memory 34\n"); exit(1); }

        /* copying features */
        s = 0; offset = 0;
        for(j = 0; j < dim; ++j)
        {
            saveflag = 0;
            if(offset < fsize && samp->fnames[j] == ins_feat[offset])
            {
                saveflag = 1;
                offset++;
            }
            if(saveflag)
            {
                smout->points[i].x[s] = samp->points[i].x[j];
                smout->fnames[s]      = samp->fnames[j];
                s++;
            }
        }
        /*error check*/
        if(s != fsize)
        {
            printf("Error: Something went wrong on InsertFeature\n");
            printf("s = %d, dim = %d, fsize = %d\n", s, dim, fsize);
            exit(1);
        }
    }
    /*setting up dimension*/
    smout->dim = fsize;

    return smout;
}

/*----------------------------------------------------------*
 * Returns sample copied                                    *
 *----------------------------------------------------------*/
sample*
data_copy_sample(sample *samp)
{
    register int i = 0, j = 0;
    int dim       = samp->dim;
    int size      = samp->size;
    sample *smout = NULL;

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory 35\n"); exit(1); }

    /*copying data*/
    smout->size         = samp->size;
    smout->dim          = samp->dim;
    smout->bias         = samp->bias;
    smout->q            = samp->q;
    smout->max_time     = samp->max_time;
    smout->mult_tempo   = samp->mult_tempo;
    smout->kernel_type  = samp->kernel_type;
    smout->kernel_param = samp->kernel_param;
    smout->alpha_aprox  = samp->alpha_aprox;
    smout->flexivel     = samp->flexivel;
    smout->p            = samp->p;
    smout->normalized   = samp->normalized;
    smout->norm         = samp->norm;
    smout->margin       = samp->margin;

    /*allocating space for new array*/
    smout->points = (point*) malloc(size*sizeof(point));
    if(smout->points == NULL) { printf("Error: Out of memory 36\n"); exit(1); }

    /*allocating space for index array (if exists)*/
    if(samp->index)
    {
        smout->index = (int*) malloc(size*sizeof(int));
        if(smout->index == NULL) { printf("Error: Out of memory 37\n"); exit(1); }
        for(i = 0; i < size; ++i) smout->index[i] = samp->index[i];
    }
    else
        smout->index = NULL;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc(dim*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory 38\n"); exit(1); }

    /*copying information to new data array*/
    for(i = 0; i < size; ++i)
    {
        /*initializing*/
        smout->points[i].y     = samp->points[i].y;
        smout->points[i].alpha = samp->points[i].alpha;

        /*allocating space for vector x*/
        smout->points[i].x = (double*) malloc(dim*sizeof(double));
        if(smout->points[i].x == NULL) { printf("Error: Out of memory 39\n"); exit(1); }

        /* copying features */
        for(j = 0; j < dim; ++j)
        {
            smout->points[i].x[j] = samp->points[i].x[j];
            smout->fnames[j]      = samp->fnames[j];
        }
    }
    return smout;
}

/*----------------------------------------------------------*
 * Returns sample copied without points                     *
 *----------------------------------------------------------*/
sample*
data_copy_sample_zero(sample *samp)
{
    int j = 0;
    sample *smout = NULL;

    /*allocating space for new sample*/
    smout = (sample*) malloc(sizeof(sample));
    if(smout == NULL) { printf("Error: Out of memory 40\n"); exit(1); }

    /*copying data*/
    smout->fnames       = NULL;
    smout->points       = NULL;
    smout->index        = NULL;
    smout->size         = 0;
    smout->dim          = samp->dim;
    smout->bias         = samp->bias;
    smout->q            = samp->q;
    smout->max_time     = samp->max_time;
    smout->mult_tempo   = samp->mult_tempo;
    smout->kernel_type  = samp->kernel_type;
    smout->kernel_param = samp->kernel_param;
    smout->alpha_aprox  = samp->alpha_aprox;
    smout->flexivel     = samp->flexivel;
    smout->p            = samp->p;
    smout->normalized   = samp->normalized;
    smout->norm         = samp->norm;
    smout->margin       = samp->margin;

    /*allocating space for new fnames*/
    smout->fnames = (int*) malloc(samp->dim*sizeof(int));
    if(smout->fnames == NULL) { printf("Error: Out of memory 41\n"); exit(1); }

    /* copying fnames */
    for(j = 0; j < samp->dim; ++j)
        smout->fnames[j] = samp->fnames[j];

    return smout;
}

/*----------------------------------------------------------*
 * Divide sample into train and test                        *
 *----------------------------------------------------------*/
void
data_part_train_test(sample **train_sample, sample **test_sample, int fold, int seed, int verbose)
{
	register int i = 0, j = 0;
	int qtdpos = 0, qtdneg = 0;
    struct sample *sample_pos = NULL, *sample_neg = NULL;

    sample_pos = data_copy_sample_zero(*train_sample);
    sample_neg = data_copy_sample_zero(*train_sample);

    for(i = 0; i < (*train_sample)->size; i++)
        if((*train_sample)->points[i].y == 1)
            sample_pos = data_insert_point(sample_pos, *train_sample, i);
        else
            sample_neg = data_insert_point(sample_neg, *train_sample, i);

    qtdpos = sample_pos->size;
    qtdneg = sample_neg->size;

    if(verbose)
    {
        printf("Total de pontos: %d\n", (*train_sample)->size);
        printf("Qtde Pos.: %d\n", qtdpos);
        printf("Qtde Neg.: %d\n\n", qtdneg);
    }

    srand(seed);

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

    data_free_sample(train_sample);
    data_free_sample(test_sample);

    (*train_sample) = data_copy_sample_zero(sample_pos);
    (*test_sample)  = data_copy_sample_zero(sample_pos);

    for(j = 0; j < sample_pos->size*(fold-1)/fold; j++)
        (*train_sample) = data_insert_point((*train_sample), sample_pos, j);
    for(; j < sample_pos->size; j++)
        (*test_sample) = data_insert_point((*test_sample), sample_pos, j);

    for(j = 0; j < sample_neg->size/fold; j++)
        (*test_sample) = data_insert_point((*test_sample), sample_neg, j);
    for(; j < sample_neg->size; j++)
        (*train_sample) = data_insert_point((*train_sample), sample_neg, j);

    data_free_sample(&sample_pos);
    data_free_sample(&sample_neg);

    /*if(verbose)
    {
        printf("Pontos de treino: %d\n", (*train_sample)->size);
        printf("Pontos de teste:  %d\n\n", (*test_sample)->size);
    }*/
}

/*----------------------------------------------------------*
 * Remove features from test sample                         *
 *----------------------------------------------------------*/
void
data_remove_test_sample_features(sample *train_sample, sample **test_sample, int verbose)
{
    register int i = 0, j = 0, k = 0;
    int totalfeat = (*test_sample)->dim-train_sample->dim;
    if((*test_sample)->dim > train_sample->dim)
    {
        int *feats = (int*) malloc(totalfeat*sizeof(int));
        if(feats == NULL) { printf("Erro de alocacao.\n"); exit(1); }
        for(k = 0, j = 0, i = 0; i < (*test_sample)->dim; i++)
        {
            if((*test_sample)->fnames[i] != train_sample->fnames[j])
                feats[k++] = (*test_sample)->fnames[i];
            else if (j < train_sample->dim-1)
                j++;
        }
        if(totalfeat != k)
            printf("Erro na remocao: era pra remover %d, mas removeu %d.\n", totalfeat, k);
        struct sample *sample_temp = data_remove_features((*test_sample), feats, k, 0);
        data_free_sample(test_sample);
        (*test_sample) = sample_temp;
        sample_temp = NULL;
        if(verbose)
        {
            printf("Remocao:\n");
            for(i = 0; i < k; i++)
            {
                printf("%4d ", feats[i]);
                if((i+1) % 20 == 0)
                    printf("\n");
            }
            printf("\n");
        }
        free(feats);
    }
}

/*----------------------------------------------------------*
 * Returns standard deviation of a vector.                  *
 *----------------------------------------------------------*/
double
data_standard_deviation(double *vet, int size)
{
    if(size == 1) return 0.0;
    int i;
    double avg, sd;

    for(avg = 0.0, i = 0; i < size; ++i)
        avg += vet[i];
    avg /= size;

    for(sd = 0.0, i = 0; i < size; ++i)
        sd += (vet[i]-avg)*(vet[i]-avg);

    return sqrt(sd/(size-1));
}

/*----------------------------------------------------------*
 * Returns 1 for a > b, -1 a < b, 0 if a = b                *
 *----------------------------------------------------------*/
int
data_compare_int_greater (const void *a, const void *b)
{
    const int *ia = (const int*) a;
    const int *ib = (const int*) b;

    /*          V (greater)*/
    return (*ia > *ib) - (*ia < *ib);
}
