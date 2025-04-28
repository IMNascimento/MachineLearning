/*********************************************************************************
 * almap.c: ALMA: Approximate Large Margin Algorithm
 * Claudio Gentile. A new approximate maximal margin classification algorithm.
 * JMLR, 2:213–242, 2001.
 *
 * Saulo Moraes Villela <saulomv@gmail.com>
 * Copyright (C) 2013
 *
 *********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <sys/time.h>
#include "data.h"
#include "utils.h"
#include "imap.h"
#include "almap.h"

#define MAX_IT   1E9

double START_TIME;

int
almap(sample *sample, double **w, double *margin, int *svs, int verbose)
{
    double gamma, secs;
    double *w_saved = NULL;
    double bias = 0;
    int *index;
    point *points;
    data data;
    double rmargin = *margin;

    if(!sample->normalized)
        sample = data_normalize_database(sample);

    register int i, j, t, c, e, k;//, r;
    int size = sample->size, dim = sample->dim;
    double min = 0, max = 0, norm = 0, *func;

    int idx;
    double y, *x = NULL;
    register double sumnorm = 0;
    double sign = 1;

    double p          = sample->p;
    double q          = p/(p-1.0);
    sample->q         = q;
    double alpha_prox = sample->alpha_aprox;

    double RATE = 1.0;
    double B    = 1.0/alpha_prox;
    double C    = sqrt(2.0);

    //Initializing data struct
    data.norm      = 0.0;
    data.func      = NULL;
    data.w         = NULL;
    data.z         = sample;

    //Allocating space for w
    if(!(*w))
    {
        data.w = (double*) malloc(dim*sizeof(double));
        if(!data.w) { printf("Error: Out of memory\n"); return -1; }
        for(i = 0; i < dim; ++i) data.w[i] = 0.0;
    }
    else
    {
        data.w = *w;
        data.norm = 0;
        for(i = 0; i < dim; ++i) data.norm += pow(fabs(data.w[i]), q);
        data.norm = pow(data.norm, 1.0/q);
        for(i = 0; i < dim; ++i) data.w[i] /= data.norm;
        data.norm = 1;
    }

    //Allocating space for index and initializing
    if(!sample->index)
    {
        sample->index = (int*) malloc(size*sizeof(int));
        if(!sample->index) { printf("Error: Out of memory\n"); return -1; }
        for(i = 0; i < size; ++i) sample->index[i] = i;
    }
    sample->bias = 0;

    //Allocating space for w_saved and func
    w_saved = (double*) malloc(dim*sizeof(double));
    if(!w_saved) { printf("Error: Out of memory\n"); return -1; }
    data.func = (double*) malloc(size*sizeof(double));
    if(!data.func) { printf("Error: Out of memory\n"); return -1; }

    //Initializing w_saved and func
    for(i = 0; i <  dim; ++i) w_saved[i]   = 0.0;
    for(i = 0; i < size; ++i) data.func[i] = 0.0;

    func   = data.func;
    norm   = data.norm;
    bias   = data.z->bias = 0;
    points = data.z->points;
    index  = data.z->index;
    if(verbose)
    {
        printf("---------------------------------------------------------------\n");
        printf(" passos     atualiz.        margem            norma        segs\n");
        printf("---------------------------------------------------------------\n");
    }

    t = 0; c = 0; k = 1;
    START_TIME = 100.0f*clock()/CLOCKS_PER_SEC;
    while(t < MAX_IT)
    {
        for(e = 0, i = 0; i < size; ++i)
        {
            //shuffling data r = i + rand()%(size-i); j = index[i]; idx = index[i] = index[r]; index[r] = j;

            idx = index[i];
            x = points[idx].x;
            y = points[idx].y;

            gamma = B * sqrt(p-1.0) * (1.0/sqrt(k));

            //calculating function
            for(func[idx] = bias, j = 0; j < dim; ++j)
                func[idx] += data.w[j] * x[j];

            //Checking if the point is a mistake
            if(y*func[idx] <= (1.0-alpha_prox)*gamma)
            {
                RATE = C / sqrt(p-1.0) * (1.0/sqrt(k));
                for(sumnorm = 0, j = 0; j < dim; ++j)
                {
                    sign = (data.w[j] >= 0) ? 1.0 : -1.0;
                    data.w[j] = sign * pow(fabs(data.w[j]), q-1.0) / pow(norm, q-2.0);
                    data.w[j] += RATE * y * x[j];
                    sumnorm += pow(fabs(data.w[j]), p);
                }
                norm = pow(sumnorm, 1.0/p);
                //printf("p = %lf -- Norm p: %lf\n", p, norm);

                for(sumnorm = 0, j = 0; j < dim; ++j)
                {
                    sign = (data.w[j] >= 0) ? 1.0 : -1.0;
                    data.w[j] = sign * pow(fabs(data.w[j]), p-1.0) / pow(norm, p-2.0);
                    sumnorm += pow(fabs(data.w[j]), q);
                }
                //bias += RATE * y;
                norm = pow(sumnorm, 1.0/q);
                //printf("q = %lf -- Norm q: %lf\n", q, norm);
                if(norm > 1)
                {
                    for(j = 0; j < dim; ++j)
                        data.w[j] /= norm;
                    //bias /= norm;
                }
                c++; e++; k++;
            }
        }
        t++; //Number of iterations update
        //printf("%7d    %8d    %12.6lf    %12.3lf    %8.3lf - error: %d\n", t, c, gamma, norm, (100.0f*clock()/CLOCKS_PER_SEC-START_TIME)/100.0f, e);
        //stop criterion
        if(e == 0 && norm > 0)
        {
            //Finding minimum and maximum functional values
            for(min = DBL_MAX, max = -DBL_MAX, i = 0; i < size; ++i)
            {
                     if(func[i] >= 0 && min > func[i]/norm) min = func[i]/norm;
                else if(func[i] <  0 && max < func[i]/norm) max = func[i]/norm;
            }
            //Saving good weights
            for(i = 0; i < dim; i++) w_saved[i] = data.w[i];

            //Obtaining real margin
            rmargin = (fabs(min) > fabs(max)) ? fabs(max) : fabs(min);

            secs = (100.0f*clock()/CLOCKS_PER_SEC-START_TIME)/100.0f;
            if(verbose) printf("%7d    %8d    %12.6lf    %12.3f    %8.3lf\n", t, c, rmargin, norm, secs);
            break;
        }
    }

    *w = w_saved;
    *margin = rmargin;
    sample->margin = rmargin;
    sample->norm = norm;
    sample->bias = bias;

    if(verbose)
    {
        printf("---------------------------------------------------------------\n");
        printf("Numero de passos atraves dos dados: %d\n", t);
        printf("Numero de atualizacoes: %d\n", c);
        printf("Margem encontrada: %lf\n\n", rmargin);
        printf("Min: %lf / Max: %lf\n", fabs(min), fabs(max));
        if(verbose > 1)
        {
            for(i = 0; i < dim; ++i)
                printf("W[%d]: %lf\n", sample->fnames[i], w_saved[i]);
            printf("Bias: %lf\n\n", sample->bias);
        }
    }
    //Freeing stuff
    free(data.func);
    free(data.w);

    if(!t)
    {
        if(verbose) printf("Convergencia do ALMAp nao foi atingida!\n");
        return 0;
    }
    return 1;
}
