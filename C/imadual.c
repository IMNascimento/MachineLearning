/*********************************************************************************
 * imadual.c: IMA algorithm in dual variables
 *
 * Saul Leite <lsaul@lncc.br>
 * Saulo Moraes <saulomv@gmail.com>
 * Copyright (C) 2006/2011
 *
 *********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "data.h"
#include "kernel.h"
#include "utils.h"
#include "imadual.h"

#define MIN_INC  1.001
#define RATE     1
#define MAX_IT   1E9
#define MAX_UP   1E9
#define EPS      1E-2

double START_TIME;
const double sqrate  = RATE*RATE;
const double tworate = 2*RATE;
extern double kernel_param;
extern int kernel_type;

int flagNaoPrimDim = 0;
long int updMax = 0;

int
imadual(sample *sample, double **w, double *margin, int *svs, int verbose)
{
    register int it;
    int ctot, passes;
    //int *index = sample->index;
    double rmargin = 0, gamma, secs;
    //double rmargin_ant = rmargin;
    datadual data;
    double *w_saved = NULL;
    double bias = 0;
    double max_time;
   // double time_ant = 0;

    register int i;
    int sv = 0, size = sample->size, dim = sample->dim;
    double min, max, norm = 0, *saved_alphas = NULL, *func;
    point *points = sample->points;

    max_time     = sample->max_time;
    kernel_type  = sample->kernel_type;
    kernel_param = sample->kernel_param;

    START_TIME = 100.0f*clock()/CLOCKS_PER_SEC;

    //Initializing data struct
    data.norm = 0.0;
    data.func = NULL;
    data.z    = sample;

    if(*margin > 0)
    {
        flagNaoPrimDim = 1;
        double raio = data_get_radius(sample, -1, 2);
        updMax = (raio*raio - rmargin*rmargin) / pow(*margin - rmargin, 2);
        if(rmargin == 0) updMax *= 1.5;
        //printf("updMax = %ld\n", updMax);
    }
    else
    {
        //Initializing alpha and bias
        for(i = 0; i < size; ++i) points[i].alpha = 0.0;
        sample->bias = 0;
        updMax = MAX_UP;
    }

    //Allocating space for index
    if(!sample->index)
    {
        sample->index = (int*) malloc(sizeof(int)*size);
        if(!sample->index) { printf("Error: Out of memory\n"); return -1; }
        for(i = 0; i < size; ++i) sample->index[i] = i;
    }

    //Allocating space for func
    data.func = (double*) malloc(sizeof(double)*size);
    if(!data.func) { printf("Error: Out of memory\n"); return -1; }

    //Allocating space for saved alphas
    saved_alphas = (double*) malloc(sizeof(double)*size);
    if(!saved_alphas) { printf("Error: Out of memory\n"); return -1; }

    //Initializing func
    for(i = 0; i < size; ++i) data.func[i] = 0.0;
    func = data.func;

    //Allocating space kernel matrix
    data.K = kernel_generate_matrix(sample);
    //kernel_print_matrix(data.K, size);

    if(verbose)
    {
        printf("-------------------------------------------------------------------\n");
        printf("  passos    atualiz.        margem          norma      svs     segs\n");
        printf("-------------------------------------------------------------------\n");
    }

    it = 0; ctot = 0; passes = 0; gamma = 0;
    //srand(0); //zerar a semente pros resultados serem sempre iguais
    //time_ant = 100.0f*clock()/CLOCKS_PER_SEC;
    while(imadual_fixed_margin_perceptron(&data, gamma, &passes, &ctot, sample->index, max_time))
    {   //Finding minimum and maximum functional values
        norm = data.norm;
        bias = data.z->bias;
        for(sv = 0, min = DBL_MAX, max = -DBL_MAX, i = 0; i < size; ++i)
        {
	        if(points[i].alpha > EPS*RATE) { sv++; saved_alphas[i] = points[i].alpha; }
            else                           { saved_alphas[i] = 0.0; }
                 if(func[i] >= 0 && min > func[i]/norm) min = func[i]/norm;
            else if(func[i] <  0 && max < func[i]/norm) max = func[i]/norm;
        }

        //Obtaining real margin
        rmargin = (fabs(min) > fabs(max)) ? fabs(max) : fabs(min);

        //Shift no bias
        double mmargin = (fabs(max) + fabs(min)) / 2.0;
        if(fabs(max) > fabs(min))
            data.z->bias += fabs(mmargin - rmargin);
        else
            data.z->bias -= fabs(mmargin - rmargin);

        //Obtaining new gamma_f
        gamma = (min-max)/2.0;
        if(gamma < MIN_INC*rmargin) gamma = MIN_INC*rmargin;
        rmargin = mmargin;

        secs = (100.0f*clock()/CLOCKS_PER_SEC-START_TIME)/100.0f;
        if(verbose) printf(" %7d    %8d    %12.6lf    %12.3f   %3d %8.2lf\n", passes, ctot, rmargin, norm, sv, secs);

        //parar se a margem comecar a estabilizar e o tempo entre as iteracoes nao for muito pequeno
        //if(!(rmargin > rmargin_ant*MIN_INC) && secs >= 2 && 100.0f*clock()/CLOCKS_PER_SEC > time_ant+0.1) break;
        //time_ant = 100.0f*clock()/CLOCKS_PER_SEC;

        ++it; //IMA iteration increment
        if(flagNaoPrimDim) break;
    }

    for(i = 0; i < size; ++i) points[i].alpha = saved_alphas[i];

    sample->bias = bias;
    norm = kernel_feature_space_norm(sample, data.K);

    /*recuperando o vetor DJ -- "pesos" das componentes*/
    if(kernel_type == 0)
        w_saved = utils_get_weight(sample);
    else
    {
        if(kernel_type == 1 && kernel_param == 1)
            w_saved = utils_get_dualweight_prodint(sample);
        else
            w_saved = utils_get_dualweight(sample);
        if(it) data_normalized(w_saved, dim, 2);
    }

    *w      = w_saved;
    *margin = rmargin;
    *svs    = sv;

    if(verbose)
    {
        printf("-------------------------------------------------------------------\n");
        printf("Numero de vezes que o Perceptron de Margem Fixa foi chamado: %d\n", it+1);
        printf("Numero de passos atraves dos dados: %d\n", passes);
        printf("Numero de atualizacoes: %d\n", ctot);
        printf("Numero de Vetores Suporte: %d\n", sv);
        printf("Margem encontrada: %lf\n\n", rmargin);
        if(verbose > 1)
        {
            for(i = 0; i < dim; i++)
                printf("W[%d]: %lf\n", sample->fnames[i], w_saved[i]);
            printf("Bias: %lf\n\n", sample->bias);
        }
    }

    //freeing stuff
    //free(index);
    kernel_free_matrix(data.K, size);
    free(data.func);
    free(saved_alphas);

    if(!it)
    {
        if(verbose) printf("Convergencia do FMP nao foi atingida!\n");
        return 0;
    }
    return 1;
}

int
imadual_fixed_margin_perceptron(datadual *data, double gamma, int *passes, int *ctot, int *index, double max_time)
{
    int t, c, e, i, j, k, s, idx, r, size = data->z->size;
    double y, lambda, norm = data->norm, time = START_TIME+max_time;
    double *func = data->func, *Kv = NULL;
    point *points = data->z->points;
    double bias = data->z->bias;
    double **K = data->K;

    t = (*passes); c = (*ctot); e = 1; s = 0;
    while(100.0f*clock()/CLOCKS_PER_SEC-time <= 0)
    {
        for(e = 0, i = 0; i < size; ++i)
        {
            idx = index[i];
            y = points[idx].y;

            //Checking if the point is a mistake
            if(y*func[idx] - gamma*norm <= 0)
            {
                lambda = (gamma) ? (1-RATE*gamma/norm) : 1;
                Kv     = K[idx];
		        norm  *= lambda;

                for(r = 0; r < size; ++r)
		        {
                    points[r].alpha *= lambda;
                    func[r]          = lambda * func[r] + RATE*y*(Kv[r]+1) + bias*(1-lambda);
                }

                norm = sqrt(norm*norm + tworate*points[idx].y*lambda*(func[idx]-bias) + sqrate*Kv[idx]);
                points[idx].alpha += RATE;
                bias += RATE * y;

                k = (i > s) ? ++s : e;
                j = index[k];
                index[k] = idx;
                index[i] = j;
                ++c; ++e;
            }
            else if(t > 0 && e > 1 && i > s) break;
        }
        ++t; //Number of iterations update

        //stop criterion
        if(e == 0)     break;
        if(t > MAX_IT) break;
        if(c > MAX_UP) break;
        if(flagNaoPrimDim) if(c > updMax) break;
    }
    data->z->bias = bias;
    data->norm    = norm;
    (*passes)  = t;
    (*ctot  )  = c;
    if(e == 0) return 1;
    else       return 0;
}
