/*********************************************************************************
 * imap.c: IMA algorithm in Primal variables with Norm Lp
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
#include <sys/time.h>
#include "data.h"
#include "kernel.h"
#include "utils.h"
#include "imap.h"

#define MIN_INC  1.001
//#define RATE     1
#define MAX_IT   1E9
#define MAX_UP   1E9
#define EPS      1E-9

double START_TIME;
double RATE = 1.0;
double maiorw = 0.0;
int n;
int maiorn = 0;
int flagNao1aDim = 0;
long int tMax = 0;

int
imap(sample *sample, double **w, double *margin, int *svs, int verbose)
{
    int it, ctot, passes;
    double gamma, secs;
    double *w_saved = NULL;
    double bias = 0.0;
    //int *index = sample->index;
    data data;
    double rmargin = *margin;
    double alpha;
    int y;

    register int i, j;
    int size = sample->size, dim = sample->dim;
    double min = 0.0, max = 0.0, norm = 1.0, *func;

    double q        = sample->q; //sample->q = q;
    double max_time = sample->max_time;
    //printf("Dim: %d / Max time = %lf\n", dim, max_time);
    double flexivel = sample->flexivel;
    n = dim;

    int t1=1, t3=1;//, t2;
    RATE = 1.0;
    double inc;

    //Initializing data struct
    data.norm      = 1.0;
    data.func      = NULL;
    data.w         = NULL;
    data.z         = sample;

    //Allocating space for w_saved and func
    w_saved = (double*) malloc(dim*sizeof(double));
    if(!w_saved) { printf("Error: Out of memory\n"); return -1; }
    data.func = (double*) malloc(size*sizeof(double));
    if(!data.func) { printf("Error: Out of memory\n"); return -1; }
    func = data.func;

    //Allocating space for w
    if(!(*w))
    {
        data.w = (double*) malloc(dim*sizeof(double));
        if(!data.w) { printf("Error: Out of memory\n"); return -1; }
        for(i = 0; i < dim; ++i) data.w[i] = 0.0;
        sample->bias = 0.0;
        flagNao1aDim = 0;
    }
    else
    {
        data.w = *w;
        if(q == 1)
            for(data.norm = 0.0, i = 0; i < dim; ++i) data.norm += fabs(data.w[i]);
        else if(q == 2)
        {
            for(data.norm = 0.0, i = 0; i < dim; ++i) data.norm += data.w[i]*data.w[i];
            data.norm = sqrt(data.norm);
        }
        else
        {
            for(data.norm = 0.0, i = 0; i < dim; ++i) data.norm += pow(fabs(data.w[i]), q);
            data.norm = pow(data.norm, 1.0/q);
        }
        for(i = 0; i < dim; ++i) data.w[i] /= data.norm;
        data.z->bias /= data.norm;
        data.norm = 1;
        flagNao1aDim = 1;
        int flag = 0;
        for(min = DBL_MAX, max = -DBL_MAX, i = 0; i < size; ++i)
        {
            y = data.z->points[i].y;
            for(func[i] = 0, j = 0; j < dim; ++j)
                func[i] += data.w[j] * data.z->points[i].x[j];
                 if(y ==  1 && func[i] < min) min = func[i];
            else if(y == -1 && func[i] > max) max = func[i];
        }
        //printf("min = %lf\n", min);
        //printf("max = %lf\n", max);
        //printf("flag = %d\n", flag);
        data.z->bias = - (min + max) / 2.0;
        //printf("bias: %lf\n", data.z->bias);
        //flag = 0;
        for(min = DBL_MAX, max = -DBL_MAX, i = 0; i < size; ++i)
        {
            y = data.z->points[i].y;
            for(func[i] = data.z->bias, j = 0; j < dim; ++j)
                func[i] += data.w[j] * data.z->points[i].x[j];
            if(func[i] * y < 0) flag++;
                 if(y ==  1 && func[i] < min) min = func[i];
            else if(y == -1 && func[i] > max) max = func[i];
        }
        //printf("flag = %d\n", flag);
        if(flag) rmargin = 0;
        else rmargin = fabs(min);
        //printf("*margin = %lf / rmargin = %lf\n", *margin, rmargin);
        if(*margin == 0) tMax = MAX_UP;
        else
        {
            double raio = data_get_radius(sample, -1, q);
            tMax = (raio*raio - rmargin*rmargin) / pow(*margin - rmargin, 2);
            if(rmargin == 0) tMax *= 1.5;
            //tMax *= 2;
        }
        //printf("tMax = %ld\n", tMax);
        //rmargin = 0;
        //tMax = (raio * raio - rmargin * rmargin) / pow(*margin - rmargin, 2);
        //printf("tMax = %d\n", tMax);
    }

    //Allocating space for index and initializing
    if(!sample->index)
    {
        sample->index = (int*) malloc(size*sizeof(int));
        if(!sample->index) { printf("Error: Out of memory\n"); return -1; }
        for(i = 0; i < size; ++i) sample->index[i] = i;
    }

    //Initializing w_saved and func
    for(i = 0; i <  dim; ++i) w_saved[i] = 0.0;
    for(i = 0; i < size; ++i) { data.func[i] = 0.0; data.z->points[i].alpha = 0.0; }

    if(verbose)
    {
        printf("----------------------------------------------------------------------\n");
        printf(" pmf    passos     atualiz.        margem            norma        segs\n");
        printf("----------------------------------------------------------------------\n");
    }

    it = 0; ctot = 0; passes = 0; gamma = 0.0; //gammaf1 = 0.0; t1 = 0; gammaf2 = 0.0; t2 = 0; gammaf3 = 0.0; t3 = 0; Cmedia = 0.0;
    //srand(0); //zerar a semente pros resultados serem sempre iguais
    START_TIME = 100.0f*clock()/CLOCKS_PER_SEC;
    while(imap_fixed_margin_perceptron(&data, gamma, &passes, &ctot, sample->index, q, max_time))
    {
        //Finding minimum and maximum functional values
        norm  = data.norm;
        bias  = data.z->bias;
        for(min = DBL_MAX, max = -DBL_MAX, i = 0; i < size; ++i)
        {
            y = data.z->points[i].y;
            alpha = data.z->points[i].alpha;
                 if((func[i] + y*alpha*flexivel) >= 0 && min > (func[i] + y*alpha*flexivel)/norm) min = (func[i] + y*alpha*flexivel)/norm;
            else if((func[i] + y*alpha*flexivel) <  0 && max < (func[i] + y*alpha*flexivel)/norm) max = (func[i] + y*alpha*flexivel)/norm;
        }

        //Saving good weights
        for(i = 0; i < dim; i++) w_saved[i] = data.w[i];

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
        inc = (1+sample->alpha_aprox)*rmargin;
        if(gamma < inc) gamma = inc;
        rmargin = mmargin;

        if(it == 2)
            t1 = ctot;
        t3 = ctot;

        if(it > 1)
        {
            RATE = sqrt(t1) / sqrt(t3);
            if(verbose) printf("RATE: %lf\n", RATE);
        }
        else if(it == 1 && verbose)
            printf("RATE: %lf\n", RATE);

        secs = (100.0f*clock()/CLOCKS_PER_SEC-START_TIME)/100.0f;
        if(verbose) printf(" %3d   %7d    %8d    %12.6lf    %12.3lf    %8.3lf ", it+1, passes, ctot, rmargin, norm, secs);

        ++it; //IMA iteration increment
        //if(it > 3)
        if(flagNao1aDim) break;
    }
    //printf("it IMA = %d\n", it);
    *w = w_saved;
    *margin = rmargin;
    sample->margin = rmargin;
    sample->norm = norm;
    sample->bias = bias;

    if(verbose)
    {
        printf("\n----------------------------------------------------------------------\n");
        printf("Numero de vezes que o Perceptron de Margem Fixa foi chamado: %d\n", it+1);
        printf("Numero de passos atraves dos dados: %d\n", passes);
        printf("Numero de atualizacoes: %d\n", ctot);
        printf("Margem encontrada: %lf\n", rmargin);
        printf("Min: %lf / Max: %lf\n\n", fabs(min), fabs(max));
        if(verbose > 1)
        {
            for(i = 0; i < dim; ++i) printf("W[%d]: %lf\n", sample->fnames[i], w_saved[i]);
            printf("Bias: %lf\n\n", sample->bias);
        }
    }
    //Freeing stuff //free(index);
    free(data.func);
    free(data.w);

    if(!it)
    {
        if(verbose) printf("Convergencia do FMP nao foi atingida!\n");
        return 0;
    }
    return 1;
}

int
imap_fixed_margin_perceptron(data *data, double gamma, int *passes, int *ctot, int *index, double q, double max_time)
{
    register int c, e, i, k, s, j;
    int t, idx, r, sign = 1;
    int size = data->z->size, dim = data->z->dim;
    double norm = data->norm, lambda = 1, y, time = START_TIME+max_time;
    double *func = data->func, *w = data->w;
    double *x = NULL;
    point *points = data->z->points;
    register double sumnorm = 0.0;
    double bias = data->z->bias;
    double maiorw_temp;
    int n_temp;
    double flexivel = data->z->flexivel;

    t = (*passes); c = (*ctot); e = 1; s = 0;
    while(100.0f*clock()/CLOCKS_PER_SEC-time <= 0)
    {
        for(e = 0, i = 0; i < size; ++i)
        {
            //shuffling data r = i + rand()%(size-i); j = index[i]; idx = index[i] = index[r]; index[r] = j;
            idx = index[i];
            x = points[idx].x;
            y = points[idx].y;

            //calculating function
            for(func[idx] = bias, j = 0; j < dim; ++j)
                func[idx] += w[j] * x[j];

            //Checking if the point is a mistake
            if(y*func[idx] <= gamma*norm - points[idx].alpha*flexivel)
            {
                lambda = (norm) ? (1-RATE*gamma/norm) : 1;
                for(r = 0; r < size; ++r)
                    points[r].alpha *= lambda;

                if(q == 1.0) //Linf
                {
                    for(sumnorm = 0, j = 0; j < dim; ++j)
                    {
                        sign = 1; if(w[j] < 0) sign = -1;
                        lambda = (norm > 0 && w[j] != 0) ? gamma * sign: 0;
                        w[j] += RATE * (y * x[j] - lambda);
                        sumnorm += fabs(w[j]);
                    }
                    norm = sumnorm;
                }
                else if(q == 2.0) //L2
                {
                    for(sumnorm = 0, j = 0; j < dim; ++j)
                    {
                        lambda = (norm > 0 && w[j] != 0) ? w[j] * gamma / norm : 0;
                        w[j] += RATE * (y * x[j] - lambda);
                        sumnorm += w[j] * w[j];
                    }
                    norm = sqrt(sumnorm);
                }
                else if(q == -1.0) //L1
                {
                    maiorw_temp = fabs(w[0]);
                    n_temp = 1;
                    for(j = 0; j < dim; ++j)
                    {
                        if(maiorw == 0 || fabs(maiorw - fabs(w[j]))/maiorw < EPS)
                        {
                            sign = 1; if(w[j] < 0) sign = -1;
                            lambda = (norm > 0 && w[j] != 0) ? gamma * sign / n : 0;
                            w[j] += RATE * (y * x[j] - lambda);
                        }
                        else
                            w[j] += RATE * (y * x[j]);

                        if(j > 0)
                        {
                            if(fabs(maiorw_temp - fabs(w[j]))/maiorw_temp < EPS)
                                n_temp++;
                            else if(fabs(w[j]) > maiorw_temp)
                            {
                                maiorw_temp = fabs(w[j]);
                                n_temp = 1;
                            }
                        }
                    }
                    maiorw = maiorw_temp;
                    n = n_temp;
                    norm = maiorw;
                    if(n > maiorn) maiorn = n;
                }
                else //outras formulações - Lp
                {
                    for(sumnorm = 0, j = 0; j < dim; ++j)
                    {
                        lambda = (norm > 0 && w[j] != 0) ? w[j] * gamma * pow(fabs(w[j]), q-2.0) * pow(norm, 1.0-q) : 0;
                        w[j] += RATE * (y * x[j] - lambda);
                        sumnorm += pow(fabs(w[j]), q);
                    }
                    norm = pow(sumnorm, 1.0/q);
                }
                bias += RATE * y;
                points[idx].alpha += RATE;

                k = (i > s) ? s++ : e;
                j = index[k];
                index[k] = idx;
                index[i] = j;
                c++; e++;
            }
            else if(t > 0 && e > 1 && i > s) break;
        }
        t++; //Number of iterations update

        //stop criterion
        if(e == 0)     break;
        if(t > MAX_IT) break;
        if(c > MAX_UP) break;
        //printf("t = %d\n", t);
        if(flagNao1aDim) if(c > tMax) break;
    }
    data->norm = norm;
    data->z->bias = bias;
    (*passes)  = t;
    (*ctot  )  = c;
    if(e == 0) return 1;
    else       return 0;
}
