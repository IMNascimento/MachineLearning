/*********************************************************************************
 * pl.c: Call to Linear Programming in Python
 *
 * Saulo Moraes Villela <saulo.moraes@ufjf.edu.br>
 * Copyright (C) 2015
 *
 *********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include "data.h"
#include "kernel.h"
#include "utils.h"
#include "pl.h"

int
linear_programming(sample *sample, double **w, double *margin, int *svs, int verbose)
{
    double *w_saved = NULL, bias, rmargin, norm;
    register int i;
    int dim = sample->dim;
    double q = sample->q;
    char fname[100] = "arquivo";
    *svs = 0;

    //Allocating space for w_saved
    w_saved = (double*) malloc(dim*sizeof(double));
    if(!w_saved) { printf("Error: Out of memory\n"); return -1; }

    if(q == -1)
    {
        if(!data_write("arquivol1.txt", sample, 0)) return -1;
        system("python main.py arquivol1.txt l1 > arquivol1.mod");
        system("glpsol -m arquivol1.mod --check --wlp arquivol1.lp > arquivol1.print");
        system("cplex -f arquivol1.cmd > arquivol1.print2");
        system("python parse_cplex_out.py > arquivol1.out");
        strcat(fname, "l1.out");
        //system("glpsol -m arquivol1.mod --dual > arquivol1.print");
        //strcat(fname, "l1.txt.l1.out");
    }
    else if(q == 1)
    {
        if(!data_write("arquivolinf.txt", sample, 0)) return -1;
        system("python main.py arquivolinf.txt linf > arquivolinf.mod");
        system("glpsol -m arquivolinf.mod --check --wlp arquivolinf.lp > arquivolinf.print");
        system("cplex -f arquivolinf.cmd > arquivolinf.print2");
        system("python parse_cplex_out.py > arquivolinf.out");
        strcat(fname, "linf.out");
        //system("glpsol -m arquivolinf.mod --dual > arquivolinf.print");
        //strcat(fname, "linf.txt.linf.out");
    }
    else
        return -1;

    FILE *file = fopen(fname, "r");
    if(!file) return -1;

    //if(fscanf(file, "infeasible") == 1) return 0;

    if(fscanf(file, "%lf", &norm) != 1) return 0;
    //if(fscanf(file, "%lf", &rmargin) != 1) return -1;
    rmargin = 1.0/norm;
    if(fscanf(file, "%lf", &bias) != 1) return 0;
    for(i = 0; i < dim; ++i) if(fscanf(file, "%lf", &w_saved[i]) != 1) return 0;
//    for(i = 0; i < dim; ++i)
//        printf("W[%d] = %.15lf\n", i, w_saved[i]);
//    printf("Bias: %.15lf\n", bias);
//    printf("Norma: %.15lf\n", norm);
//    printf("Margem: %.15lf\n", rmargin);

    *w = w_saved;
    *margin = rmargin;
    sample->margin = rmargin;
    sample->norm = norm;
    sample->bias = bias;

    return 1;
}
