/*********************************************************************************
 * kernel.c: Kernel functions lib                                                *
 *********************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "data.h"
#include "kernel.h"

double kernel_param; // = 1.0f;
int kernel_type; //  = 1;

/*----------------------------------------------------------*
 * Clears Kernel matrix                                     *
 *----------------------------------------------------------*/
void
kernel_free_matrix(double **matrix, int size)
{
    register int i;
    /*freeing data*/
    for(i = 0; i < size; ++i)
        free(matrix[i]);
    free(matrix);
    matrix = NULL;
}

/*----------------------------------------------------------*
 * Returns norm in the feature space                        *
 *----------------------------------------------------------*/
double
kernel_feature_space_norm(sample *sample, double** matrix)
{
    register int i = 0, j = 0;
    double sum1 = 0.0;
    double sum  = 0.0;

    for(i = 0; i < sample->size; ++i)
    {
        if(sample->points[i].alpha > 0)
        {
            sum1 = 0.0;
            for(j = 0; j < sample->size; ++j)
            {
                if(sample->points[j].alpha > 0)
                    sum1 += sample->points[j].y * sample->points[j].alpha * matrix[j][i];
            }
            sum += sample->points[i].alpha * sample->points[i].y * sum1;
        }
    }
    sum = sqrt(sum);

    return sum;
}

/*----------------------------------------------------------*
 * Generates Kernel matrix                                  *
 *----------------------------------------------------------*/
double**
kernel_generate_matrix(sample *sample)
{
    double **matrix = NULL;
    register int i = 0, j = 0;

    /* Allocating space for new matrix */
    matrix = (double**) malloc((sample->size)*sizeof(double*));
    if(matrix == NULL) { printf("Error: Out of memory\n"); exit(1); }

    for(i = 0; i < sample->size; ++i)
    {
        matrix[i] = (double*) malloc((sample->size)*sizeof(double));
        if(matrix[i] == NULL) { printf("Error: Out of memory\n"); exit(1); }
    }

    /* Calculating Matrix */
    for(i = 0; i < sample->size; ++i)
        for(j = i; j < sample->size; ++j)
        {
            matrix[i][j] = kernel_function((&sample->points[i]), &(sample->points[j]), sample->dim);
            matrix[j][i] = matrix[i][j];
        }
    return matrix;
}

/*----------------------------------------------------------*
 * Generates H matrix                                       *
 *----------------------------------------------------------*/
double**
kernel_generate_matrix_H(sample *sample)
{
    double **matrix = NULL;
    register int i = 0, j = 0;

    /* Allocating space for new matrix */
    matrix = (double**) malloc((sample->size)*sizeof(double*));
    if(matrix == NULL) { printf("Error: Out of memory\n"); exit(1); }
    for(i = 0; i < sample->size; ++i)
    {
        matrix[i] = (double*) malloc((sample->size)*sizeof(double));
        if(matrix[i] == NULL) { printf("Error: Out of memory\n"); exit(1); }
    }

    /* Calculating Matrix */
    for(i = 0; i < sample->size; ++i)
        for(j = i; j < sample->size; ++j)
        {
            matrix[i][j] = kernel_function((&sample->points[i]), &(sample->points[j]), sample->dim) * sample->points[i].y * sample->points[j].y;
            matrix[j][i] = matrix[i][j];
        }
    return matrix;
}

/*----------------------------------------------------------*
 * Generates H matrix without a dimenson                    *
 *----------------------------------------------------------*/
double**
kernel_generate_matrix_H_without_dim(sample *sample, int dim)
{
    double **matrix = NULL;
    register int i = 0, j = 0;

    /* Allocating space for new matrix */
    matrix = (double**) malloc((sample->size)*sizeof(double*));
    if(matrix == NULL) { printf("Error: Out of memory\n"); exit(1); }
    for(i = 0; i < sample->size; ++i)
    {
        matrix[i] = (double*) malloc((sample->size)*sizeof(double));
        if(matrix[i] == NULL) { printf("Error: Out of memory\n"); exit(1); }
    }

    /* Calculating Matrix */
    for(i = 0; i < sample->size; ++i)
        for(j = i; j < sample->size; ++j)
        {
            matrix[i][j] = kernel_function_without_dim((&sample->points[i]), &(sample->points[j]), dim, sample->dim) * sample->points[i].y * sample->points[j].y;
            matrix[j][i] = matrix[i][j];
        }
    return matrix;
}

/*----------------------------------------------------------*
 * Calculates kernel function between two points            *
 *----------------------------------------------------------*/
double
kernel_function(point *one, point *two, int dim)
{
    register int i = 0;
    register double t, sum = 0.0;
    register double *a = one->x-1, *b = two->x-1;

    switch(kernel_type)
    {
        case 0: //Produto Interno
            for(i = 0; i < dim; ++i)
                sum += (*(++a)) * (*(++b));
            break;
        case 1: //Polinomial
            for(i = 0; i < dim; ++i)
                sum += (*(++a)) * (*(++b));
            sum = (kernel_param > 1) ? pow(sum+1, kernel_param) : sum;
            break;

        case 2: //Gaussiano
            for(i = 0; i < dim; ++i)
            { t = (*(++a)) - (*(++b)); sum += t * t; }
	        sum = exp(-1 * sum * kernel_param);
            break;

        case 3: //Label
            if(one->y == two->y)
                sum = 1;
    }
    /*The '+1' here accounts for the bias term "b" in SVM formulation since
      <w,x> = \sum_i \alpha_i y_i k(x_i,x) + b and b=\sum_i \alpha_i y_i*/
    return sum;// + 1.0f;
}

/*------------------------------------------------------------------*
 * Calculates kernel function between two points without a dimenson *
 *------------------------------------------------------------------*/
double
kernel_function_without_dim(point *one, point *two, int j, int dim)
{
    register int i = 0;
    register double t, sum = 0.0;
    register double *a = one->x, *b = two->x;

    switch(kernel_type)
    {
        case 0: //Produto Interno
            for(i = 0; i < dim; ++i)
                if(i != j)
                    sum += a[i] * b[i];
            break;

        case 1: //Polinomial
            for(i = 0; i < dim; ++i)
                if(i != j)
                    sum += (*(a+i)) * (*(b+i));
            sum = (kernel_param > 1) ? pow(sum+1, kernel_param) : sum;
            break;

        case 2: //Gaussiano
            for(i = 0; i < dim; ++i)
                if(i != j)
                { t = (*(a+i)) - (*(b+i)); sum += t * t; }
	        sum = exp(-1 * sum * kernel_param);
            break;
    }
    /*The '+1' here accounts for the bias term "b" in SVM formulation since
      <w,x> = \sum_i \alpha_i y_i k(x_i,x) + b and b=\sum_i \alpha_i y_i*/
    return sum;// + 1.0f;
}

/*----------------------------------------------------------*
 * Prints matrix                                            *
 *----------------------------------------------------------*/
void
kernel_print_matrix(double **matrix, int size)
{
    register int i = 0, j = 0;
    for(i = 0; i < size; ++i)
    {
        for(j = 0; j < size; ++j)
            printf("%lf ", matrix[i][j]);
        printf("\n");
    }
}


/*----------------------------------------------------------*
 * Allocates a new matrix with one less sample point.       *
 *----------------------------------------------------------*/
double**
kernel_remove_element(double **o_matrix, int size, int num)
{
    double** matrix = NULL;
    register int i = 0, j = 0;
    int c = 0, d = 0;

    /* Allocating space for new matrix */
    matrix = (double**) malloc((size-1)*sizeof(double*));
    if(matrix == NULL) { printf("Error: Out of memory\n"); exit(1); }

    for(i = 0; i < size-1; ++i)
    {
        matrix[i] = (double*) malloc((size-1)*sizeof(double));
        if(matrix[i] == NULL) { printf("Error: Out of memory\n"); exit(1); }
    }

    /* Copying matrix */
    for(i = 0; i < size; ++i)
    {
        if(i != num)
        {
            d = c;
            for(j = i; j < size; ++j)
            {
                if(j != num)
                {
                    matrix[c][d] = o_matrix[i][j];
                    matrix[d][c] = o_matrix[i][j];
                    d++;
                }
            }
            c++;
        }
    }
    return matrix;
}

/*----------------------------------------------------------*
 * reads kernel matrix from file                            *
 *----------------------------------------------------------*/
double**
kernel_read_matrix(char *fname)
{
    register int i = 0;
    register int j = 0;
    int size   = 0;
    double val = 0;
    FILE *file = fopen(fname, "r");
    double** matrix = NULL;

    fscanf(file, "%d", &size);

    /* Allocating space for new matrix */
    matrix = (double**) malloc(size*sizeof(double*));
    if(matrix == NULL) { printf("Error: Out of memory\n"); exit(1); }

    for(i = 0; i < size; ++i)
    {
        matrix[i] = (double*) malloc(size*sizeof(double));
        if(matrix[i] == NULL) { printf("Error: Out of memory\n"); exit(1); }
    }

    i = 0; j = 0;
    while(!feof(file))
    {
        if(i >= size || j >= size) break;
        if(i == j)
        {
            matrix[j][i] = 10.0;
        }
        else
        {
            if(fscanf(file, "%lf", &val) != 1) break;
            matrix[j][i] = val;
            matrix[i][j] = val;
        }

        j++;
        j = fmod(j, size);
        if(j == 0) i++;
        if(j < i) j = i;
    }
    fclose(file);

    return matrix;
}
