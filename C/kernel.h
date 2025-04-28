/*****************************************************
 * Kernel functions lib                              *
 *****************************************************/

#ifndef KERNEL_H_INCLUDED
#define KERNEL_H_INCLUDED

double** kernel_generate_matrix(sample *sample);
double** kernel_generate_matrix_H(sample *sample);
double** kernel_generate_matrix_H_without_dim(sample *sample, int dim);
double kernel_function(point *one, point *two, int dim);
double kernel_function_without_dim(point *one, point *two, int j, int dim);
void kernel_free_matrix(double **matrix, int size);
double kernel_feature_space_norm(sample *sample, double** matrix);

double** kernel_read_matrix(char *fname);
double** kernel_remove_element(double** o_matrix, int size, int num);
void kernel_print_matrix(double **matrix, int size);

#endif // KERNEL_H_INCLUDED
