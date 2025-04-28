/* Data input/output lib
 Data feature manipulate*/

#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED

/*Training points: Used to store data points */
struct point
{
    int    y;
    double *x;
    double  alpha;
};
typedef struct point point;

/*Sammple struct: Used to store sample and feature names*/
struct sample
{
    int size;
    int dim;
    double bias;
    double margin;
    double norm;
    point* points;
    int* fnames;
    int* index; //indices dos pontos (para a correcao do IMA ser mais rapida)
    double q;  //norma q
    double p;  //norma p
    double max_time; //tempo maximo (IMA)
    double mult_tempo; //multiplicador de tempo (IMA)
    double kernel_param;
    int kernel_type;
    double flexivel;
    double alpha_aprox;
    int normalized;
};
typedef struct sample sample;

/* File linked list: Used to read x values from file*/
struct x_ll
{
    double val;
    int fname;
    struct x_ll* next;
};
typedef struct x_ll x_ll; /* x linked list */

/* Data linked list: Used to data values from file*/
struct pnt_ll
{
    point data;
    struct pnt_ll* next;
};
typedef struct pnt_ll pnt_ll; /* point linked list */

int data_load(char* fname, sample **sample, int verbose);
int data_load_routine(FILE* file, sample *sample, int verbose);
int data_write(char *fname, sample *sample, int sv_only);
int data_write_weights(char *fname, double *w, int dim);
int data_load_weights(char *fname, double **w, int dim);
int data_write_routine(FILE *file, sample *sample, int sv_only);
double data_calculate_norm(sample *data, double **K);
void data_norm(sample *sample, double p);
sample* data_normalize_database(sample *samp);
void data_normalized(double *w, int dim, double q);
double data_get_radius(sample *sample, int index, double q);
double data_get_dist_centers(sample *sample, int index);
double data_get_dist_centers_without_feats(sample *sample, int *feats, int fsize, int index);
double data_get_variance(sample *sample, int index);
void data_free_sample(sample **sample);
sample* data_join_samples(sample *sample1, sample *sample2);
sample* data_copy_sample(sample *samp);
sample* data_copy_sample_zero(sample *samp);
void data_part_train_test(sample **train_sample, sample **test_sample, int fold, int seed, int verbose);
void data_remove_test_sample_features(sample *train_sample, sample **test_sample, int verbose);
sample* data_remove_features(sample *samp, int *rem_feat, int fsize, int verbose);
sample* data_insert_features(sample *samp, int *ins_feat, int fsize, int verbose);
sample* data_insert_point(sample *samp, sample *samp_in, int index);
sample* data_remove_point(sample *samp, int point);
double data_standard_deviation(double *vet, int size);
int data_compare_int_greater (const void *a, const void *b);

#endif // DATA_H_INCLUDED
