/*****************************************************
 * SMO classifier lib                                *
 *****************************************************/
#ifndef SMO_H_INCLUDED
#define SMO_H_INCLUDED

typedef struct smo_learning_data
{
    double error;
    char done; /*character here because it is one byte*/
    int_dll *sv;
} smo_learning_data;

int smo_examine_example(sample *sample, smo_learning_data* l_data,
        double** matrix, int_dll* head, int i1, int verbose);

int smo_max_errors(sample *sample, smo_learning_data* l_data,
        double** matrix, int_dll* head, int i1, double e1, int verbose );

int smo_iterate_non_bound(sample *sample, smo_learning_data *l_data,
        double** matrix, int_dll* head, int i1, int verbose);

int smo_iterate_all_set(sample *sample, smo_learning_data *l_data,
        double** matrix, int_dll* head, int i1, int verbose);

int smo_take_step(sample *sample, smo_learning_data *l_data,
        double** matrix, int_dll* head,int i1, int i2, int verbose);

double smo_function(sample *sample,
        double** matrix, int_dll *head, int index);

int smo_training_routine(sample *sample, smo_learning_data* l_data,
        double** matrix, int_dll* head,int verbose);

void smo_test_learning(sample *sample,
        double **matrix, smo_learning_data* l_data, int_dll *head);

int smo_train(sample *sample, double** w, double *margin, int *svs, int verbose);

int smo_train_matrix(sample *sample, double** matrix, double *margin, int *svs, int verbose);

#endif // SMO_H_INCLUDED
