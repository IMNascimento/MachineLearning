/*****************************************************
 * Admissible ordered search selection lib           *
 *                                                   *
 * Saul Leite <lsaul@lncc.br>                        *
 * sep 23, 2004                                      *
 *****************************************************/
#ifndef AOS_H_INCLUDED
#define AOS_H_INCLUDED

struct aos_select_weight
{
    int fname;
    int indice;
    double w;
    double val;
    double pmargin;
    double raio;
    double dcents;
    double golub;
    double fisher;
};
typedef struct aos_select_weight aos_select_weight;

/* Gamma Struct: Used to stack margin values*/
struct aos_select_gamma
{
    int *fnames;
    int level;
    int sv;
    int train;
    double value; /*valor usado como criterio de escolha*/
    double pgamma; /*projected gamma*/
    double rgamma; /*real gamma p/ display*/
    double raio; /*raio*/
    double dcents; /*distancia entre os centros*/
    double golub; /*golub - estatistica*/
    double fisher; /*fisher - estatistica*/
    double *w;
    double bias;
};
typedef struct aos_select_gamma aos_select_gamma;

struct aos_select_hash
{
    int length;
    int width;
    struct aos_select_gamma ***elements;
};
typedef struct aos_select_hash aos_select_hash;

struct aos_select_heap
{
    int size;
    struct aos_select_gamma **elements;
};
typedef struct aos_select_heap aos_select_heap;

sample* aos_select_features(char *filename, sample *sample,
        int (*train)(struct sample*,double**,double*,int*,int),
        int breadth, int depth, double bonus, int cut,
        int look_ahead_depth, int skip, int startover,
        int doleave_oo, int forma_ordenacao, int forma_escolha,
        crossvalidation *cv, int verbose);

double aos_select_look_ahead(sample *sample, aos_select_heap *heap,
        aos_select_hash *hash, int *fnames_orig, double *w_orig,
        int (*train)(struct sample*,double**,double*,int*,int),
        int depth, int level_orig, int look_ahead_depth,
        double bonus, int forma_escolha, int verbose);

void aos_select_feature_main_loop(sample *sample, aos_select_heap **heap,
        aos_select_hash **hash,
        int (*train)(struct sample*,double**,double*,int*,int),
        int breadth, double bonus, int* lool, int startdim,
        int look_ahead_depth, double *g_margin, int cut, int skip,
        int startover, int doleave_oo, int depth, int forma_ordenacao,
        int forma_escolha, int ftime, crossvalidation *cv, int verbose);

int aos_select_hash_add(aos_select_hash *hash, aos_select_gamma *elmt);
aos_select_hash* aos_select_hash_create(int length, int width);
void aos_select_hash_free(aos_select_hash **hash);
void aos_select_hash_set_null(aos_select_hash *hash, aos_select_gamma *elmt);
void aos_select_hash_print(aos_select_hash *hash, int dim);

aos_select_heap* aos_select_heap_create();
int aos_select_heap_insert(aos_select_heap* heap, aos_select_gamma* tok, int cont);
aos_select_gamma* aos_select_heap_pop(aos_select_heap* heap);
void aos_select_heap_free(aos_select_heap** heap);
void aos_select_heap_print(aos_select_heap *heap);
int aos_select_heap_projected(aos_select_heap *heap);
void aos_select_heap_percolate(aos_select_heap *heap, int i);
void aos_select_heap_cut(aos_select_heap *heap, aos_select_hash *hash, int levelat, int cut, double g_margin, int verbose);

int aos_select_node_equal(aos_select_gamma* one, aos_select_gamma* two);
int aos_select_compare_int_greater(const void *a, const void *b);
int aos_select_compare_weight_greater(const void *a, const void *b);
int aos_select_compare_weightradius_greater(const void *a, const void *b);
int aos_select_compare_weightcenter_greater(const void *a, const void *b);
int aos_select_compare_weightradiuscenter_greater(const void *a, const void *b);
int aos_select_compare_weightfisher_greater(const void *a, const void *b);
int aos_select_compare_weightgolub_greater(const void *a, const void *b);

#endif // AOS_H_INCLUDED
