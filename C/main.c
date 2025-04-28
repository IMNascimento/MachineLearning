/*****************************************
 *           FEATURE SELECTION           *
 *                                       *
 * IMA - Incremental Margin Algorithm    *
 * SMO - Sequential Minimal Optimization *
 * RFE - Recursive Feature Elimination   *
 * AOS - Admissible Ordered Search       *
 *                                       *
 * Saulo Moraes Villela                  *
 * Raul Fonseca Neto                     *
 * Saul de Castro Leite                  *
 * Adilson Elias Xavier                  *
 *                                       *
 * Copyright (C) 2009 / 2010 / 2011      *
 *****************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
//#include <windows.h>
//#include <conio.h>
//#include <gsl/gsl_rng.h>
//#include <gsl/gsl_randist.h>
//#include <gsl/gsl_matrix.h>
//#include <gsl/gsl_linalg.h>
#include "data.h"
#include "kernel.h"
#include "utils.h"
#include "imap.h"
#include "almap.h"
#include "imadual.h"
#include "smo.h"
#include "rfe.h"
#include "aos.h"
#include "golub.h"
#include "fisher.h"
#include "pl.h"

/*cv->erro_atual   = 0;
cv->erro_inicial = 0;
cv->erro_limite  = 0;
cv->fold         = 0;
cv->jump         = 0;
cv->qtde         = 0;
cv->seed         = NULL;*/

int verbose = 1; //ver resultados
double max_time = 100.0f; //tempo maximo
double mult_tempo = 0.01; //multiplicador do tempo para demais dimensoes
double q = 2, p = 2; //norma p e q (conjugadas)
int dimensao = 0; //dimensao desejada (qtde de caracteristicas restantes - profundidade)
int fold = 3; //k-fold cross-validation
int qtdecroos = 1; //qtde de cross-validation
double errocross = 0, *errocrossVet; //erro de cross-validation
int normalizar = 0; //normalizar dados
int verfeat = 0; //ver features nas informacoes
int totalfeat = 0; //total de features a serem inseridas ou removidas
int *feats = NULL; //vetor de features a serem inseridas ou removidas
int flag_feat = 0; //flag para verificar se a feature pertence ao conjunto original

int kernel_type = 9; //tipo de kernel (9 = IMA Primal)
double kernel_param = 1; //parametro do kernel

double flexivel = 0; //flexibilização da margem (deixar errar)
double alpha_prox = 0; // alfa aproximação para o ALMAp

char arquivo_entrada[100] = "db/iris.data";
char arquivo_entrada_temp[100] = "db/";
char base_entrada[100] = "iris";
char arquivo_saida[100] = "aos_imap_";


int
abrir_arquivo(struct sample **sample)
{
    if(!(*sample))
        if(!data_load(arquivo_entrada, sample, 0))
        {
            printf("Erro na leitura do arquivo.\n");
            return 0;
        }
    // (*sample)->normalized = 0;
    return 1;
}

void
rfe(struct sample **sample, int op_rfe)
{
    int jump = 0; //qtde de caracteristicas eliminadas por vez (RFE)
    crossvalidation *cv = (crossvalidation*) malloc(sizeof(crossvalidation));

    if(op_rfe == 1)
    {
        printf("Valor da norma q: ");
        scanf("%lf", &q);
        (*sample)->kernel_type  = 9;
        printf("Valor da flexibilizacao (0 - sem flexibilizacao): ");
        scanf("%lf", &flexivel);
        (*sample)->flexivel = flexivel;
        (*sample)->alpha_aprox  = alpha_prox;
    }
    else if(op_rfe == 4)
    {
        q = 1;
        (*sample)->kernel_type  = 9;
    }
    else
    {
        q = 2;
        printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
        scanf("%d", &kernel_type);
        if(kernel_type == 1)
            printf("Grau do polinomio: ");
        else if(kernel_type == 2)
            printf("Gamma do gaussiano: ");
        if(kernel_type != 0)
            scanf("%lf", &kernel_param);
    }
    printf("Dimensao desejada (max. %d): ", (*sample)->dim);
    scanf("%d", &dimensao);
    printf("Caracteristicas eliminadas por vez: ");
    scanf("%d", &jump);
    //printf("Quantidade de Cross-Validation: ");
    //scanf("%d", &cv->qtde);
    cv->qtde = 0;
    if(cv->qtde > 0)
    {
        printf("K-fold: ");
        scanf("%d", &cv->fold);
        printf("De quantas em quantas dimensoes: ");
        scanf("%d", &cv->jump);
        printf("Margem de erro: ");
        scanf("%lf", &cv->erro_limite);
    }

    (*sample)->q            = q;
    (*sample)->kernel_type  = kernel_type;
    (*sample)->kernel_param = kernel_param;
    (*sample)->max_time     = max_time;
    (*sample)->mult_tempo   = mult_tempo;

    switch(op_rfe)
    {
        case 1: //RFE IMAp
                strcpy(arquivo_saida, "rfe_imap_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = rfe_select_features(arquivo_saida, (*sample), imap, ((*sample)->dim)-dimensao, jump, 0, 1, cv, verbose);
                break;

        case 2: //RFE IMA Dual
                strcpy(arquivo_saida, "rfe_imad_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = rfe_select_features(arquivo_saida, (*sample), imadual, ((*sample)->dim)-dimensao, jump, 0, 1, cv, verbose);
                break;

        case 3: //RFE SMO
                strcpy(arquivo_saida, "rfe_smo_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = rfe_select_features(arquivo_saida, (*sample), smo_train, ((*sample)->dim)-dimensao, jump, 0, 1, cv, verbose);
                break;

        case 4: //RFE PL
                strcpy(arquivo_saida, "rfe_pl_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = rfe_select_features(arquivo_saida, (*sample), linear_programming, ((*sample)->dim)-dimensao, jump, 0, 1, cv, verbose);
                break;

        default:puts("Opcao invalida.\n");
                break;
    }
    free(cv);
}

void
aos(struct sample **sample, int op_aos)
{
    int ramificacao = 0; //fator de ramificacao
    int prof_look_ahead = 0; //profundidade do look ahead no AOS (look_ahead_depth)
    int startover = 999999; //profundidade do corte de todos os nodos do heap e do hash (recomeço da solução)
    int cut = 0; //corte de niveis acima
    int forma_ordenacao = 0; //ordenar pelo w ou por outras formas
    int forma_escolha = 0; //escolher pela margem ou margem * dist. centros
    crossvalidation *cv = (crossvalidation*) malloc(sizeof(crossvalidation));

    if(op_aos == 1)
    {
        printf("Valor da norma q: ");
        scanf("%lf", &q);
        (*sample)->kernel_type  = 9;
        printf("Valor da flexibilizacao (0 - sem flexibilizacao): ");
        scanf("%lf", &flexivel);
        (*sample)->flexivel = flexivel;
        (*sample)->alpha_aprox = alpha_prox;
    }
    else if(op_aos == 4)
    {
        q = 1;
        (*sample)->kernel_type  = 9;
    }
    else
    {
        q = 2;
        printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
        scanf("%d", &kernel_type);
        if(kernel_type == 1)
            printf("Grau do polinomio: ");
        else if(kernel_type == 2)
            printf("Gamma do gaussiano: ");
        if(kernel_type != 0)
            scanf("%lf", &kernel_param);
    }
    printf("Dimensao desejada (max. %d): ", (*sample)->dim);
    scanf("%d", &dimensao);
    printf("Fator de ramificacao (max. %d): ", (*sample)->dim);
    scanf("%d", &ramificacao);
    printf("Ordenacao da ramificacao: (1)W (2)W/centro (3)W*raio/centro (4)W*raio (5)W*Golub (6)W*Fisher: ");
    scanf("%d", &forma_ordenacao);
    printf("Escolha: (1)Margem (2)Margem*Dist.Centros: ");
    scanf("%d", &forma_escolha);
    printf("Profundidade do Look-Ahead: ");
    scanf("%d", &prof_look_ahead);
    printf("Profundidade do corte(Cut): ");
    scanf("%d", &cut);
    //printf("Profundidade do startover: ");
    //scanf("%d", &startover);

    //printf("Quantidade de Cross-Validation: ");
    //scanf("%d", &cv->qtde);
    cv->qtde = 0;
    if(cv->qtde > 0)
    {
        printf("K-fold: ");
        scanf("%d", &cv->fold);
        printf("De quantas em quantas dimensoes: ");
        scanf("%d", &cv->jump);
        printf("Margem de erro: ");
        scanf("%lf", &cv->erro_limite);
    }

    (*sample)->q            = q;
    (*sample)->kernel_type  = kernel_type;
    (*sample)->kernel_param = kernel_param;
    (*sample)->max_time     = max_time;
    (*sample)->mult_tempo   = mult_tempo;

    switch(op_aos)
    {
        case 1: //AOS IMAp
                strcpy(arquivo_saida, "aos_imap_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = aos_select_features(arquivo_saida, (*sample), imap,
                                    ramificacao,                                //bredth
                                    ((*sample)->dim)-dimensao,                  //depth
                                    0,                                          //bonus
                                    cut,                                        //cut
                                    prof_look_ahead,                            //look ahead
                                    1,                                          //skip non-SV on leave one out
                                    startover,                                  //startover
                                    0,                                          //leave one out
                                    forma_ordenacao,                            //forma de ordenacao
                                    forma_escolha,                              //forma de escolha (margem ou margem*centro)
                                    cv,                                         //cross-validation
                                    verbose);                                   //ver resultados
                break;

        case 2: //AOS IMA Dual
                (*sample)->mult_tempo = 2; //"gambiarra" para passar o IMA Dual como parâmetro
                strcpy(arquivo_saida, "aos_imad_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = aos_select_features(arquivo_saida, (*sample), imadual,
                                    ramificacao,                                //bredth
                                    ((*sample)->dim)-dimensao,                  //depth
                                    0,                                          //bonus
                                    cut,                                        //cut
                                    prof_look_ahead,                            //look ahead
                                    1,                                          //skip non-SV on leave one out
                                    startover,                                  //startover
                                    0,                                          //leave one out
                                    forma_ordenacao,                            //forma de ordenacao
                                    forma_escolha,                              //forma de escolha (margem ou margem*centro)
                                    cv,                                         //cross-validation
                                    verbose);                                   //ver resultados
                break;

        case 3: //AOS SMO
                strcpy(arquivo_saida, "aos_smo_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = aos_select_features(arquivo_saida, (*sample), smo_train,
                                    ramificacao,                                //bredth
                                    ((*sample)->dim)-dimensao,                  //depth
                                    0,                                          //bonus
                                    cut,                                        //cut
                                    prof_look_ahead,                            //look ahead
                                    1,                                          //skip
                                    startover,                                  //startover
                                    0,                                          //leave one out
                                    forma_ordenacao,                            //forma de ordenacao
                                    forma_escolha,                              //forma de escolha (margem ou margem*centro)
                                    cv,                                         //cross-validation
                                    verbose);                                   //ver resultados
                break;

        case 4: //AOS PL
                strcpy(arquivo_saida, "aos_pl_");
                strcat(arquivo_saida, base_entrada);
                (*sample) = aos_select_features(arquivo_saida, (*sample), linear_programming,
                                    ramificacao,                                //bredth
                                    ((*sample)->dim)-dimensao,                  //depth
                                    0,                                          //bonus
                                    cut,                                        //cut
                                    prof_look_ahead,                            //look ahead
                                    1,                                          //skip non-SV on leave one out
                                    startover,                                  //startover
                                    0,                                          //leave one out
                                    forma_ordenacao,                            //forma de ordenacao
                                    forma_escolha,                              //forma de escolha (margem ou margem*centro)
                                    cv,                                         //cross-validation
                                    verbose);                                   //ver resultados
                break;


        default:puts("Opcao invalida.\n");
                break;
    }
    printf("Base de dados: %s\n", base_entrada);
    if(op_aos == 1)
        printf("Classificador: IMA Primal -- Norma: %.1lf\n", q);
    else if(op_aos == 4)
        printf("Classificador: PL -- Norma: %.1lf\n", q);
    else
    {
        if(op_aos == 2)
            printf("Classificador: IMA Dual -- Kernel: ");
        else
            printf("Classificador: SMO -- Kernel: ");
        if(kernel_type == 0)
            printf("Produto Interno\n");
        if(kernel_type == 1)
            printf("Polinomial -- Grau: %.0f\n", kernel_param);
        else if(kernel_type == 2)
            printf("Gaussiano -- Gamma: %.3f\n", kernel_param);
    }
    printf("Fator de ramificacao: %d\n", ramificacao);
    printf("Ordenacao da ramificacao: ");
    switch(forma_ordenacao)
    {
        case 2: puts("W/centro"); break;
        case 3: puts("W*raio/centro"); break;
        case 4: puts("W*raio"); break;
        case 5: puts("W*golub"); break;
        case 6: puts("W*fisher"); break;
        default:puts("W");
    }
    printf("Escolha: ");
    if(forma_escolha == 2)
        puts("margem*dist.centros");
    else
        puts("margem");
    printf("Profundidade do Look-Ahead: %d\n", prof_look_ahead);
    printf("Profundidade do corte(Cut): %d\n\n", cut);
    free(cv);
}

void
separar_base()
{

}

int
main()
{
    register int i = 0, j = 0;
    int qtdpos = 0, qtdneg = 0;
    int opcao, op_rfe, op_aos, op_arq_ent, op_fold, op_golub, op_fisher;
    double *w = NULL;
    double margin = 0;
    int svs = 0;
    int plot, norma = 2;
    int semente;
    int removerFeat;

    struct sample *sample = NULL,
                  *test_sample = NULL,
                  *sample_temp = NULL;

    do {//menu
        printf("\t\t########################################################\n");
        printf("\t\t#                                                      #\n");
        printf("\t\t#              .:: FEATURE SELECTION ::.               #\n");
        printf("\t\t#                                                      #\n");
        printf("\t\t########################################################\n");
        printf("\n");
        puts(" 1 - Arquivo de Entrada");
        puts(" 2 - Tempos");
        puts(" 3 - Ver Resultados?");
        puts(" 4 - IMA Primal");
        puts(" 5 - IMA Dual");
        puts(" 6 - SMO");
        puts(" 7 - RFE");
        puts(" 8 - AOS");
        puts(" 9 - Golub");
        puts("10 - Fisher");
        puts("11 - K-Fold");
        puts("12 - Informacoes da Base");
        puts("13 - Remover Features");
        puts("14 - Adicionar Features");
        puts("15 - Validacao com K-Fold");
        puts("16 - Separar Base em Treino/Teste");
        puts("17 - Salvar Bases de Treino/Teste");
        puts("18 - ALMAp");
        puts("19 - Normalizar Base de Dados");
        puts("20 - Programacao Linear");
        puts(" 0 - Sair");
        printf("\n");
        scanf("%d%*c", &opcao);
        switch(opcao)
        {
            case 1: //database
                    puts("Bases de Teste: ------------------------");
                    puts(" 1 - iris.data");
                    puts(" 2 - toy.data");
                    puts(" 3 - test.data");
                    puts("Bases Comuns: --------------------------");
                    puts("11 - mushroom.data");
                    puts("Bases Sinteticas: ----------------------");
                    puts("21 - sc.data");
                    puts("22 - lp4.data");
                    puts("Bases Nao Lineares: --------------------");
                    puts("31 - sonar.data");
                    puts("32 - ionosphere.data");
                    puts("33 - wdbc.data");
                    puts("34 - bupa.data");
                    puts("35 - pima.data");
                    puts("36 - wine.data");
                    puts("Bases de Microarray: -------------------");
                    puts("41 - prostate.data");
                    puts("42 - breast.data");
                    puts("43 - colon.data");
                    puts("44 - leukemia.data");
                    puts("45 - dlbcl.data");
                    puts("46 - prostate_tumor.data");
                    puts("Bases de Microarray com 100 features: --");
                    puts("51 - prostate100rfe.data");
                    puts("52 - breast100rfe.data");
                    puts("53 - colon100rfe.data");
                    puts("54 - leukemia100rfe.data");
                    puts("55 - dlbcl100rfe.data");
                    puts("Bases com 20 features: -----------------");
                    //puts("61 - 20_mushroom.data");
                    puts("62 - 20_sc.data");
                    puts("63 - 20_lp4.data");
                    puts("64 - 20_sonar.data");
                    puts("65 - 20_ionosphere.data");
                    puts("66 - 20_prostate.data");
                    //puts("67 - 20_breast.data");
                    puts("68 - 20_colon.data");
                    //puts("69 - 20_leukemia.data");
                    /*
                    puts("----------Bases com 30 features:---------");
                    puts("71 - mushroom_30.data");
                    puts("72 - sc_30.data");
                    puts("73 - lp4_30.data");
                    puts("74 - sonar_30.data");
                    puts("75 - ionosphere_30.data");
                    puts("76 - prostate_30.data");
                    puts("77 - breast_30.data");
                    puts("78 - colon_30.data");
                    puts("79 - leukemia_30.data");
                    */
                    puts("Bases Artificiais: ---------------------");
                    puts("81 - artificial_1.data");
                    puts("82 - artificial_2.data");
                    puts("83 - artificial_3.data");
                    puts("84 - artificial_4.data");
                    puts("85 - artificial_5.data");
                    puts("----------------------------------------");
                    puts(" 0 - outra base");
                    puts("----------------------------------------");
                    scanf("%d", &op_arq_ent);
                    switch(op_arq_ent)
                    {
                        case  1: strcpy(arquivo_entrada, "db/iris.data");           strcpy(base_entrada, "iris");           break;
                        case  2: strcpy(arquivo_entrada, "db/toy.data");            strcpy(base_entrada, "toy");            break;
                        case  3: strcpy(arquivo_entrada, "db/test.data");           strcpy(base_entrada, "test");           break;

                        case 11: strcpy(arquivo_entrada, "db/mushroom.data");       strcpy(base_entrada, "mushroom");       break;

                        case 21: strcpy(arquivo_entrada, "db/sc.data");             strcpy(base_entrada, "sc");             break;
                        case 22: strcpy(arquivo_entrada, "db/lp4.data");            strcpy(base_entrada, "lp4");            break;

                        case 31: strcpy(arquivo_entrada, "db/sonar.data");          strcpy(base_entrada, "sonar");          break;
                        case 32: strcpy(arquivo_entrada, "db/ionosphere.data");     strcpy(base_entrada, "ionosphere");     break;
                        case 33: strcpy(arquivo_entrada, "db/wdbc.data");           strcpy(base_entrada, "wdbc");           break;
                        case 34: strcpy(arquivo_entrada, "db/bupa.data");           strcpy(base_entrada, "bupa");           break;
                        case 35: strcpy(arquivo_entrada, "db/pima.data");           strcpy(base_entrada, "pima");           break;
                        case 36: strcpy(arquivo_entrada, "db/wine.data");           strcpy(base_entrada, "wine");           break;

                        case 41: strcpy(arquivo_entrada, "db/prostate.data");       strcpy(base_entrada, "prostate");       break;
                        case 42: strcpy(arquivo_entrada, "db/breast.data");         strcpy(base_entrada, "breast");         break;
                        case 43: strcpy(arquivo_entrada, "db/colon.data");          strcpy(base_entrada, "colon");          break;
                        case 44: strcpy(arquivo_entrada, "db/leukemia.data");       strcpy(base_entrada, "leukemia");       break;
                        case 45: strcpy(arquivo_entrada, "db/dlbcl.data");          strcpy(base_entrada, "dlbcl");          break;
                        case 46: strcpy(arquivo_entrada, "db/prostatetumor.data");  strcpy(base_entrada, "prostatetumor");  break;

                        case 51: strcpy(arquivo_entrada, "db/prostate100rfe.data"); strcpy(base_entrada, "prostate100rfe"); break;
                        case 52: strcpy(arquivo_entrada, "db/breast100rfe.data");   strcpy(base_entrada, "breast100rfe");   break;
                        case 53: strcpy(arquivo_entrada, "db/colon100rfe.data");    strcpy(base_entrada, "colon100rfe");    break;
                        case 54: strcpy(arquivo_entrada, "db/leukemia100rfe.data"); strcpy(base_entrada, "leukemia100rfe"); break;
                        case 55: strcpy(arquivo_entrada, "db/dlbcl100rfe.data");    strcpy(base_entrada, "dlbcl100rfe");    break;

                        case 61: strcpy(arquivo_entrada, "db/20_mushroom.data");    strcpy(base_entrada, "20_mushroom");    break;
                        case 62: strcpy(arquivo_entrada, "db/20_sc.data");          strcpy(base_entrada, "20_sc");          break;
                        case 63: strcpy(arquivo_entrada, "db/20_lp4.data");         strcpy(base_entrada, "20_lp4");         break;
                        case 64: strcpy(arquivo_entrada, "db/20_sonar.data");       strcpy(base_entrada, "20_sonar");       break;
                        case 65: strcpy(arquivo_entrada, "db/20_ionosphere.data");  strcpy(base_entrada, "20_ionosphere");  break;
                        case 66: strcpy(arquivo_entrada, "db/20_prostate.data");    strcpy(base_entrada, "20_prostate");    break;
                        case 67: strcpy(arquivo_entrada, "db/20_breast.data");      strcpy(base_entrada, "20_breast");      break;
                        case 68: strcpy(arquivo_entrada, "db/20_colon.data");       strcpy(base_entrada, "20_colon");       break;
                        case 69: strcpy(arquivo_entrada, "db/20_leukemia.data");    strcpy(base_entrada, "20_leukemia");    break;

                        /*
                        case 71: strcpy(arquivo_entrada, "db/mushroom_30.data");    strcpy(base_entrada, "mushroom_30");    break;
                        case 72: strcpy(arquivo_entrada, "db/sc_30.data");          strcpy(base_entrada, "sc_30");          break;
                        case 73: strcpy(arquivo_entrada, "db/lp4_30.data");         strcpy(base_entrada, "lp4_30");         break;
                        case 74: strcpy(arquivo_entrada, "db/sonar_30.data");       strcpy(base_entrada, "sonar_30");       break;
                        case 75: strcpy(arquivo_entrada, "db/ionosphere_30.data");  strcpy(base_entrada, "ionosphere_30");  break;
                        case 76: strcpy(arquivo_entrada, "db/prostate_30.data");    strcpy(base_entrada, "prostate_30");    break;
                        case 77: strcpy(arquivo_entrada, "db/breast_30.data");      strcpy(base_entrada, "breast_30");      break;
                        case 78: strcpy(arquivo_entrada, "db/colon_30.data");       strcpy(base_entrada, "colon_30");       break;
                        case 79: strcpy(arquivo_entrada, "db/leukemia_30.data");    strcpy(base_entrada, "leukemia_30");    break;
                        */

                        case 81: strcpy(arquivo_entrada, "db/artificial_1.data");   strcpy(base_entrada, "artificial_1");   break;
                        case 82: strcpy(arquivo_entrada, "db/artificial_2.data");   strcpy(base_entrada, "artificial_2");   break;
                        case 83: strcpy(arquivo_entrada, "db/artificial_3.data");   strcpy(base_entrada, "artificial_3");   break;
                        case 84: strcpy(arquivo_entrada, "db/artificial_4.data");   strcpy(base_entrada, "artificial_4");   break;
                        case 85: strcpy(arquivo_entrada, "db/artificial_5.data");   strcpy(base_entrada, "artificial_5");   break;

                        case 0:  puts("Digite o nome do arquivo da base de dados (pasta db): ");
                                 scanf("%s", arquivo_entrada);
                                 strcpy(base_entrada, arquivo_entrada);
                                 strcpy(arquivo_entrada_temp, "db/");
                                 strcat(arquivo_entrada_temp, arquivo_entrada);
                                 strcpy(arquivo_entrada, arquivo_entrada_temp);
                                 break;
                        default: puts("Opcao invalida.\n");
                    }
                    data_free_sample(&sample);
                    data_free_sample(&test_sample);
                    break;

            case 2: //max time
                    printf("Tempo maximo em segundos: ");
                    scanf("%lf", &max_time);
                    max_time *= 100.0f;
                    printf("Multiplicador p/ demais dimensoes: ");
                    scanf("%lf", &mult_tempo);
                    break;

            case 3: //verbose
                    printf("0 - Nao\n1 - Somente Res.\n2 - Todos\n\n");
                    scanf("%d", &verbose);
                    break;

            case 4: //IMAp
                    if(abrir_arquivo(&sample))
                    {
                        printf("Norma p (1) ou q (2): ");
                        scanf("%d", &norma);
                        if(norma == 1)
                        {
                            printf("Valor da norma p: ");
                            scanf("%lf", &p);
                            if(p == 1.0)
                                q = -1.0;
                            else
                                q = p/(p-1.0);
                        }
                        else
                        {
                            printf("Valor da norma q: ");
                            scanf("%lf", &q);
                            if(q == -1.0)
                                p = 1.0;
                            else if(q == 1.0)
                                p = 100.0;
                            else
                                p = q/(q-1.0);
                        }

                        printf("Valor da flexibilizacao (0 - sem flexibilizacao): ");
                        scanf("%lf", &flexivel);

                        printf("Valor da aproximacao alfa (1 - alfa): ");
                        scanf("%lf", &alpha_prox);

                        w = NULL;
                        margin = 0;
                        svs = 0;

                        sample->p            = p;
                        sample->q            = q;
                        sample->max_time     = max_time;
                        sample->kernel_type  = 9;
                        sample->flexivel     = flexivel;
                        sample->alpha_aprox  = alpha_prox;

                        if(imap(sample, &w, &margin, &svs, verbose))
                        {
                            printf("Treinamento com sucesso...\n");
                            printf("Margem = %lf, Vetores Suporte = %d\n\n", margin, svs);
                            if(sample->dim == 2 || sample->dim == 3)
                            {
                                printf("Deseja plotar o grafico? (0)Nao (1)Sim: ");
                                scanf("%d", &plot);
                                if(plot)
                                {
                                    if(sample->dim == 2)
                                        utils_plot_2d(sample, w, base_entrada, "imap");
                                    else
                                        utils_plot_3d(sample, w, base_entrada, "imap");
                                }
                            }
                        }
                        else
                            printf("Treinamento falhou.\n\n");
                        free(w);
                        free(sample->index);
                        sample->index = NULL;
                    }
                    break;

            case 5: //IMA Dual
                    if(abrir_arquivo(&sample))
                    {
                        printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                        scanf("%d", &kernel_type);
                        if(kernel_type == 1)
                            printf("Grau do polinomio: ");
                        else if(kernel_type == 2)
                            printf("Gamma do gaussiano: ");
                        if(kernel_type != 0)
                           scanf("%lf", &kernel_param);

                        w = NULL;
                        margin = 0;
                        svs = 0;

                        sample->q            = 2;
                        sample->max_time     = max_time;
                        sample->kernel_type  = kernel_type;
                        sample->kernel_param = kernel_param;

                        if(imadual(sample, &w, &margin, &svs, verbose))
                        {
                            printf("Treinamento com sucesso...\n");
                            printf("Margem = %lf, Vetores Suporte = %d\n\n", margin, svs);
                            if(sample->dim == 2 || sample->dim == 3)
                            {
                                printf("Deseja plotar o grafico? (0)Nao (1)Sim: ");
                                scanf("%d", &plot);
                                if(plot)
                                {
                                    if(sample->dim == 2)
                                        utils_plot_2d(sample, w, base_entrada, "imad");
                                    else
                                        utils_plot_3d(sample, w, base_entrada, "imad");
                                }
                            }
                        }
                        else
                            printf("Treinamento falhou.\n\n");
                        free(w);
                        free(sample->index);
                        sample->index = NULL;
                    }
                    break;

            case 6: //SMO
                    if(abrir_arquivo(&sample))
                    {
                        printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                        scanf("%d", &kernel_type);
                        if(kernel_type == 1)
                            printf("Grau do polinomio: ");
                        else if(kernel_type == 2)
                            printf("Gamma do gaussiano: ");
                        if(kernel_type != 0)
                           scanf("%lf", &kernel_param);

                        w = NULL;
                        margin = 0;
                        svs = 0;

                        sample->q            = 2;
                        sample->kernel_type  = kernel_type;
                        sample->kernel_param = kernel_param;

                        if(smo_train(sample, &w, &margin, &svs, verbose))
                        {
                            printf("Treinamento com sucesso...\n");
                            printf("Margem = %lf, Vetores Suporte = %d\n\n", margin, svs);
                            if(sample->dim == 2 || sample->dim == 3)
                            {
                                printf("Deseja plotar o grafico? (0)Nao (1)Sim: ");
                                scanf("%d", &plot);
                                if(plot)
                                {
                                    if(sample->dim == 2)
                                        utils_plot_2d(sample, w, base_entrada, "smo");
                                    else
                                        utils_plot_3d(sample, w, base_entrada, "smo");
                                }
                            }
                        }
                        else
                            printf("Treinamento falhou.\n\n");
                        free(w);
                    }
                    break;

            case 7: //RFE
                    if(abrir_arquivo(&sample))
                    {
                        if(!test_sample)
                        {
                            printf("Dividir base em treino/teste? (0)Nao (1)Sim: ");
                            scanf("%d", &flag_feat);
                            if(flag_feat)
                            {
                                //printf("K-Fold: ");
                                //scanf("%d", &fold);
                                fold = 3;
                                data_part_train_test(&sample, &test_sample, fold, 0, verbose);
                            }
                        }
                        puts("RFE");
                        puts("1 - IMAp");
                        puts("2 - IMA Dual");
                        puts("3 - SMO");
                        puts("4 - PL");
                        scanf("%d", &op_rfe);
                        if((op_rfe == 1 || op_rfe == 2 || op_rfe == 3 || op_rfe == 4) && abrir_arquivo(&sample))
                            rfe(&sample, op_rfe);
                        else
                            puts("Opcao invalida.\n");
                    }
                    break;

            case 8: //AOS
                    if(abrir_arquivo(&sample))
                    {
                        if(!test_sample)
                        {
                            printf("Dividir base em treino/teste? (0)Nao (1)Sim: ");
                            scanf("%d", &flag_feat);
                            if(flag_feat)
                            {
                                //printf("K-Fold: ");
                                //scanf("%d", &fold);
                                fold = 3;
                                data_part_train_test(&sample, &test_sample, fold, 0, verbose);
                            }
                        }
                        puts("AOS");
                        puts("1 - IMAp");
                        puts("2 - IMA Dual");
                        puts("3 - SMO");
                        puts("4 - PL");
                        scanf("%d", &op_aos);
                        if((op_aos == 1 || op_aos == 2 || op_aos == 3 || op_aos == 4) && abrir_arquivo(&sample))
                            aos(&sample, op_aos);
                        else
                            puts("Opcao invalida.\n");
                    }
                    break;

            case 9: //Golub
                    if(abrir_arquivo(&sample))
                    {
                        if(!test_sample)
                        {
                            printf("Dividir base em treino/teste? (0)Nao (1)Sim: ");
                            scanf("%d", &flag_feat);
                            if(flag_feat)
                            {
                                //printf("K-Fold: ");
                                //scanf("%d", &fold);
                                fold = 3;
                                data_part_train_test(&sample, &test_sample, fold, 0, verbose);
                            }
                        }
                        //printf("Digite a dimensao desejada (max. %d): ", sample->dim);
                        //scanf("%d", &dimensao);
                        dimensao = 1;
                        puts("Golub");
                        puts("1 - IMAp");
                        puts("2 - IMA Dual");
                        puts("3 - SMO");
                        scanf("%d", &op_golub);
                        switch(op_golub)
                        {
                            case 1: //IMAp
                                    //printf("Valor da norma q: ");
                                    //scanf("%lf", &q);

                                    sample->q            = 2;
                                    sample->max_time     = max_time;
                                    sample->kernel_type  = 9;

                                    strcpy(arquivo_saida, "golub_imap_");
                                    strcat(arquivo_saida, base_entrada);

                                    sample = golub_select_features(arquivo_saida, sample, imap, dimensao, verbose);
                                    break;

                            case 2: //IMA Dual
                                    printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                    scanf("%d", &kernel_type);
                                    if(kernel_type == 1)
                                        printf("Grau do polinomio: ");
                                    else if(kernel_type == 2)
                                        printf("Gamma do gaussiano: ");
                                    if(kernel_type != 0)
                                        scanf("%lf", &kernel_param);

                                    sample->q            = 2;
                                    sample->kernel_type  = kernel_type;
                                    sample->kernel_param = kernel_param;
                                    sample->max_time     = max_time;

                                    strcpy(arquivo_saida, "golub_imad_");
                                    strcat(arquivo_saida, base_entrada);

                                    sample = golub_select_features(arquivo_saida, sample, imadual, dimensao, verbose);
                                    break;

                            case 3: //SMO
                                    printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                    scanf("%d", &kernel_type);
                                    if(kernel_type == 1)
                                        printf("Grau do polinomio: ");
                                    else if(kernel_type == 2)
                                        printf("Gamma do gaussiano: ");
                                    if(kernel_type != 0)
                                        scanf("%lf", &kernel_param);

                                    sample->q            = 2;
                                    sample->kernel_type  = kernel_type;
                                    sample->kernel_param = kernel_param;

                                    strcpy(arquivo_saida, "golub_smo_");
                                    strcat(arquivo_saida, base_entrada);

                                    sample = golub_select_features(arquivo_saida, sample, smo_train, dimensao, verbose);
                                    break;

                            default:puts("Opcao invalida.\n");
                                    break;
                        }
                    }
                    break;

            case 10://Fisher
                    if(abrir_arquivo(&sample))
                    {
                        if(!test_sample)
                        {
                            printf("Dividir base em treino/teste? (0)Nao (1)Sim: ");
                            scanf("%d", &flag_feat);
                            if(flag_feat)
                            {
                                //printf("K-Fold: ");
                                //scanf("%d", &fold);
                                fold = 3;
                                data_part_train_test(&sample, &test_sample, fold, 0, verbose);
                            }
                        }
                        //printf("Digite a dimensao desejada (max. %d): ", sample->dim);
                        //scanf("%d", &dimensao);
                        dimensao = 1;
                        puts("Fisher");
                        puts("1 - IMAp");
                        puts("2 - IMA Dual");
                        puts("3 - SMO");
                        scanf("%d", &op_fisher);
                        switch(op_golub)
                        {
                            case 1: //IMAp
                                    //printf("Valor da norma q: ");
                                    //scanf("%lf", &q);

                                    sample->q            = 2;
                                    sample->max_time     = max_time;
                                    sample->kernel_type  = 9;

                                    strcpy(arquivo_saida, "fisher_imap_");
                                    strcat(arquivo_saida, base_entrada);

                                    sample = fisher_select_features(arquivo_saida, sample, imap, dimensao, verbose);
                                    break;

                            case 2: //IMA Dual
                                    printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                    scanf("%d", &kernel_type);
                                    if(kernel_type == 1)
                                        printf("Grau do polinomio: ");
                                    else if(kernel_type == 2)
                                        printf("Gamma do gaussiano: ");
                                    if(kernel_type != 0)
                                        scanf("%lf", &kernel_param);

                                    sample->q            = 2;
                                    sample->kernel_type  = kernel_type;
                                    sample->kernel_param = kernel_param;
                                    sample->max_time     = max_time;

                                    strcpy(arquivo_saida, "fisher_imad_");
                                    strcat(arquivo_saida, base_entrada);

                                    sample = fisher_select_features(arquivo_saida, sample, imadual, dimensao, verbose);
                                    break;

                            case 3: //SMO
                                    printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                    scanf("%d", &kernel_type);
                                    if(kernel_type == 1)
                                        printf("Grau do polinomio: ");
                                    else if(kernel_type == 2)
                                        printf("Gamma do gaussiano: ");
                                    if(kernel_type != 0)
                                        scanf("%lf", &kernel_param);

                                    sample->q            = 2;
                                    sample->kernel_type  = kernel_type;
                                    sample->kernel_param = kernel_param;

                                    strcpy(arquivo_saida, "fisher_smo_");
                                    strcat(arquivo_saida, base_entrada);

                                    sample = fisher_select_features(arquivo_saida, sample, smo_train, dimensao, verbose);
                                    break;

                            default:puts("Opcao invalida.\n");
                                    break;
                        }
                    }
                    break;

            case 11://K-Fold
                    if(abrir_arquivo(&sample))
                    {
                        printf("Quantidade: ");
                        scanf("%d", &qtdecroos);
                        errocrossVet = (double*) malloc(qtdecroos*sizeof(double));
                        if(!errocrossVet) return -1;
                        printf("K-Fold: ");
                        scanf("%d", &fold);
                        //fold = 10;

                        puts("1 - IMAp");
                        puts("2 - IMA Dual");
                        puts("3 - SMO");
                        puts("4 - PL");
                        scanf("%d", &op_fold);
                        switch(op_fold)
                        {
                            case 1: //IMAp
                                    printf("Norma p (1) ou q (2): ");
                                    scanf("%d", &norma);
                                    if(norma == 1)
                                    {
                                        printf("Valor da norma p: ");
                                        scanf("%lf", &p);
                                        if(p == 1.0)
                                            q = -1.0;
                                        else
                                            q = p/(p-1.0);
                                    }
                                    else
                                    {
                                        printf("Valor da norma q: ");
                                        scanf("%lf", &q);
                                        if(q == 1.0)
                                            p = 100.0;
                                        else
                                            p = q/(q-1.0);
                                    }

                                    printf("Valor da flexibilizacao (0 - sem flexibilizacao): ");
                                    scanf("%lf", &flexivel);

                                    printf("Valor da aproximacao alfa (1 - alfa): ");
                                    scanf("%lf", &alpha_prox);

                                    w = NULL;
                                    margin = 0;
                                    svs = 0;

                                    sample->p            = p;//2;
                                    sample->q            = q;//2;
                                    sample->max_time     = max_time;
                                    sample->kernel_type  = 9;
                                    sample->flexivel     = flexivel;
                                    sample->alpha_aprox  = alpha_prox;

                                    for(errocross = 0, i = 0; i < qtdecroos; i++)
                                    {
                                        if(verbose) printf("\nExecucao %d / %d:\n", i+1, qtdecroos);
                                        errocrossVet[i] = utils_k_fold(sample, imap, fold, i, verbose);
                                        errocross += errocrossVet[i];
                                        printf("Erro Execucao %d / %d: %.2lf%%\n", i+1, qtdecroos, errocrossVet[i]);
                                    }
                                    printf("\nErro Medio %d-Fold Cross Validation: %.2lf %c %.2lf\n", fold, errocross/qtdecroos, 241, data_standard_deviation(errocrossVet, qtdecroos));
                                    break;

                            case 2: //IMA Dual
                                    printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                    scanf("%d", &kernel_type);
                                    if(kernel_type == 1)
                                        printf("Grau do polinomio: ");
                                    else if(kernel_type == 2)
                                        printf("Gamma do gaussiano: ");
                                    if(kernel_type != 0)
                                        scanf("%lf", &kernel_param);

                                    sample->q            = 2;
                                    sample->max_time     = max_time;
                                    sample->kernel_type  = kernel_type;
                                    sample->kernel_param = kernel_param;

                                    for(errocross = 0, i = 0; i < qtdecroos; i++)
                                    {
                                        if(verbose) printf("\nExecucao %d / %d:\n", i+1, qtdecroos);
                                        errocrossVet[i] = utils_k_fold(sample, imadual, fold, i, verbose);
                                        errocross += errocrossVet[i];
                                        printf("Erro Execucao %d / %d: %.2lf%%\n", i+1, qtdecroos, errocrossVet[i]);
                                    }
                                    printf("\nErro Medio %d-Fold Cross Validation: %.2lf %c %.2lf\n", fold, errocross/qtdecroos, 241, data_standard_deviation(errocrossVet, qtdecroos));
                                    break;

                            case 3: //SMO
                                    printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                    scanf("%d", &kernel_type);
                                    if(kernel_type == 1)
                                        printf("Grau do polinomio: ");
                                    else if(kernel_type == 2)
                                        printf("Gamma do gaussiano: ");
                                    if(kernel_type != 0)
                                        scanf("%lf", &kernel_param);

                                    sample->q            = 2;
                                    sample->max_time     = max_time;
                                    sample->kernel_type  = kernel_type;
                                    sample->kernel_param = kernel_param;

                                    for(errocross = 0, i = 0; i < qtdecroos; i++)
                                    {
                                        if(verbose) printf("\nExecucao %d / %d:\n", i+1, qtdecroos);
                                        errocrossVet[i] = utils_k_fold(sample, smo_train, fold, i, verbose);
                                        errocross += errocrossVet[i];
                                        printf("Erro Execucao %d / %d: %.2lf%%\n", i+1, qtdecroos, errocrossVet[i]);
                                    }
                                    printf("\nErro Medio %d-Fold Cross Validation: %.2lf %c %.2lf\n", fold, errocross/qtdecroos, 241, data_standard_deviation(errocrossVet, qtdecroos));
                                    break;

                            case 4: //PL
                                    printf("Valor da norma q: ");
                                    scanf("%lf", &q);

                                    w = NULL;
                                    margin = 0;
                                    svs = 0;

                                    sample->q            = q;
                                    sample->kernel_type  = 9;

                                    for(errocross = 0, i = 0; i < qtdecroos; i++)
                                    {
                                        if(verbose) printf("\nExecucao %d / %d:\n", i+1, qtdecroos);
                                        errocrossVet[i] = utils_k_fold(sample, linear_programming, fold, i, verbose);
                                        errocross += errocrossVet[i];
                                        printf("Erro Execucao %d / %d: %.2lf%%\n", i+1, qtdecroos, errocrossVet[i]);
                                    }
                                    printf("\nErro Medio %d-Fold Cross Validation: %.2lf %c %.2lf\n", fold, errocross/qtdecroos, 241, data_standard_deviation(errocrossVet, qtdecroos));
                                    break;

                            default:puts("Opcao invalida.\n");
                                    break;
                        }
                        free(errocrossVet);
                    }
                    break;

            case 12://Informacoes
                    if(abrir_arquivo(&sample))
                    {
                        for(qtdpos = 0, qtdneg = 0, i = 0; i < sample->size; i++)
                            if(sample->points[i].y == 1)
                                qtdpos++;
                            else
                                qtdneg++;
                        printf("Pontos Pos. : %d\n", qtdpos);
                        printf("Pontos Neg. : %d\n", qtdneg);
                        printf("Pontos Total: %d\n", sample->size);
                        printf("Dimensao    : %d\n", sample->dim);
                        printf("Ver features? (0)Nao (1)Sim: ");
                        scanf("%d", &verfeat);
                        if(verfeat)
                            for(i = 0; i < sample->dim; i++)
                            {
                                printf("%4d ", sample->fnames[i]);
                                if((i+1) % 20 == 0)
                                    printf("\n");
                            }
                        printf("\n\n");

                        if(test_sample)
                        {
                            printf("Teste:\n");
                            for(qtdpos = 0, qtdneg = 0, i = 0; i < test_sample->size; i++)
                                if(test_sample->points[i].y == 1)
                                    qtdpos++;
                                else
                                    qtdneg++;
                            printf("Pontos Pos. : %d\n", qtdpos);
                            printf("Pontos Neg. : %d\n", qtdneg);
                            printf("Pontos Total: %d\n", test_sample->size);
                            printf("Dimensao    : %d\n", test_sample->dim);
                            printf("Ver features? (0)Nao (1)Sim: ");
                            scanf("%d", &verfeat);
                            if(verfeat)
                                for(i = 0; i < test_sample->dim; i++)
                                {
                                    printf("%4d ", test_sample->fnames[i]);
                                    if((i+1) % 20 == 0)
                                        printf("\n");
                                }
                            printf("\n\n");

                            if(test_sample->dim > sample->dim)
                            {
                                printf("Igualar features de teste as de treino? (1)Sim (0)Nao: ");
                                scanf("%d", &flag_feat);
                                if(flag_feat)
                                {
                                    data_remove_test_sample_features(sample, &test_sample, verbose);
                                    printf("Teste:\n");
                                    for(qtdpos = 0, qtdneg = 0, i = 0; i < test_sample->size; i++)
                                        if(test_sample->points[i].y == 1)
                                            qtdpos++;
                                        else
                                            qtdneg++;
                                    printf("Pontos Pos. : %d\n", qtdpos);
                                    printf("Pontos Neg. : %d\n", qtdneg);
                                    printf("Pontos Total: %d\n", test_sample->size);
                                    printf("Dimensao    : %d\n", test_sample->dim);
                                    printf("Ver features? (0)Nao (1)Sim: ");
                                    scanf("%d", &verfeat);
                                    if(verfeat)
                                        for(i = 0; i < test_sample->dim; i++)
                                        {
                                            printf("%4d ", test_sample->fnames[i]);
                                            if((i+1) % 20 == 0)
                                                printf("\n");
                                        }
                                    printf("\n\n");
                                }
                            }
                        }
                    }
                    break;

            case 13://Remover Features
                    if(!sample)
                        if(!data_load(arquivo_entrada, &sample, 0))
                        {
                            printf("Erro na leitura do arquivo.\n");
                            break;
                        }

                    printf("Remover quantas features: ");
                    scanf("%d", &totalfeat);
                    feats = (int*) malloc(totalfeat*sizeof(int));
                    if(feats == NULL) { printf("Erro de alocacao.\n"); exit(1); }
                    for(i = 0; i < totalfeat; i++)
                    {
                        printf("Feature %d: ", i+1);
                        scanf("%d", &feats[i]);
                        for(flag_feat = 0, j = 0; j < sample->dim; j++)
                            if(feats[i] == sample->fnames[j])
                                flag_feat = 1;
                        if(!flag_feat)
                        {
                            printf("Feature %d nao pertence ao conjunto.\n", feats[i]);
                            i--;
                        }
                    }
                    sample_temp = data_remove_features(sample, feats, totalfeat, 0);
                    data_free_sample(&sample);
                    sample = sample_temp;
                    sample_temp = NULL;
                    free(feats);
                    break;

            case 14://Adicionar Features
                    if(!sample)
                        if(!data_load(arquivo_entrada, &sample, 0))
                        {
                            printf("Erro na leitura do arquivo.\n");
                            break;
                        }

                    printf("Adicionar quantas features: ");
                    scanf("%d", &totalfeat);
                    feats = (int*) malloc(totalfeat*sizeof(int));
                    if(feats == NULL) { printf("Erro de alocacao.\n"); exit(1); }
                    for(i = 0; i < totalfeat; i++)
                    {
                        printf("Feature %d: ", i+1);
                        scanf("%d", &feats[i]);
                        for(flag_feat = 0, j = 0; j < sample->dim; j++)
                            if(feats[i] == sample->fnames[j])
                                flag_feat = 1;
                        if(!flag_feat)
                        {
                            printf("Feature %d nao pertence ao conjunto.\n", feats[i]);
                            i--;
                        }
                    }
                    sample_temp = data_insert_features(sample, feats, totalfeat, 0);
                    data_free_sample(&sample);
                    sample = sample_temp;
                    sample_temp = NULL;
                    free(feats);
                    break;

            /*
            case 14://normalizar
                    printf("Normalizar dados? (0) Nao (1) Sim: ");
                    scanf("%d", &normalizar);
                    if(!sample)
                        if(!data_load(arquivo_entrada, &sample, 0))
                        {
                            printf("Erro na leitura do arquivo.\n");
                            break;
                        }
                    if(normalizar) data_norm(sample);
                    break;
            */

            case 15://Validacao com K-Fold
                    if(!sample)
                        if(!data_load(arquivo_entrada, &sample, 0))
                        {
                            printf("Erro na leitura do arquivo.\n");
                            break;
                        }

                    printf("Quantidade de K-Fold: ");
                    scanf("%d", &qtdecroos);
                    //printf("K-Fold: ");
                    //scanf("%d", &fold);
                    fold = 10;

                    puts("1 - IMAp");
                    puts("2 - IMA Dual");
                    puts("3 - SMO");
                    scanf("%d", &op_fold);

                    if(!test_sample)
                        data_part_train_test(&sample, &test_sample, 3, 0, verbose);
                    else if(test_sample->dim > sample->dim)
                        data_remove_test_sample_features(sample, &test_sample, verbose);

                    switch(op_fold)
                    {
                        case 1: //IMAp
                                printf("Norma p (1) ou q (2): ");
                                scanf("%d", &norma);
                                if(norma == 1)
                                {
                                    printf("Valor da norma p: ");
                                    scanf("%lf", &p);
                                    if(p == 1.0)
                                        q = -1.0;
                                    else
                                        q = p/(p-1.0);
                                }
                                else
                                {
                                    printf("Valor da norma q: ");
                                    scanf("%lf", &q);
                                    if(q == 1.0)
                                        p = 100.0;
                                    else
                                        p = q/(q-1.0);
                                }

                                printf("Valor da flexibilizacao (0 - sem flexibilizacao): ");
                                scanf("%lf", &flexivel);

                                printf("Valor da aproximacao alfa (1 - alfa): ");
                                scanf("%lf", &alpha_prox);

                                w = NULL;
                                margin = 0;
                                svs = 0;

                                sample->p            = p;//2;
                                sample->q            = q;//2;
                                sample->max_time     = max_time;
                                sample->kernel_type  = 9;
                                sample->flexivel     = flexivel;
                                sample->alpha_aprox  = alpha_prox;

                                utils_validation(sample, test_sample, imap, fold, qtdecroos, verbose);
                                break;

                        case 2: //IMA Dual
                                printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                scanf("%d", &kernel_type);
                                if(kernel_type == 1)
                                    printf("Grau do polinomio: ");
                                else if(kernel_type == 2)
                                    printf("Gamma do gaussiano: ");
                                if(kernel_type != 0)
                                    scanf("%lf", &kernel_param);

                                sample->q            = 2;
                                sample->max_time     = max_time;
                                sample->kernel_type  = kernel_type;
                                sample->kernel_param = kernel_param;

                                utils_validation(sample, test_sample, imadual, fold, qtdecroos, verbose);
                                break;

                        case 3: //SMO
                                printf("Kernel (0)Produto Interno (1)Polinomial (2)Gaussiano: ");
                                scanf("%d", &kernel_type);
                                if(kernel_type == 1)
                                    printf("Grau do polinomio: ");
                                else if(kernel_type == 2)
                                    printf("Gamma do gaussiano: ");
                                if(kernel_type != 0)
                                    scanf("%lf", &kernel_param);

                                sample->q            = 2;
                                sample->max_time     = max_time;
                                sample->kernel_type  = kernel_type;
                                sample->kernel_param = kernel_param;

                                utils_validation(sample, test_sample, smo_train, fold, qtdecroos, verbose);
                                break;

                        default:puts("Opcao invalida.\n");
                                break;
                    }
                    //data_free_sample(&sample_fold);
                    break;

            case 16://Separar Base em Treino/Teste
                    if(!sample)
                        if(!data_load(arquivo_entrada, &sample, 0))
                        {
                            printf("Erro na leitura do arquivo.\n");
                            break;
                        }
                    if(!test_sample)
                    {
                        //printf("K-Fold: ");
                        //scanf("%d", &fold);
                        printf("Semente: ");
                        scanf("%d", &semente);
                        fold = 3;
                        data_part_train_test(&sample, &test_sample, fold, semente, verbose);
                    }
                    else
                        printf("A base ja esta separada.\n");
                    break;

            case 17://Salvar Bases de Treino/Teste
                    if(!sample)
                        if(!data_load(arquivo_entrada, &sample, 0))
                        {
                            printf("Erro na leitura do arquivo.\n");
                            break;
                        }
                    if(!test_sample)
                    {
                        //printf("K-Fold: ");
                        //scanf("%d", &fold);
                        fold = 3;
                        data_part_train_test(&sample, &test_sample, fold, 0, verbose);
                    }
                    strcpy(arquivo_saida, base_entrada);
                    char temp[5];
                    //temp[0] = (char)semente;
                    //temp[1] = '\0';
                    strcat(arquivo_saida, "_train_");
                    itoa(semente, temp, 10);
                    strcat(arquivo_saida, temp);
                    data_write(arquivo_saida, sample, 0);
                    strcpy(arquivo_saida, base_entrada);
                    strcat(arquivo_saida, "_test_");
                    strcat(arquivo_saida, temp);
                    data_write(arquivo_saida, test_sample, 0);
                    break;

            case 18: //ALMAp
                    if(abrir_arquivo(&sample))
                    {
                        printf("Valor da norma p: ");
                        scanf("%lf", &p);

                        printf("Valor da aproximacao alfa (1 - alfa): ");
                        scanf("%lf", &alpha_prox);

                        w = NULL;
                        margin = 0;
                        svs = 0;

                        sample->p            = p;
                        sample->max_time     = max_time;
                        sample->kernel_type  = 9;
                        sample->alpha_aprox  = alpha_prox;

                        if(almap(sample, &w, &margin, &svs, verbose))
                        {
                            printf("Treinamento com sucesso...\n");
                            printf("Margem = %lf, Vetores Suporte = %d\n\n", margin, svs);
                            if(sample->dim == 2 || sample->dim == 3)
                            {
                                printf("Deseja plotar o grafico? (0)Nao (1)Sim: ");
                                scanf("%d", &plot);
                                if(plot)
                                {
                                    if(sample->dim == 2)
                                        utils_plot_2d(sample, w, base_entrada, "almap");
                                    else
                                        utils_plot_3d(sample, w, base_entrada, "almap");
                                }
                            }
                        }
                        else
                            printf("Treinamento falhou.\n\n");
                        free(w);
                        free(sample->index);
                        sample->index = NULL;
                    }
                    break;

            case 19://Normalizar Base de Dados
                    if(abrir_arquivo(&sample))
                    {
                        printf("Valor da norma p: ");
                        scanf("%lf", &p);
                        sample->p = p;
                        if(!sample->normalized)
                            sample = data_normalize_database(sample);
                        else
                            printf("Base ja normalizada.\n");
                    }
                    break;

            case 20: //PL
                    if(abrir_arquivo(&sample))
                    {
                        printf("Valor da norma q: ");
                        scanf("%lf", &q);

                        w = NULL;
                        margin = 0;
                        svs = 0;

                        sample->q            = q;
                        sample->kernel_type  = 9;

                        if(linear_programming(sample, &w, &margin, &svs, verbose))
                        {
                            printf("Treinamento com sucesso...\n");
                            printf("Margem = %lf, Vetores Suporte = %d\n\n", margin, svs);
                            if(sample->dim == 2 || sample->dim == 3)
                            {
                                printf("Deseja plotar o grafico? (0)Nao (1)Sim: ");
                                scanf("%d", &plot);
                                if(plot)
                                {
                                    if(sample->dim == 2)
                                        utils_plot_2d(sample, w, base_entrada, "pl");
                                    else
                                        utils_plot_3d(sample, w, base_entrada, "pl");
                                }
                            }
                        }
                        else
                            printf("Treinamento falhou.\n\n");

                        printf("Deseja remover features com componentes nulas? (0)Nao (1)Sim: ");
                        scanf("%d", &removerFeat);
                        if(removerFeat)
                        {
                            for(totalfeat = 0, i = 0; i < sample->dim; i++)
                                if(w[i] == 0.0)
                                    totalfeat++;
                            feats = (int*) malloc(totalfeat*sizeof(int));
                            if(feats == NULL) { printf("Erro de alocacao.\n"); exit(1); }
                            for(i = 0, j = 0; j < sample->dim; j++)
                                if(w[j] == 0.0)
                                    feats[i++] = j+1;
                            sample_temp = data_remove_features(sample, feats, totalfeat, 0);
                            data_free_sample(&sample);
                            sample = sample_temp;
                            sample_temp = NULL;
                            strcat(arquivo_entrada, "_pl");
                            data_write(arquivo_entrada, sample, 0);
                            free(feats);
                        }
                        free(w);
                    }
                    break;

            case 0: break;

            default:puts("Opcao invalida.\n");
                    break;
        }
    } while(opcao != 0);
    data_free_sample(&sample);
    data_free_sample(&test_sample);
    data_free_sample(&sample_temp);
    return 0;
}
