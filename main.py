from my_statistics.get_statistics_all import generate_statistics
from core import my_orchestrator
import core.my_loader as my_loader

from collections import Counter
import pickle

if __name__ == "__main__":

    base_path = 'dataset/'
    statistics_path = 'statistics/results'
    
    # reference tables
    df_code_subj = my_loader.load_df_subject(base_path)
    df_code_type = my_loader.load_df_classes(base_path)
    df_code_mov = my_loader.load_df_movements(base_path)
    df_code_mov_eng = my_loader.load_df_movements_eng()

    my_justice = 'TRIBUNAIS_SUPERIORES'
    # my_justice = 'JUSTICA_FEDERAL'
    # my_justice = 'JUSTICA_MILITAR'
    # my_justice = 'JUSTICA_ELEITORAL'
    # my_justice = 'JUSTICA_TRABALHO'
    # my_justice = 'JUSTICA_ESTADUAL'

    ####################################################
    ############## FULL DATASET STATISTICS #############
    ####################################################

    # generate_statistics(my_justice, statistics_path)

    ####################################################
    ################## LOAD DATAFRAMES #################
    ####################################################

    df, df_assu, df_mov = \
        my_orchestrator.load_dataframes('TRIBUNAIS_SUPERIORES',
                                        ['STJ'],
                                        base_path)

    df_copy = df.copy(deep=True)
    df_assu_copy = df_assu.copy(deep=True)
    df_mov_copy = df_mov.copy(deep=True)
    
    ####################################################
    ################## PREPROCESSING ###################
    ####################################################

    df_mov_copy = my_orchestrator.pre_process(
                df_assu_copy,
                df_mov_copy,
                df_code_subj,
                df_code_type,
                df_code_mov
            )

    ####################################################
    ################ FEATURES CREATION #################
    ####################################################

    df_1_gram = my_orchestrator.create_1_gram(df_mov_copy,
                                              df_code_mov)

    ####################################################
    ################ DBSCAN CLUSTERING #################
    ####################################################

    cluster_labels = my_orchestrator.cluster_dbscan(df_1_gram)

    df_1_gram, df_copy, df_mov_copy, df_assu_copy, \
    df, df_mov, df_assu = my_orchestrator.\
        add_cluster_labels_to_dfs(cluster_labels,
                                  df_1_gram,
                                  df_copy,
                                  df_mov_copy,
                                  df_assu_copy,
                                  df,
                                  df_mov,
                                  df_assu)

    ####################################################
    ############ AGGLOMERATIVE CLUSTERING ##############
    ####################################################

    print('get 10 most frequent clusters')

    common_10 = my_orchestrator.get_top10_clusters(cluster_labels)

    for c in common_10:
        print('cluster id: ' + str(c[0]) + ', records: ' + str(c[1]))
    
    print('get most frequent lawsuit type in top 10 clusters')

    common_10 = [x[0] for x in common_10]
    code_col = 'classeProcessual'
    name_col = 'classeNome'
    level = 1
    most_freq_lawsuit_type = my_orchestrator.\
                                get_most_frequent_attribute(
                                                    df_copy, 
                                                    df_code_type, 
                                                    common_10, 
                                                    None, 
                                                    code_col, 
                                                    name_col)

    print(most_freq_lawsuit_type)

    print('filter traces containing four most recurring types for ' + \
          'agglomerative clustering')

    print('visualize dendrogram with cutting point defined as 2.4')

    my_orchestrator.cluster_agglomerative(df_1_gram, 
                                          df_copy, 
                                          [1720, 1032, 11881, 985],
                                          common_10)

    ####################################################
    ################ CENTROIDS HEATMAP #################
    ####################################################

    key_activities = [
        'Baixa Definitiva_22',
        'Distribuição_26',
        'Redistribuição_36',
        'Conclusão_51',
        'Juntada_67',
        'Publicação_92',
        'Protocolo de Petição_118',
        'Remessa_123',
        'Julgamento_193',
        'Trânsito em julgado_848',
        'Arquivamento_861',
        'Guarda Permanente_867',
        'Despacho_11009',
    ]

    my_orchestrator.generate_centroids_heatmap(df_1_gram, 
                                               df_copy, 
                                               [1720, 1032, 11881, 985],
                                               common_10,
                                               key_activities)

    ####################################################
    ######## SUBJECTS SPECIAL APPEAL CLUSTERS  #########
    ####################################################

    code_col = 'assuntoCodigoNacional'
    name_col = 'assuntoNome'
    level = 0

    print('get most frequent lawsuit subjects \n')

    my_orchestrator.get_clusters_subject(
                         [37, 30, 40], 
                         df_assu_copy, 
                         df_code_subj, 
                         level, 
                         code_col,
                         name_col, 
                         [1720, 1032, 11881, 985])

    print('done!')

    ####################################################
    ########## DFGs SPECIAL APPEAL CLUSTERS  ###########
    ####################################################

    

    noise_thres = 0.3
    lawsuit_type = [11881,1032]
    clusters = [37,30,40]
    saving_path = 'images/'

    all_acts = \
        df_mov_copy.drop_duplicates('movimentoCodigoNacional')\
        ['movimentoCodigoNacional'].to_list()

    key_acts_code = [int(f[f.find('_')+1:]) for f in key_activities]
    remove_acts = [f for f in all_acts if f not in key_acts_code]

    print('show image special appeal DFGs \n')

    my_orchestrator.get_clusters_dfgs(
            df_mov_copy,
            df_code_mov_eng,
            clusters,
            lawsuit_type,
            noise_thres,
            remove_acts,
            saving_path=None
    )

    print('show image appeal clusters DFG with no processing')

    my_orchestrator.get_clusters_dfgs_no_processing(
            df_mov,
            df_code_mov,
            clusters,
            lawsuit_type,
            noise_thres=0,
            act_code_rem=None,
            saving_path=None
    )

