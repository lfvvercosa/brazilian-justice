import pandas as pd
import os
import numpy as np
import pm4py
import pickle
import matplotlib.pyplot as plt
from collections import Counter

import core.my_loader as my_loader
import core.my_filter as my_filter
import core.my_parser as my_parser
import core.my_utils as my_utils
import core.my_cluster as my_cluster
import core.my_model as my_model
import core.my_visual as my_visual
import core.my_stats as my_stats



def load_dataframes(my_justice, just_specs, base_path):
    df = my_loader.load_just_spec_df(just_specs, 
                                     base_path, 
                                     my_justice)
    df = my_parser.parse_data(df)

    df_assu = df[['id',
              'processoNumero',
              'assuntoCodigoNacional',
              'classeProcessual']].\
            explode('assuntoCodigoNacional')
    df_assu = df_assu[~df_assu['assuntoCodigoNacional'].isna()]

    df_mov = df.explode('movimento')
    df_mov = my_parser.parse_data_mov(df_mov)


    return df, df_assu, df_mov


def pre_process( 
                df_assu, 
                df_mov,
                df_code_assu,
                df_code_cla,
                df_code_mov,
                ):

    ## Removing null traces

    print('### Total traces start:')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = my_filter.filter_null_traces(df_mov)

    print('### Total traces after removing null')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Removing invalid traces

    df_mov = my_filter.filter_invalid_traces(df_mov,
                                            df_assu,
                                            df_code_assu,
                                            df_code_cla,
                                            df_code_mov)

    print('### Total traces after removing invalid')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Map movements

    level_magistrado = 1
    level_serventuario = 2

    df_mov = my_utils.map_movements(df_mov, 
                                    df_code_mov, 
                                    level_magistrado,
                                    level_serventuario)


    print('### Total traces after mapping movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = df_mov.sort_values(['id','movimentoDataHora'])
    df_mov = df_mov.reset_index(drop=True)

    ## Remove traces with single activity

    print('### Total traces before filter single movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = my_filter.filter_single_act_traces(df_mov)

    print('### Total traces after filter single movements')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Apply TF-IDF (Remove activities that appear too rarely)

    min_perc = 0.05

    rem_cols = my_cluster.infreq_toofreq_cols(df_mov, min_perc, None)

    # Also remove meaningless activities:
    #    - 'Expedição de Documento' (60)
    #    - 'Ato Ordinário' (11383)

    if 60 not in rem_cols:
        rem_cols.append(60)

    if 11383 not in rem_cols:
        rem_cols.append(11383)

    df_mov = my_filter.rem_spec_act(df_mov, 
                                    rem_act=rem_cols,
                                    act_col='movimentoCodigoNacional',
                                   )

    ## Remove repeated activities in sequence (autoloops)

    df_mov = my_filter.rem_autoloop2(df_mov, 
                                    id_col='id', 
                                    act_col='movimentoCodigoNacional')

    ## Remove traces with rare start or end activities 
    ## (possibly didnt yet finish or had already started)

    print('### Total traces after removing specific activities' + \
          ' and autoloop')
    print(my_utils.get_total_traces_df_mov(df_mov))

    df_mov = my_filter.filter_non_complete_traces(df_mov, 0.01)

    print('### Total traces after filtering non-complete')
    print(my_utils.get_total_traces_df_mov(df_mov))

    ## Remove traces with outlier duration

    df_mov = my_filter.filter_outlier_duration_trace(df_mov, 0.05, 0.95)

    print('### Total traces after filtering outlier duration')
    print(my_utils.get_total_traces_df_mov(df_mov))

    
    return df_mov


def create_1_gram(df_mov, df_code_mov):
    ## Create n-gram
    n = 1
    id_col = 'id'
    activity_col = 'movimentoCodigoNacional'

    df_gram = my_cluster.create_n_gram(df_mov, 
                                       id_col, 
                                       activity_col,
                                       n)

    print('### Total n-gram features: ' + str(len(df_gram.columns)))

    # Winsorize
    min_perc = 0.05
    max_perc = 0.05
    df_gram = my_filter.winsorize_df(df_gram, min_perc, max_perc)

    # Normalize it
    df_gram_norm = (df_gram - df_gram.min())/\
                   (df_gram.max() - df_gram.min())
    df_gram_norm = df_gram_norm.round(4)

    rename_cols = my_utils.map_mov_code_name(
                        list(df_gram_norm.columns), 
                        df_code_mov,
                        n)
    df_gram_norm = df_gram_norm.rename(columns=rename_cols)


    return df_gram_norm


def cluster_dbscan(df_gram):
    min_pts = 40
    eps = 0.95

    print('#### number of features: ' + str(len(df_gram.columns)))

    cluster_labels = my_cluster.cluster_dbscan(df_gram, eps, min_pts)

    cluster_count = Counter(cluster_labels)

    print('### clusters: ' + str(cluster_count))


    return cluster_labels


def add_cluster_labels_to_dfs(cluster_labels,
                             df_1_gram,
                             df,
                             df_mov,
                             df_assu,
                             df2,
                             df_mov2,
                             df_assu2):

    df_1_gram['cluster_label'] = cluster_labels

    df_clu = df_1_gram[['cluster_label']]
    df_clu['id'] = df_clu.index
    df_clu = df_clu.reset_index(drop=True)

    df = df.merge(df_clu, on='id',how='left')
    df = df[~df['cluster_label'].isna()]
    
    df_mov = df_mov.merge(df_clu, on='id',how='left')

    df_assu = df_assu.merge(df_clu, on='id',how='left')
    df_assu = df_assu[~df_assu['cluster_label'].isna()]
    df_assu = df_assu.set_index('id', drop=False)

    df2 = df2.merge(df_clu, on='id',how='left')
    df2 = df2[~df2['cluster_label'].isna()]

    df_mov2 = df_mov2.merge(df_clu, on='id',how='left')

    df_assu2 = df_assu2.merge(df_clu, on='id',how='left')
    df_assu2 = df_assu2[~df_assu2['cluster_label'].isna()]
    df_assu2 = df_assu2.set_index('id', drop=False)


    return df_1_gram, df, df_mov, df_assu, df2, df_mov2, df_assu2


def get_top10_clusters(cluster_labels):
    cluster_labels_no_outlier = [x for x in cluster_labels if x != -1]
    labels_counter = Counter(cluster_labels_no_outlier)
    
    
    return labels_counter.most_common(10)


def get_most_frequent_attribute(df_attr, 
                                df_code, 
                                clusters, 
                                level, 
                                code_col, 
                                name_col):
    
    df_traces = df_attr[df_attr['cluster_label'].isin(clusters)]
    df_traces = df_traces[[code_col]]


    return my_stats.\
        get_attribute_traces(df_code, 
                             df_traces, 
                             level, 
                             code_col, 
                             name_col)


def cluster_agglomerative(df_gram, 
                          df, 
                          common_classes,
                          common_clusters):

    df_gram_cent = my_filter.\
                    filter_df_gram_class_clus(df_gram, 
                                              df, 
                                              common_classes,
                                              common_clusters)
    df_cent = my_cluster.get_centroids(df_gram_cent,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                       None, 
                                       None,
                                       False)
    df_cent = df_cent.drop(index='cluster_label')
    df_dendro = df_cent.T

    my_visual.visualize_dendrogram(df_dendro, 
                                    2.4,
                                    None
                                   )


def generate_centroids_heatmap(df_gram, 
                               df, 
                               common_classes,
                               common_clusters,
                               key_activities):
    df_gram_cent = my_filter.\
                    filter_df_gram_class_clus(df_gram, 
                                              df, 
                                              common_classes,
                                              common_clusters)
    df_cent = my_cluster.get_centroids(df_gram_cent,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                       None, 
                                       None,
                                       False)

    df_cent = df_cent.loc[key_activities]
    df_cent = df_cent.round(1)

    cluster_order = [0,39,26,67,31,29,32,37,30,40]
    cluster_order = ['cluster_' + str(c) for c in cluster_order]

    df_cent = df_cent[cluster_order]

    # df_cent = df_cent.drop(index='cluster_label')
    my_visual.gen_heatmap(df_cent, None, None, None)


def get_clusters_subject(clusters, 
                         df_subj, 
                         df_code, 
                         level, 
                         code_col, 
                         name_col,
                         types):

    ## Get cluster subjects
    df_traces = df_subj[df_subj['classeProcessual'].isin(types)]
    df_traces = df_traces[df_traces['cluster_label'].isin(clusters)]
    df_traces = df_traces[[code_col, 'cluster_label']]

    for c in clusters:
        df_temp = df_traces[df_traces['cluster_label'] == c]
        df_temp = df_temp[[code_col]]

        df_res = my_stats.\
                    get_attribute_traces(df_code, 
                                        df_traces, 
                                        level, 
                                        code_col, 
                                        name_col)

        print('### Subjects Cluster ' + str(c))
        print()
        print(df_res)
        print()


def get_clusters_dfgs_no_processing(
                      df_mov,
                      df_code_mov,
                      clusters,
                      lawsuit_type,
                      noise_thres,
                      act_code_rem,
                      saving_path
                     ):
    
    df_vis = df_mov.copy(deep=True)

    if act_code_rem:
        df_vis = my_filter.\
                    rem_spec_act(df_vis, 
                                 rem_act=act_code_rem,
                                 act_col='movimentoCodigoNacional')
    
    if lawsuit_type:
        df_vis = df_vis[df_vis['classeProcessual'].isin(lawsuit_type)]

    if clusters:
        df_vis = df_vis[df_vis['cluster_label'].isin(clusters)]

        log = my_visual.convert_df_to_log(df_vis, df_code_mov)
        dfg_freq, sa_freq, ea_freq = \
            my_model.create_dfg(log,performance=False)

        dfg_freq, sa_freq, ea_freq = \
                        my_model.rem_noise_dfg_with_excep(
                                        dfg_freq,
                                        sa_freq,
                                        ea_freq,
                                        noise_thres)

        if saving_path:
            isExist = os.path.exists(saving_path)
            
            if not isExist:
                os.makedirs(saving_path)
                
            pm4py.save_vis_dfg(dfg_freq, sa_freq, ea_freq, saving_path + \
                'no_processing.png')
        else:
            pm4py.view_dfg(dfg_freq, sa_freq, ea_freq)
            plt.show(block=True)
            plt.close()


def get_clusters_dfgs(df_mov, 
                      df_code_mov_eng,
                      clusters,
                      lawsuit_type,
                      noise_thres,
                      act_code_rem,
                      saving_path
                     ):
    
    df_vis = df_mov.copy(deep=True)

    if act_code_rem:
        df_vis = my_filter.\
                    rem_spec_act(df_vis, 
                                 rem_act=act_code_rem,
                                 act_col='movimentoCodigoNacional')
    
    if lawsuit_type:
        df_vis = df_vis[df_vis['classeProcessual'].isin(lawsuit_type)]

    df_temp = df_vis.copy(deep=True)

    for c in clusters:
        df_vis = df_temp[df_temp['cluster_label'] == c]
        log = my_visual.convert_df_to_log(df_vis, df_code_mov_eng)
        dfg_freq, sa_freq, ea_freq = \
            my_model.create_dfg(log,performance=False)

        dfg_freq, sa_freq, ea_freq = \
                        my_model.rem_noise_dfg_with_excep(
                                        dfg_freq,
                                        sa_freq,
                                        ea_freq,
                                        noise_thres)

        if saving_path:
            isExist = os.path.exists(saving_path)
            
            if not isExist:
                os.makedirs(saving_path)

            pm4py.save_vis_dfg(dfg_freq, sa_freq, ea_freq, saving_path + \
                str(c) + '.png')
        else:
            pm4py.view_dfg(dfg_freq, sa_freq, ea_freq)
            plt.show(block=True)
            plt.close()