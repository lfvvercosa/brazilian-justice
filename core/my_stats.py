import pandas as pd
import numpy as np
from collections import Counter


def is_in_previous_loop(pos, loop_acts):
    for p in pos:
        if p in loop_acts:
            return True
    
    return False


def loops_by_trace(l, only_consecutive = False):
    size = len(l)
    count_loops = 0
    loop_acts = []
   

    for i in range(size):
        loop = [l[i]]

        for j in range(i+1,size):
            k = 0
            pos = []

            while j + k < size and \
                  k < len(loop) and \
                  l[j + k] == loop[k]:
                        pos.append(j + k)
                        k += 1

            if only_consecutive:
                if len(pos) == len(loop):
                    if not is_in_previous_loop(pos, loop_acts):
                        count_loops += 1
                        loop_acts += pos
            elif len(pos) > 0:
                if not is_in_previous_loop(pos, loop_acts):
                    count_loops += 1
                    loop_acts += pos

            if j + k < size:
                loop.append(l[j + k])


    return count_loops


def get_loops_by_cluster(df_mov, id_col, act_col, c1, c2, only_consec):
    print('### get_loops_by_cluster')
    
    # df_work = df_mov.groupby(id_col).agg(act)

    df_work = df_mov.groupby(id_col).agg({act_col:list, 'cluster_label':min})
    df_work['loop_count'] = df_work.\
                            apply(lambda df_work: loops_by_trace(
                                                    df_work[act_col],
                                                    only_consec
                                                                ), axis=1
                                 )
    
    s_c1 = df_work[df_work['cluster_label'].isin(c1[1])]['loop_count']
    s_c2 = df_work[df_work['cluster_label'].isin(c2[1])]['loop_count']


    return {'' + str(c1[0]) : s_c1, 
            '' + str(c2[0]) : s_c2}


def get_traces_duration(df_mov):
    df_work = df_mov.groupby('id').\
        agg({'movimentoDataHora':['min','max']})
    df_work['duration'] = (df_work[('movimentoDataHora', 'max')] - \
                           df_work[('movimentoDataHora', 'min')])\
                           / np.timedelta64(1, 'M')

    
    return df_work[['duration']]


def get_number_of_movs(df_gram, 
                       df, 
                       class_list, 
                       thrs_dur_min,
                       thrs_dur_max, 
                       clus_list):
    cols = list(df_gram.columns)
    cols.remove('cluster_label')

    df_sum = df_gram[cols]
    df_sum = df_sum.sum(axis=1).to_frame(name='total_movs')
    df_sum = df_sum.join(df[['classeProcessual','duration','cluster_label']])
    df_sum = df_sum[df_sum['classeProcessual'].isin(class_list)]
    df_sum = df_sum[df_sum['cluster_label'].isin(clus_list)]
    
    if thrs_dur_min:
        df_sum = df_sum[df_sum['duration'] > thrs_dur_min]

    if thrs_dur_max:
        df_sum = df_sum[df_sum['duration'] < thrs_dur_max]
    
    return df_sum['total_movs']


def get_class_in_clusters(df, my_classes, clusters):
    df_work = df[df['cluster_label'].isin(clusters)]
    # df_work = df_work[df_work['classeProcessual'].isin(my_classes)]

    df_work = df_work.groupby(['classeProcessual'])['processoNumero'].count()

    df_work = df_work.sort_values()

                            
    print(df_work.groupby(['classeProcessual','cluster_label'])\
                          ['processoNumero'].count())

    print()


def get_most_freq_val(s, n):
    return s.value_counts()[:n].index.tolist()


def get_percent_traces_by_clusters(perc_cluster, cluster_labels):
    new_cluster_labels = [c for c in cluster_labels if c != -1]
    cluster_count = Counter(new_cluster_labels)
    most_common = cluster_count.most_common()

    n = int(perc_cluster * len(most_common))
    total_n = 0
    total = 0

    for i in range(len(most_common)):
        if i < n:
            total_n += most_common[i][1]
        
        total += most_common[i][1]

    print('percent clusters: ' + str(perc_cluster))
    print('number of clusters: ' + str(n))
    print('percent traces: ' + str(total_n/total))
    print()


def get_attribute_level(breadscrum, desired_level):
    breadscrum = breadscrum.split(':')

    if len(breadscrum) > desired_level:
        return int(breadscrum[desired_level])
    else:
        return -1


def get_attribute_traces(df_code, df_traces, level, code_col, name_col):
    
    temp = df_code[[code_col,'breadscrum']]
    df_traces = df_traces.merge(temp, on=[code_col], how='left')

    if level is not None:
        df_traces['code_new_level'] = df_traces.apply(lambda df: \
            get_attribute_level(
                df['breadscrum'],
                level), axis=1)
    else:
         df_traces['code_new_level'] = df_traces[code_col]
    
    df_traces = df_traces.groupby('code_new_level').\
                    agg(count=(code_col, 'count'))

    df_traces = df_traces.sort_values('count', ascending=False)
    df_traces[code_col] = df_traces.index 

    temp = df_code[[code_col,name_col]]
    df_traces = df_traces.merge(temp, on=[code_col], how='left')

    total = df_traces['count'].sum()
    df_traces['Percent'] = (df_traces['count']/total).round(2)


    return df_traces[[name_col, code_col, 'Percent']]