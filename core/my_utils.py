import pandas as pd
import numpy as np
import pm4py
import core.my_filter as my_filter


MAGISTRADO = '1'
SERVENTUARIO = '14'


def get_movement_level(breadscrum, level_mag, level_ser, cur_level):
    breadscrum = breadscrum.split(':')

    if breadscrum[0] == MAGISTRADO:
        level = level_mag
    else:
        level = level_ser

    if len(breadscrum) > level:
        return int(breadscrum[level])
    else:
        return -1
        # return int(cur_level)


def create_dfgs_for_clusters(df_mov, cluster_labels, traces_perc):
    dfgs = {}

    for c in cluster_labels:
        df_temp = df_mov[df_mov['cluster_label'] == c]
        log = my_filter.filter_most_frequent(df_temp, 
                                   traces_perc=traces_perc)
        dfg_temp,sa,ea = pm4py.discover_dfg(log=log,
                                            activity_key='concept:name',
                                            timestamp_key='time:timestamp',
                                            case_id_key='case:concept:name',
                         )
        dfgs[c] = (dfg_temp,sa,ea)
    
    return dfgs


def get_total_traces_df_mov(df_mov):
    return len(df_mov.drop_duplicates(subset=['id']).index)


def map_movements(df_mov, 
                  df_code_mov, 
                  level_magistrado,
                  level_serventuario,
                  ):

    temp = df_code_mov[['movimentoCodigoNacional','breadscrum']]
    df_mov = df_mov.merge(temp, on=['movimentoCodigoNacional'], how='left')

    df_mov['movimentoCodigoNacional'] = df_mov.apply(lambda df: \
        get_movement_level(
            df['breadscrum'],
            level_magistrado,
            level_serventuario,
            df['movimentoCodigoNacional']), axis=1)


    df_temp = df_mov[df_mov['movimentoCodigoNacional'] == -1]

    df_mov = df_mov[~df_mov['id'].isin(df_temp['id'])]
    df_mov = df_mov.drop(columns=['breadscrum'])

    return df_mov


def map_mov_code_name(l, df_code_mov, n):
    print('### mapping move code to name')

    if n == 1:
        df_codes = pd.DataFrame(l, columns=['movimentoCodigoNacional'])
        df_codes = df_codes.merge(df_code_mov,
                                'left',
                                'movimentoCodigoNacional')
        df_codes = df_codes.drop(columns=['breadscrum'])
        df_codes = df_codes.set_index('movimentoCodigoNacional')


        return df_codes.to_dict()['movimentoNome']
    else:
        df_temp = df_code_mov[['movimentoCodigoNacional', 'movimentoNome']]
        df_temp = df_temp.set_index('movimentoCodigoNacional')
        
        dict_code_mov = df_temp.to_dict()['movimentoNome']

        map_mov_code = {}

        for c in l:
            temp = []
            for t in c:
                temp.append(dict_code_mov[t])
            map_mov_code[c] = tuple(temp)
        
        
        return map_mov_code


def get_act_name(df_mov, df_code_mov):
    df_temp = df_code_mov[['movimentoCodigoNacional', 'movimentoNome']]
    df_vis = df_mov.merge(df_temp, on='movimentoCodigoNacional', how='left')
    df_vis = df_vis.drop(columns=['movimentoCodigoNacional'])
    df_vis = df_vis.rename(columns={'movimentoNome':'movimentoCodigoNacional'})

    return df_vis



   