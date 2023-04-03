import datetime
import pandas as pd
import numpy as np
import pm4py

from scipy.stats.mstats import winsorize
from contextlib import redirect_stdout

from core import my_stats


def filter_subject_exclusive(df, target_subj, df_assu):
    df_remove = df_assu[~df_assu['assuntoCodigoNacional'].isin(target_subj)]
    df = df[~df['id'].isin(df_remove['id'])]

    return df


def filter_subject_inclusive(df, target_subj, df_assu):
    df_keep = df_assu[df_assu['assuntoCodigoNacional'].isin(target_subj)]
    df = df[df['id'].isin(df_keep['id'])]

    return df


def find_invalid_assu(df_assu, df_code_assu):
    invalid_assu = df_assu[~df_assu['assuntoCodigoNacional'].\
            isin(df_code_assu['assuntoCodigoNacional'])]
    
    return invalid_assu[['id']]


def find_invalid_class(df_assu, df_code_class):
    invalid_class = df_assu[~df_assu['classeProcessual'].\
        isin(df_code_class['classeProcessual'])]

    return invalid_class[['id']]


def find_invalid_mov(df_mov, df_code_mov):
    invalid_mov = df_mov[(~df_mov['movimentoCodigoNacional'].isna()) & \
              (~df_mov['movimentoDataHora'].isna())]
    invalid_mov = invalid_mov[(invalid_mov['movimentoDataHora'] < \
                datetime.datetime(year=1950, month=1, day=1)) | \
                (invalid_mov['movimentoDataHora'] > \
                datetime.datetime(year=2022, month=1, day=1)) | \
                (~invalid_mov['movimentoCodigoNacional'].\
                    isin(df_code_mov['movimentoCodigoNacional']))]\
              

    return invalid_mov[['id']]


def filter_invalid_traces(df_mov,
                          df_assu,
                          df_code_assu, 
                          df_code_class,
                          df_code_mov):
    invalid_assu = find_invalid_assu(df_assu, df_code_assu)
    invalid_class = find_invalid_class(df_assu, df_code_class)
    invalid_mov = find_invalid_mov(df_mov, df_code_mov)

    invalid_traces = pd.concat([invalid_assu, invalid_class])
    invalid_traces = pd.concat([invalid_traces, invalid_mov])

    df_mov = df_mov[~df_mov['id'].isin(invalid_traces['id'])]


    return df_mov


def filter_null_traces(df_mov):
    null_traces = df_mov[(df_mov['processoNumero'].isna()) | \
                         (df_mov['movimentoCodigoNacional'].isna()) | \
                         (df_mov['movimentoDataHora'].isna()) | \
                         (df_mov['orgaoJulgadorCodigo'].isna())]

    df_mov = df_mov[~df_mov['id'].isin(null_traces['id'])]

    return df_mov


def filter_activities(df, act_perc):
    df_temp = df['movimentoCodigoNacional'].\
                value_counts().\
                sort_values(ascending=False).\
                to_frame(name='count')
    df_temp['ranking'] = np.arange(len(df_temp))
    
    threshold = int(act_perc * len(df_temp))
    df_temp = df_temp[df_temp['ranking'] < threshold]
    df_temp['movimentoCodigoNacional'] = df_temp.index
    df_temp = df_temp.drop(columns=['ranking'])

    df = df.merge(df_temp, on='movimentoCodigoNacional', how='left')
    df = df[~df['count'].isna()].drop(columns=['count'])

    return df


def filter_class_subject(df,
                         df_assu,
                         target_class, 
                         target_subj, 
                         is_only_subject):

    print('### filtering class and subject...')

    df = df[df['classeProcessual'].isin(target_class)]

    if is_only_subject:
        func = filter_subject_exclusive
    else:
        func = filter_subject_inclusive

    df = func(df, target_subj, df_assu)

    return df


def filter_by_code(df_mov, code):
    df_rem = df_mov[df_mov['movimentoCodigoNacional'] == code]
    df_mov = df_mov[~df_mov['id'].isin(df_rem['id'])]

    return df_mov


def filter_most_frequent_aux(log, total_variants, traces_perc):
    variants = pm4py.get_variants(log)
    variants_ordered = {}
    variants_allowed = []
    threshold = int(traces_perc * total_variants)
    count = 0

    for k in variants:
        variants_ordered[k] = len(variants[k])
    
    variants_ordered = {k: v for k, v in sorted(variants_ordered.items(), 
                                                key=lambda item: item[1],
                                                reverse=True)}

    for k in variants_ordered:
        with open('out.txt', 'a') as f:
            with redirect_stdout(f):
                print('variant: ' + str(k))
                print('freq:' + str(variants_ordered[k]))
                print('')

        if count < threshold:
            variants_allowed.append(k)
        else:
            break

        count += 1

    return variants_allowed


def filter_most_frequent(df, traces_perc):
    df = pm4py.utils.format_dataframe(df, 
                                case_id='id', 
                                activity_key='movimentoCodigoNacional', 
                                timestamp_key='movimentoDataHora')
    df = df[['case:concept:name','concept:name','time:timestamp']]
    log = pm4py.convert_to_event_log(df)
    total_variants = len(pm4py.get_variants(log))

    variants_allowed = filter_most_frequent_aux(log, total_variants, traces_perc)

    log = pm4py.filter_variants(log, 
                                variants=variants_allowed, 
                                retain=True)

       
    return log


def filter_autoloop(df_work, act_col):
    df_work = df_work.apply(filter_autoloop_aux).to_frame(name='enum_act')
    df_work = df_work.explode('enum_act')
    df_work[act_col] = df_work['enum_act'].str[0]
    df_work['enum_mov'] = df_work['enum_act'].str[1]
    df_work = df_work.drop(columns='enum_act')

    return df_work


def filter_autoloop_aux(l):
    count = 0
    m = []

    for idx, val in enumerate(l):
        if idx - 1 >= 0:
            if l[idx - 1] != l[idx]:
                count += 1
        m.append((val, count))


    return m


def rem_autoloop(df, id_col, act_col):
    print('### removing autoloop')

    df_work = df.groupby(id_col)[act_col].apply(list)
    df_work = filter_autoloop(df_work, act_col)
    df = df.sort_values(['id','movimentoDataHora'])
    df.index = df['id']
    df['enum_mov'] = df_work['enum_mov']
    df = df.drop_duplicates(subset=[id_col, act_col, 'enum_mov'])
    df = df.drop(columns=['enum_mov'])
    df = df.reset_index(drop=True)

    return df


def filter_autoloop2(df_work):
    df_work = df_work.apply(filter_autoloop_aux2).to_frame(name='keep')
    df_work = df_work.explode('keep')

    return df_work


def filter_autoloop_aux2(l):
    size_l = len(l)
    m = [True] * size_l

    if size_l > 0:
        for i in range(1,size_l):
            if l[i - 1] == l[i]: 
                m[i - 1] = False

    if size_l > 1:
        if l[1] == l[0]:
            m[1] = False
        if l[-1] == l[-2]:
            m[-2] = False  

    m[0] = True
    m[-1] = True


    return m


def rem_autoloop2(df, id_col, act_col):
    df_work = df.groupby(id_col)[act_col].apply(list)
    df_work = filter_autoloop2(df_work)
    
    df = df.sort_values(['id','movimentoDataHora'])
    df.index = df['id']
    df['keep'] = df_work['keep']
    df = df[df['keep'] == True]

    df = df.drop(columns=['keep'])
    df = df.reset_index(drop=True)

    return df


def rem_loop(df_mov, id_col, act_col, only_consec):
    print('### removing loops')

    df_work = df_mov.groupby(id_col)[act_col].apply(list)
    df_work = filter_loop(df_work, act_col, only_consec)

    df_mov = df_mov.sort_values(['id','movimentoDataHora'])
    df_mov.index = df_mov['id']
    df_mov['bool'] = df_work['bool']

    df_temp = df_mov[['id','bool']]
    df_temp = df_temp[df_temp['bool'] == False]
    df_temp = df_temp.drop_duplicates(['id'])

    print('### removing loops from ' + str(len(df_temp)) + ' instances')

    df_mov = df_mov[df_mov['bool'] == True]

    df_mov = df_mov.drop(columns=['bool'])
    df_mov = df_mov.reset_index(drop=True)


    return df_mov


def filter_loop(df_work, act_col, only_consec):
    df_work = df_work.apply(
                lambda df_work: filter_loop_aux(df_work, only_consec)
              ).to_frame(name='bool_act')
                
    df_work = df_work.explode('bool_act')
    df_work[act_col] = df_work['bool_act'].str[0]
    df_work['bool'] = df_work['bool_act'].str[1]
    df_work = df_work.drop(columns='bool_act')


    return df_work


def filter_loop_aux(l, only_consecutive = False):
    size = len(l)
    lf = [True]*size

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
                    set_false(lf, pos)
            else:
                set_false(lf, pos)

            if j + k < size:
                loop.append(l[j + k])


    return list(zip(l,lf))


def set_false(lf, pos):
    for p in pos:
        lf[p] = False

    return lf

def rem_spec_act(df, rem_act, act_col):
    print('### removing specific list of activities...')

    df_rem = pd.DataFrame(rem_act, columns=[act_col])
    df_rem['is_remove'] = 1

    df = df.merge(df_rem, on=act_col, how='left')
    df = df[df['is_remove'] != 1]
    df = df.drop(columns=['is_remove'])


    return df


def identify_start_end_last_aux(l, rem_act):
    size_l = len(l)
    count = 0

    if size_l > 0:
        m = [None] * size_l
        m[0] = 'Start'
        m[-1] = 'End'
    
    while l[count] in rem_act:
        count += 1

        if len(l) - 1 <= count:
            break 
    
    if count != 0:
        m[count - 1] = 'Start Last'

    count = -1

    while l[count] in rem_act:
        count -= 1

        if len(l) - 1 < abs(count):
            break

    if count != -1:
        m[count + 1] = 'End Last'

    return m


def identify_start_end_last(df_work, rem_act):
    df_work = df_work.apply(lambda df_work: identify_start_end_last_aux(
                                df_work,
                                rem_act,
                           )).to_frame(name='keep')
    df_work = df_work.explode('keep')

    return df_work


def rem_spec_act2(df, rem_act, act_col, id_col):
    df_work = df.groupby(id_col)[act_col].apply(list)
    df_work = identify_start_end_last(df_work, rem_act)
    
    df_rem = pd.DataFrame(rem_act, columns=[act_col])
    df_rem['is_remove'] = 1

    df = df.merge(df_rem, on=act_col, how='left')

    # df = df.sort_values(['id','movimentoDataHora'])
    df.index = df['id']
    # df = df.set_index(id_col, drop=False)

    df['keep'] = df_work['keep']

    df = df[(df['is_remove'] != 1) | (~df['keep'].isna())]

    df = df.drop(columns=['keep', 'is_remove'])
    df = df.reset_index(drop=True)


    return df


def rem_unfrequent(s, percent):
    size = len(s)
    count = int(size * percent)

    s_include = s.groupby(s).filter(lambda x: len(x) > count)


    return s[s.isin(s_include)]


def group_unfrequent(s, percent):
    s = s.astype("string")
    freq = s.value_counts(normalize=True)
    mapping = s.map(freq)
    s = s.mask(mapping < percent, 'Other')

    
    return s


def winsorize_df(df, min_perc, max_perc):
    for c in df.columns:
        temp = winsorize(df[c], (min_perc,max_perc))
        
        if temp.min() != temp.max():
            df[c] = temp

    return df


def filter_single_act_traces(df_mov):
    df_work = df_mov.groupby('id').\
                        agg(count=('movimentoCodigoNacional','count'))
    df_work = df_work[df_work['count'] == 1]

    df_mov = df_mov[~df_mov['id'].isin(df_work.index)]


    return df_mov


def filter_non_complete_traces(df_mov, thrs):
    df_temp = df_mov.groupby('id').agg(
                        mov_first=('movimentoCodigoNacional','first'),
                        mov_last=('movimentoCodigoNacional','last')
                    )
    total = len(df_temp)
    min_val = int(thrs * total)

    df_first = df_temp.groupby('mov_first').\
                        agg(count=('mov_last','count')).\
                        sort_values('count')
    df_first = df_first[df_first['count'] >= min_val]
    mov_first = df_first.index.to_list()

    df_last = df_temp.groupby('mov_last').\
                        agg(count=('mov_first','count')).\
                        sort_values('count')
    df_last = df_last[df_last['count'] >= min_val]
    mov_last = df_last.index.to_list()

    df_temp = df_temp[(df_temp['mov_first'].isin(mov_first)) & 
                      (df_temp['mov_last'].isin(mov_last))]

    df_mov = df_mov[df_mov['id'].isin(df_temp.index)]


    return df_mov


def filter_outlier_duration_trace(df_mov, quant_min, quant_max):
    df_dur = my_stats.get_traces_duration(df_mov)
    val_min = df_dur.quantile(quant_min)[0]
    val_max = df_dur.quantile(quant_max)[0]
    df_dur = df_dur[(df_dur['duration'] >= val_min) & \
                    (df_dur['duration'] <= val_max)]

    df_mov = df_mov[df_mov['id'].isin(df_dur.index)]                    


    return df_mov


def get_zeroed_columns(df):
    df_work = df.sum()
    df_work = df_work[df_work == 0]

    return df_work.index.to_list()


def filter_df_gram_class_clus(df_gram, df, class_list, clus_list):
    df_gram_cent = df_gram[df_gram['cluster_label'].isin(clus_list)]
    df_gram_cent = df_gram_cent.join(df[['classeProcessual']])
    df_gram_cent = df_gram_cent[df_gram_cent['classeProcessual'].\
                                    isin(class_list)]
    df_gram_cent = df_gram_cent.drop(columns=['classeProcessual'])
    # df_gram_cent = df_gram_cent.drop(columns=get_zeroed_columns(df_gram_cent))
    df_gram_cent = df_gram_cent.drop(columns=['cluster_label'])
    
    # df_gram_cent = (df_gram_cent - df_gram_cent.min())/ \
    #                (df_gram_cent.max() - df_gram_cent.min())
    
    df_gram_cent = df_gram_cent.join(df_gram[['cluster_label']])

    return df_gram_cent


def filter_df_gram_outlier_by_acts(df_gram):
    df_gram_out = df_gram[df_gram['cluster_label'] == -1]
    df_gram_out = df_gram_out[(df_gram_out['Protocolo de Petição_118'] > 0) | 
                              (df_gram_out['Redistribuição_36'] > 0)]

    df_gram = df_gram[df_gram['cluster_label'] != -1]
    df_gram = pd.concat([df_gram, df_gram_out])


    return df_gram
