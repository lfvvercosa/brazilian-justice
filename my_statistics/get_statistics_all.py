from my_statistics.core import stat_parser 
from my_statistics.core import court_role
import core.my_loader as my_loader

import pandas as pd
import datetime
import os
import boto3
import numpy as np


def generate_statistics(my_justice, dir_name):


    """## Create Results Dict"""

    # checking if the directory
    # exist or not.

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    just_specs = court_role.\
                    get_justice_filenames(my_justice)

    my_dataset = {
        'JUSTICE':[],
        'JUSTICE_SPECIFIC':[],

        'NON_NULL_COURTS':[],
        'NON_NULL_SUBJECTS':[],
        'NON_NULL_MOVEMENTS':[],
        'NON_NULL_TRACES':[],
        'NON_NULL_CLASSES':[],

        'NULL_COURTS':[],
        'NULL_SUBJECTS':[],
        'NULL_MOVEMENTS':[],
        'NULL_TRACES':[],
        'NULL_CLASSES':[],
        
        'INVALID_SUBJECTS':[],
        'INVALID_MOVEMENTS':[],
        'INVALID_TRACES':[],
        'INVALID_CLASSES':[],

        'USABLE_SUBJECTS':[],
        'USABLE_MOVEMENTS':[],
        'USABLE_TRACES':[],
        'USABLE_CLASSES':[],

        'UNIQUE_TRACE_ID':[],
        'UNIQUE_USABLE_TRACES':[],
        'UNIQUE_SUBJECTS':[],
        'UNIQUE_MOVEMENTS':[],
        'UNIQUE_CLASSES':[],

        'PERCENT_UNIQUE_VARIANTS':[],
        'PERCENT_USABLE_MOVS':[],
        'PERCENT_USABLE_TRACES':[],
        'PERCENT_USABLE_CLASSES':[],
        'PERCENT_USABLE_SUBJECTS':[],

        'AVG_MONTHS_BY_TRACE':[],
        'STD_MONTHS_BY_TRACE':[],
        'AVG_MOVS_BY_TRACE':[],
        'STD_MOVS_BY_TRACE':[],
        'AVG_TRACES_BY_COURT':[],
        'STD_TRACES_BY_COURT':[],
        'AVG_SUBJ_BY_TRACE':[],
        'STD_SUBJ_BY_TRACE':[],
    }

    classes_dfs = []
    subjects_dfs = []
    movements_dfs = []

    traces_by_court_dfs = []
    mov_by_traces_dfs = []
    months_by_trace_dfs = []
    subj_by_trace_dfs = []
    variants_dfs = []

    df_cla = pd.read_csv('df_classes.csv', sep=',')
    df_mov = pd.read_csv('df_movimentos.csv', sep=',')
    df_assu = pd.read_csv('df_assuntos.csv', sep=',')

    df_cla = df_cla.rename(columns={
        'cod_item':'classeProcessual',
        'nome':'classeNome',
    }).drop(columns=['cod_item_pai'])

    df_cla['classeProcessual'] = df_cla['classeProcessual'].astype(int)

    df_mov = df_mov.rename(columns={
        'cod_item':'movimentoCodigoNacional',
        'nome':'movimentoNome',
    }).drop(columns=['cod_item_pai'])

    df_assu = df_assu.rename(columns={
        'cod_item':'assuntoCodigoNacional',
        'nome':'assuntoNome',
    }).drop(columns=['cod_item_pai'])


    for my_j in just_specs:
        my_justice_spec = my_j

        my_dataset['JUSTICE'].append(my_justice)
        my_dataset['JUSTICE_SPECIFIC'].append(my_j)

        print('justiça: ' + str(my_justice))
        print('specif: ' + str(my_justice_spec))
        print('working...')
        
        subfolder = 'processos-' + my_j.lower()
        my_files = [
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_1.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_2.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_3.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_4.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_5.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_6.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_7.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_8.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_9.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_10.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_11.json',
            'dataset/'+my_justice.lower()+'/' + subfolder + \
                '/processos-'+ my_j.lower() +'_12.json',
        ]

        df = None
        
        print('Loading JSON files...')

        for f in my_files:
            try:
                temp = pd.read_json(f)
                print('file ' + f + ' was loaded.')

                if df is None:
                    df = temp
                else:
                    df = pd.concat([df, temp])
            except Exception as e:
                pass

        """## Extract Data from JSON Columns"""

        df['id'] = np.arange(len(df))


        df['assuntoCodigoNacional'] = df['dadosBasicos'].\
            apply(stat_parser.extract_codigo_nacional_assunto)
        
        df['classeProcessual'] = df['dadosBasicos'].\
            apply(stat_parser.extract_classe_processual)
        

        df['orgaoJulgadorCodigo'] = df['dadosBasicos'].\
            apply(stat_parser.extract_orgao_julgador_codigo)
        
        df['processoNumero'] = df['dadosBasicos'].\
            apply(stat_parser.extract_numero_processo)

        # Number of courts
        print('Courts...')
        temp = df[~df['orgaoJulgadorCodigo'].isna()]
        num_orgaos_jud = len(pd.unique(temp['orgaoJulgadorCodigo']))
        my_dataset['NON_NULL_COURTS'].append(num_orgaos_jud)

        # Null courts
        x = df['orgaoJulgadorCodigo'].isna().sum()
        my_dataset['NULL_COURTS'].append(x)

        num_unique_traces = len(pd.unique(df['processoNumero']))
        my_dataset['UNIQUE_TRACE_ID'].append(num_unique_traces)

        test = df[['processoNumero']].groupby('processoNumero').\
            agg(count=('processoNumero', 'count'))
        test = test.sort_values(by=['count'], ascending=False)

        temp = df[['orgaoJulgadorCodigo','processoNumero']]
        temp = temp.groupby('orgaoJulgadorCodigo').\
                agg(traces_num=('processoNumero','count'))

        my_dataset['AVG_TRACES_BY_COURT'].\
            append(round(temp['traces_num'].mean(),2))
        my_dataset['STD_TRACES_BY_COURT'].\
            append(round(temp['traces_num'].std(),2))

        # Null subjects
        my_df_assu = df[['id','processoNumero','assuntoCodigoNacional']].\
                    explode('assuntoCodigoNacional')

        x = my_df_assu['assuntoCodigoNacional'].isna().sum()
        my_dataset['NULL_SUBJECTS'].append(x)

        # Number of subjects
        print('Subjects...')
        temp = my_df_assu
        temp2 = temp[~temp['assuntoCodigoNacional'].isna()]

        my_dataset['NON_NULL_SUBJECTS'].append(len(temp2))
        my_dataset['UNIQUE_SUBJECTS'].\
            append(len(temp2['assuntoCodigoNacional'].drop_duplicates()))

        invalid_subj = temp2[~temp2['assuntoCodigoNacional'].\
                isin(df_assu['assuntoCodigoNacional'])]
        my_dataset['INVALID_SUBJECTS'].append(len(invalid_subj))
        invalid_traces = invalid_subj
        invalid_traces = invalid_traces[['id','processoNumero']]
        
        valid_subj = temp2[~temp2['assuntoCodigoNacional'].\
                            isin(invalid_subj['assuntoCodigoNacional'])]

        my_dataset['USABLE_SUBJECTS'].append(len(valid_subj.index))
        valid_subj = None

        temp2 = temp[temp['assuntoCodigoNacional'].isna()]
        temp = temp[~temp['id'].isin(temp2['id'])]
        temp = temp.groupby(['id','processoNumero']).\
                agg(subject_count=('assuntoCodigoNacional','count'))

        my_dataset['AVG_SUBJ_BY_TRACE'].\
            append(round(temp['subject_count'].mean(),2))
        my_dataset['STD_SUBJ_BY_TRACE'].\
            append(round(temp['subject_count'].std(),2))

        subj_by_trace_dfs.append(temp[['subject_count']])

        my_df_assu = my_df_assu.drop_duplicates(subset=['assuntoCodigoNacional'])

        subjects_dfs.append(my_df_assu)

        my_df_assu = None

        # Trace by court
        df_trace_by_court = df.drop(columns=['movimento']).\
                            groupby('orgaoJulgadorCodigo').\
                            agg(trace_number=('processoNumero','count'))

        traces_by_court_dfs.append(df_trace_by_court)

        # Null classes
        print('Classes...')
        x = df['classeProcessual'].isna().sum()
        my_dataset['NULL_CLASSES'].append(x)
        
        # Number of classes
        my_df_cla = df[['classeProcessual']]
        my_df_cla = my_df_cla[~my_df_cla['classeProcessual'].isna()]

        if x == 0:
            df['classeProcessual'] = df['classeProcessual'].astype(int)
        else:
            raise Exception('null class')

        my_dataset['NON_NULL_CLASSES'].append(len(my_df_cla))
        my_dataset['UNIQUE_CLASSES'].\
            append(len(my_df_cla[['classeProcessual']].drop_duplicates()))

        my_dataset['INVALID_CLASSES'].append(
            len(df[~df['classeProcessual'].isin(df_cla['classeProcessual'])])
        )
        temp = df[~df['classeProcessual'].isin(df_cla['classeProcessual'])]
        temp = temp[['id','processoNumero']]

        my_dataset['USABLE_CLASSES'].append(
            len(df[df['classeProcessual'].isin(df_cla['classeProcessual'])])
        )

        invalid_traces = pd.concat([invalid_traces, temp])
        
        my_df_cla = my_df_cla.drop_duplicates(subset=['classeProcessual'])

        classes_dfs.append(my_df_cla)

        # Number of movements

        print('Movements...')

        df = df.explode('movimento')

        df['movimentoCodigoNacional'] = \
            df['movimento'].apply(stat_parser.extract_codigo_nacional_movimento)
        df['movimentoDataHora'] = \
            df['movimento'].apply(stat_parser.extract_data_hora_movimento)

        # Number of non-null traces and null traces
        print('Traces...')

        temp = df[(df['processoNumero'].isna()) | \
            (df['movimentoCodigoNacional'].isna()) | \
            (df['movimentoDataHora'].isna()) | \
            (df['orgaoJulgadorCodigo'].isna())]
        temp = temp.drop_duplicates(['id'])

        temp2 = df[~df['id'].isin(temp['id'])]
        temp2 = temp2.drop_duplicates(['id'])

        num_traces = len(temp2.index)
        my_dataset['NON_NULL_TRACES'].append(num_traces)

        null_traces = temp

        my_dataset['NULL_TRACES'].append(len(temp.index))


        temp = df[['id',
                'processoNumero',
                'movimentoCodigoNacional',
                'movimentoDataHora']]
        temp2 = temp[(~temp['movimentoCodigoNacional'].isna()) & \
                    (~temp['movimentoDataHora'].isna())]

        my_dataset['NON_NULL_MOVEMENTS'].append(len(temp2))
        my_dataset['UNIQUE_MOVEMENTS'].append(len(temp2[['movimentoCodigoNacional']].\
                                                drop_duplicates()))

        

        temp = df[(df['movimentoCodigoNacional'].isna()) | \
                (df['movimentoDataHora'].isna())] \
                [['id',
                'processoNumero',
                'movimentoCodigoNacional',
                'orgaoJulgadorCodigo']]
        
        null_mov = len(temp)

        my_dataset['NULL_MOVEMENTS'].append(null_mov)

        null_movs = temp[['id', 
                        'processoNumero',
                        'orgaoJulgadorCodigo']]

        temp = df[(~df['movimentoCodigoNacional'].isna()) & \
                (~df['movimentoDataHora'].isna())]
        temp = temp[(temp['movimentoDataHora'] < \
                    datetime.datetime(year=1900, month=1, day=1)) | \
                    (temp['movimentoDataHora'] > \
                    datetime.datetime(year=2022, month=1, day=1)) | \
                    (~temp['movimentoCodigoNacional'].\
                        isin(df_mov['movimentoCodigoNacional']))]\
                [['id',
                    'processoNumero',
                    'movimentoDataHora',
                    'orgaoJulgadorCodigo',
                    'movimentoCodigoNacional']]
        
        my_dataset['INVALID_MOVEMENTS'].append(len(temp))

        valid_movs = df[(~df['movimentoCodigoNacional'].isna()) & \
                (~df['movimentoDataHora'].isna())]
        valid_movs = valid_movs[(valid_movs['movimentoDataHora'] >= \
                    datetime.datetime(year=1900, month=1, day=1)) & \
                    (valid_movs['movimentoDataHora'] <= \
                    datetime.datetime(year=2022, month=1, day=1)) & \
                    (valid_movs['movimentoCodigoNacional'].\
                        isin(df_mov['movimentoCodigoNacional']))]\
        
        my_dataset['USABLE_MOVEMENTS'].append(len(valid_movs.index))

        temp = temp[['processoNumero',
                    'orgaoJulgadorCodigo',
                    'id']]
        temp = temp[(~temp['processoNumero'].isna()) & \
                    (~temp['orgaoJulgadorCodigo'].isna())]

        invalid_traces = pd.concat([invalid_traces, temp])
        invalid_traces = invalid_traces.\
            drop_duplicates(subset=['id','processoNumero'])
        invalid_traces = invalid_traces[~invalid_traces['id'].\
                                        isin(null_traces['id'])]

        my_dataset['INVALID_TRACES'].append(len(invalid_traces.index))

        temp = df[['id',
                'processoNumero',
                'movimentoCodigoNacional',
                'movimentoDataHora']]
        temp = temp[~temp['id'].isin(invalid_traces['id'])]
        temp = temp[~temp['id'].isin(null_traces['id'])]
        temp2 = temp.drop_duplicates(subset=['id','processoNumero'])

        my_dataset['USABLE_TRACES'].append(len(temp2.index))
        
        temp = temp.sort_values('movimentoDataHora')
        temp = temp.groupby(['id','processoNumero'])\
                ['movimentoCodigoNacional'].\
                apply(lambda x: ','.join(map(str, x)))
        
        print('### valid traces: ' + str(len(temp.index)))
        
        temp = temp.drop_duplicates() 
        
        variants_dfs.append(temp.to_frame())

        invalid_traces = None
        print('### unique traces: ' + str(len(temp.index)))
        
        my_dataset['UNIQUE_USABLE_TRACES'].append(len(temp.index))

        temp = df[(df['movimentoCodigoNacional'].isna()) | \
                (df['movimentoDataHora'].isna())] \
                [['processoNumero',
                    'movimentoCodigoNacional',
                    'movimentoDataHora',
                    'id']]

        temp2 = df[(df['movimentoDataHora'] < \
                    datetime.datetime(year=1900, month=1, day=1)) | \
                (df['movimentoDataHora'] > \
                    datetime.datetime(year=2022, month=1, day=1))] \
                [['processoNumero',
                    'movimentoCodigoNacional',
                    'movimentoDataHora',
                    'id']]

        temp = df[(~df['id'].isin(temp['id'])) & \
                (~df['id'].isin(temp2['id'])) ]


        temp2 = temp.groupby(['id','processoNumero']).\
                agg(mov_number=('movimentoCodigoNacional','count'))

        my_dataset['AVG_MOVS_BY_TRACE'].\
            append(round(temp2['mov_number'].mean(),2))
        my_dataset['STD_MOVS_BY_TRACE'].\
            append(round(temp2['mov_number'].std(),2))

        mov_by_traces_dfs.append(temp2)
        

        # Months by trace
        temp = temp.groupby(['id','processoNumero']).\
            agg({'movimentoDataHora':['min','max']})
        temp[('movimentoDataHora', 'min')] = \
            pd.to_datetime(temp[('movimentoDataHora', 'min')])
        temp[('movimentoDataHora', 'max')] = \
            pd.to_datetime(temp[('movimentoDataHora', 'max')])

        temp['duration'] = (temp[('movimentoDataHora', 'max')] - \
                            temp[('movimentoDataHora', 'min')])/np.timedelta64(1, 'M')

        months_by_trace_dfs.append(temp[['duration']])

        my_dataset['AVG_MONTHS_BY_TRACE'].append(round(temp['duration'].mean(),2))
        my_dataset['STD_MONTHS_BY_TRACE'].append(round(temp['duration'].std(),2))

        temp = df[~df['movimentoCodigoNacional'].isna()]\
                [[
                    'movimentoCodigoNacional',
                ]]

        movements_dfs.append(temp.drop_duplicates(\
                                subset=['movimentoCodigoNacional']))

        perc_usab_movs = round(
            my_dataset['USABLE_MOVEMENTS'][-1]/ \
            (my_dataset['NON_NULL_MOVEMENTS'][-1] + \
            my_dataset['NULL_MOVEMENTS'][-1]),
            2
        )

        perc_usab_subj = round(
            my_dataset['USABLE_SUBJECTS'][-1]/ \
            (my_dataset['NON_NULL_SUBJECTS'][-1] + \
            my_dataset['NULL_SUBJECTS'][-1]),
            2
        )

        perc_usab_class = round(
            my_dataset['USABLE_CLASSES'][-1]/ \
            (my_dataset['NON_NULL_CLASSES'][-1] + \
            my_dataset['NULL_CLASSES'][-1]),
            2
        )

        perc_usab_traces = round(
            my_dataset['USABLE_TRACES'][-1]/ \
            (my_dataset['NON_NULL_TRACES'][-1] + \
            my_dataset['NULL_TRACES'][-1]),
            2
        )

        if my_dataset['USABLE_TRACES'][-1] > 0:
            perc_variants = round(
                my_dataset['UNIQUE_USABLE_TRACES'][-1]/ \
                my_dataset['USABLE_TRACES'][-1],
                2
            )
        else:
            perc_variants = 0

        my_dataset['PERCENT_USABLE_MOVS'].append(
            perc_usab_movs
        )

        my_dataset['PERCENT_USABLE_SUBJECTS'].append(
            perc_usab_subj
        )

        my_dataset['PERCENT_USABLE_CLASSES'].append(
            perc_usab_class
        )

        my_dataset['PERCENT_USABLE_TRACES'].append(
            perc_usab_traces
        )

        my_dataset['PERCENT_UNIQUE_VARIANTS'].append(
            perc_variants
        )

        """## Save Statistics"""

        df_save = pd.DataFrame.from_dict(my_dataset)
        name = dir_name + '/' + my_justice.lower() + '.csv'

        df_save.to_csv(
                    path_or_buf=name,
                    sep='\t',
                    header=True,
                    index=False
                    )

        print('iteration done!')
        print()


    # Calculate to all

    avgs = [
    'AVG_MONTHS_BY_TRACE',
    'STD_MONTHS_BY_TRACE',
    'AVG_MOVS_BY_TRACE',
    'STD_MOVS_BY_TRACE',
    'AVG_TRACES_BY_COURT',
    'STD_TRACES_BY_COURT' ,
    'AVG_SUBJ_BY_TRACE',
    'STD_SUBJ_BY_TRACE',
    ]

    uniques = [
        'UNIQUE_SUBJECTS',
        'UNIQUE_MOVEMENTS',
        'UNIQUE_CLASSES',
        'UNIQUE_USABLE_TRACES',
    ]

    usables = [
        'USABLE_SUBJECTS',
        'USABLE_MOVEMENTS',
        'USABLE_TRACES',
        'USABLE_CLASSES',
    ]

    percents = [
        'PERCENT_UNIQUE_VARIANTS',
        'PERCENT_USABLE_MOVS',
        'PERCENT_USABLE_TRACES',
        'PERCENT_USABLE_CLASSES',
        'PERCENT_USABLE_SUBJECTS',
    ]


    for k in my_dataset:
        if k == 'JUSTICE':
            my_dataset['JUSTICE'].append(my_justice)
        elif k == 'JUSTICE_SPECIFIC':
            my_dataset['JUSTICE_SPECIFIC'].append('ALL')
        elif k not in avgs and k not in uniques and \
            k not in percents:
            my_dataset[k].\
                append(sum(my_dataset[k]))


    for e in avgs:
        # Months by trace
        df_all = None
        list_dfs = None
        
        if 'AVG_MONTHS' in e:
            list_dfs = months_by_trace_dfs
            col = 'duration'
        elif 'AVG_MOVS' in e:
            list_dfs = mov_by_traces_dfs
            col = 'mov_number'
        elif 'AVG_TRACES' in e:
            list_dfs = traces_by_court_dfs
            col = 'trace_number'
        elif 'AVG_SUBJ' in e:
            list_dfs = subj_by_trace_dfs
            col = 'subject_count'
        else:
            continue

        for df in list_dfs:
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all,df])
            
        my_dataset[e].\
            append(round(df_all[col].mean(),2))

        std_e = 'STD' + e[3:]

        my_dataset[std_e].\
            append(round(df_all[col].std(),2))


    for e in uniques:

        df_all = None
        list_dfs = None

        if 'UNIQUE_SUBJECTS' == e:
            list_dfs = subjects_dfs
            col = 'assuntoCodigoNacional'
        elif 'UNIQUE_MOVEMENTS' == e:
            list_dfs = movements_dfs
            col = 'movimentoCodigoNacional'
        elif 'UNIQUE_CLASSES' == e:
            list_dfs = classes_dfs
            col = 'classeProcessual'
        elif 'UNIQUE_USABLE_TRACES' == e:
            list_dfs = variants_dfs
            col = 'movimentoCodigoNacional'
        else:
            raise Exception('not recognized unique')

        for df in list_dfs:
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all,df])
        
        df_all = df_all[~df_all[col].isna()]

        my_dataset[e].\
            append(len(df_all[col].drop_duplicates()))


    perc_usab_movs = round(
            my_dataset['USABLE_MOVEMENTS'][-1]/ \
            (my_dataset['NON_NULL_MOVEMENTS'][-1] + \
            my_dataset['NULL_MOVEMENTS'][-1]),
            2
        )

    perc_usab_subj = round(
        my_dataset['USABLE_SUBJECTS'][-1]/ \
        (my_dataset['NON_NULL_SUBJECTS'][-1] + \
        my_dataset['NULL_SUBJECTS'][-1]),
        2
    )

    perc_usab_class = round(
        my_dataset['USABLE_CLASSES'][-1]/ \
        (my_dataset['NON_NULL_CLASSES'][-1] + \
        my_dataset['NULL_CLASSES'][-1]),
        2
    )

    perc_usab_traces = round(
        my_dataset['USABLE_TRACES'][-1]/ \
        (my_dataset['NON_NULL_TRACES'][-1] + \
        my_dataset['NULL_TRACES'][-1]),
        2
    )

    if my_dataset['USABLE_TRACES'][-1] > 0:
        perc_variants = round(
            my_dataset['UNIQUE_USABLE_TRACES'][-1]/ \
            my_dataset['USABLE_TRACES'][-1],
            2
        )
    else:
        perc_variants = 0


    my_dataset['PERCENT_USABLE_MOVS'].append(
        perc_usab_movs
    )

    my_dataset['PERCENT_USABLE_SUBJECTS'].append(
        perc_usab_subj
    )

    my_dataset['PERCENT_USABLE_CLASSES'].append(
        perc_usab_class
    )

    my_dataset['PERCENT_USABLE_TRACES'].append(
        perc_usab_traces
    )

    my_dataset['PERCENT_UNIQUE_VARIANTS'].append(
        perc_variants
    )

        
    df_save = pd.DataFrame.from_dict(my_dataset)    
    df_save.to_csv(
                path_or_buf=name,
                sep='\t',
                header=True,
                index=False
                )

    print('done!')