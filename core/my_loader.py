import pandas as pd
import datetime


def load_df(my_files):
    df = None

    for f in my_files:
        try:
            temp = pd.read_json(f)

            if df is None:
                df = temp
            else:
                df = pd.concat([df, temp])

            print('file found: ' + str(f))

        except Exception as e:
            pass
    
    return df


def load_just_spec_df(just_specs, base_path, my_justice):
    df = None 

    print('### loading justices data...')

    for my_j in just_specs:
        my_files = []

        for i in range(1,13):
            my_path = base_path + my_justice.lower() + \
                '/processos-' + my_j.lower() + '/processos-' + my_j.lower() + \
                '_' + str(i) + '.json'
            my_files.append(my_path)

        df_temp = load_df(my_files)

        if df is None:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])

    return df


def load_df_classes(base_path):
    df_cla = pd.read_csv(base_path + 'df_classes.csv', sep=',')
    df_cla = df_cla.rename(columns={
        'cod_item':'classeProcessual',
        'nome':'classeNome',}).drop(columns=['cod_item_pai'])

    return df_cla


def load_df_subject(base_path):
    df_assu = pd.read_csv(base_path + 'df_assuntos.csv', sep=',')
    df_assu = df_assu.rename(columns={
    'cod_item':'assuntoCodigoNacional',
    'nome':'assuntoNome',}).drop(columns=['cod_item_pai'])

    return df_assu


def load_df_movements(base_path):
    df_mov = pd.read_csv(base_path + 'df_movimentos.csv', sep=',')
    df_mov = df_mov.rename(columns={
    'cod_item':'movimentoCodigoNacional',
    'nome':'movimentoNome',}).drop(columns=['cod_item_pai'])
    df_mov['movimentoNome'] = df_mov['movimentoNome'] + '_' + \
                              df_mov['movimentoCodigoNacional'].astype(str)


    return df_mov


def load_df_movements_eng():
    my_dict = {
        'movimentoCodigoNacional':[3,22,26,36,51,67,92,118,123,132,193,848,861,867,893,977,978,1061,11009],
        # 'movimentoNome':['Decision (3)','Discharge (22)','Distribution (26)','Redistribution (36)','Conclusion (51)','Attachment (67)','Publication (92)','Petition (118)','Referral (123)','Receival (132)','Trial (193)','Res Judicata (848)','Filing (861)','Storage (867)','Unfiling (893)','Receival (977)','Referral (978)','Made Available (1061)','Dispatch (11009)']
        'movimentoNome':['Decision','Discharge','Distribution','Redistribution','Conclusion','Attachment','Publication','Petition','Referral','Receival','Trial','Res Judicata','Filing','Storage','Unfiling','Receival','Referral','Made Available','Dispatch']

    }

    return pd.DataFrame.from_dict(my_dict)






