from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix

import pm4py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

import core.my_filter as my_filter
import core.my_utils as my_utils
import core.my_cluster as my_cluster
import core.my_model as my_model

from matplotlib import rc


def convert_df_to_log(df_mov, df_code_mov):
    df_vis = my_utils.get_act_name(df_mov, df_code_mov)
    df_vis['id'] = df_vis['id'].astype(str)

    df_vis = df_vis.rename(columns={
        'id':'case:concept:name',
        'movimentoCodigoNacional':'concept:name',
        'movimentoDataHora':'time:timestamp',
    })

    df_vis = df_vis[[
        'case:concept:name',
        'concept:name',
        'time:timestamp',
    ]]

   
    return pm4py.convert_to_event_log(df_vis)


def visualize_dfgs(dfgs):
    for c,dfg in dfgs.items():
        print('cluster: ' + str(c))
        pm4py.view_dfg(dfg[0], dfg[1], dfg[2])


def visualize_distribs(df, cat_col, cluster_labels, limit, thrs, clusters):
    fig = plt.figure(figsize=(15,75))
    
    c = Counter(cluster_labels)
    
    if -1 in c:
        del c[-1]
    
    if limit:
        clusters = c.most_common(limit)
        clusters = [c[0] for c in clusters]       

    for pos,value in enumerate(clusters):
        df_plot = df[df['cluster_label'] == value]
        s_plot = my_filter.rem_unfrequent(df_plot[cat_col], thrs)
        plt.subplot(3, 3, pos + 1)
        plt.hist(s_plot.astype("string").to_list())
        plt.title('cluster ' + str(value))
        
    plt.subplots_adjust(bottom=-0.2)
    plt.show(block=True)
    plt.close(fig)


def visualize_distribs2(s, thrs, classe):
    counts = s.value_counts()
    total_values = len(s)
    min_val = int(thrs*total_values)
    to_drop = counts[counts < min_val].index
    s = s[~s.isin(to_drop)]
    s_list = s.to_list()

    x = [c[0] for c in Counter(s_list).most_common()]
    y = [c[1] for c in Counter(s_list).most_common()]
    y = [round(c/total_values,2) for c in y]

    plt.bar(x,y)
    plt.title('Classe ' + str(classe))
    plt.xlabel('Movimentação')
    plt.ylabel('Quantidade (%)')

    plt.show(block=True)
    plt.close()

    # print('### total values: ' + str(total_values))

    # fig = plt.figure(figsize=(10,50))
    # plt.hist(s_ordered, edgecolor = "black", align='left')
    # plt.title('Classe ' + str(classe))
    # plt.show(block=True)
    # plt.close(fig)


def visualize_distribs_attr(df, cat_col, limit, thrs, attr):
    fig = plt.figure(figsize=(15,75))
    
    df_temp = df.copy(deep=True)
    # df_temp = df_temp[df_temp['cluster_label'] != -1]

    if limit:
        c = Counter(df[cat_col].to_list())
        attr = c.most_common(limit)
        attr = [c[0] for c in attr]

    for pos,value in enumerate(attr):
        df_plot = df_temp[df_temp[cat_col] == value]
        s_plot = df_plot['cluster_label']
        s_plot = my_filter.rem_unfrequent(df_plot['cluster_label'], thrs)
        plt.subplot(3, 2, pos + 1)
        plt.hist(s_plot.astype("string").to_list())
        plt.title(cat_col + ' ' + str(value))

    plt.subplots_adjust(bottom=-0.2)
    plt.show(block=True)
    plt.close(fig)


def visualize_distribs_mov(df_gram):
    cols = list(df_gram.columns)
    cols.remove('cluster_label')
    
    df_sum = df_gram[cols]
    df_sum = df_sum.sum(axis=1).to_frame(name='total_movs')
    df_sum = df_sum.join(df_gram[['cluster_label']],
                         how='left')
    s_clust = df_sum[df_sum['cluster_label'] != -1]['total_movs']
    s_outl = df_sum[df_sum['cluster_label'] == -1]['total_movs']
    my_min = min(s_clust.min(),s_outl.min())
    my_max = max(s_clust.max(),s_outl.max())

    fig = plt.figure(figsize=(15,75))
    
    plt.subplot(1, 2, 1)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.hist(s_clust, range=(my_min,my_max))
    plt.title('Number Movs Cluster', fontsize=17)
    plt.subplot(1, 2, 2)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.hist(s_outl)
    plt.title('Number Movs Outliers', fontsize=17)

    plt.show(block=True)
    plt.close(fig)


def visualize_distribs_dict(d, is_numeric=True, title=None):
    fig = plt.figure(figsize=(15,75))
    size_d = len(d)

    if title:
        plt.suptitle(title)

    if size_d <= 4:
        n_cols = 2
        n_rows = 2
    elif size_d <=9:
        n_cols = 3
        n_rows = 3
    else:
        raise Exception('too many histograms!')

    count = 1

    if is_numeric:
        my_min = 0
        my_max = 0
        my_max_y = 0

        for k in d:
            my_min = min(my_min,d[k].min())
            my_max = max(my_max,d[k].max())

        for k in d:
            counts, _ = np.histogram(d[k], range=(my_min,my_max))
            mode = max(counts)
            my_max_y = max(my_max_y, mode)

    # my_min = 0
    # my_max = 5

    for k in d:
        plt.subplots_adjust(bottom=-0.2)
        plt.subplot(n_rows, n_cols, count)

        if is_numeric:
            ax = plt.gca()
            ax.set_ylim([0, my_max_y*1.05])
            ax.set_xlim([my_min, my_max])

            # counts, bins = np.histogram(d[k], range=(my_min,my_max))
            plt.hist(d[k], range=(my_min,my_max), bins=20, edgecolor='black')
            # plt.stairs(counts, bins, fill=True)
        else:
            plt.hist(d[k])

        if size_d <= 4:
            plt.xticks(fontsize=17)
            plt.yticks(fontsize=17)
     

        if size_d <= 4:
            plt.title(k, fontsize=17)
        else:
            plt.title(k)
        
        count += 1

    plt.show(block=True)
    plt.close(fig)


def remove_infrequent_clusters(df_vis, cluster_count, min_pts):
    clusters = cluster_count.most_common(len(cluster_count))
    freq_clusters = []

    for c in clusters:
        if c[1] < min_pts:
            break

        freq_clusters.append(c[0])

    df_keep = pd.DataFrame(freq_clusters, columns=['cluster_label'])
    df_keep['keep'] = True

    df_vis['id'] = df_vis.index
    df_vis = df_vis.merge(df_keep, on='cluster_label', how='left')
    df_vis = df_vis[df_vis['keep'] == True]
    df_vis = df_vis.drop(columns=['keep'])
    df_vis = df_vis.set_index('id', drop=True)

    return df_vis


def visualize_clusters_in_2D(df_gram_norm, cluster_count, min_pts):
    df_vis = df_gram_norm.copy(deep=True)

    if cluster_count:
        df_vis = remove_infrequent_clusters(df_vis, cluster_count, min_pts)

    cols = list(df_vis.columns)
    cols.remove('cluster_label')

    my_data = df_vis[cols].to_numpy()
    my_labels = df_vis['cluster_label']

    df_red, colors_number = reduce_dimension_tsne(my_data, my_labels)

    sns.scatterplot(x="comp-1", y="comp-2", hue=df_red.y.tolist(),
                palette=sns.color_palette("colorblind", colors_number),
                data=df_red).set(title="1-gram clusters T-SNE projection")

    plt.show(block=True)
    plt.close()


def reduce_dimension_tsne(my_data, my_labels):
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(my_data)

    df_tsne = pd.DataFrame()
    df_tsne["y"] = my_labels
    df_tsne["comp-1"] = z[:,0]
    df_tsne["comp-2"] = z[:,1]

    colors_number = len(df_tsne["y"].drop_duplicates())

    return df_tsne, colors_number


def reduce_dimension_pca(my_data, my_labels):
    pca = PCA(n_components=2, whiten=True)
    z = pca.fit_transform(my_data)

    df_pca = pd.DataFrame()
    df_pca["y"] = my_labels
    df_pca["comp-1"] = z[:,0]
    df_pca["comp-2"] = z[:,1]

    colors_number = len(df_pca["y"].drop_duplicates())

    return df_pca, colors_number


def visualize_dendrogram(df, color_threshold, colors):
    rc('font',**{'family':'serif','serif':['Times New Roman']})
    X = df.to_numpy()
    dist_matrix = distance_matrix(X,X)
    Z = hierarchy.linkage(dist_matrix, 'average')

    def llf(id):
        return df.index[id][8:]

    def color(k):
        print('k is: ' + str(k))
        return colors[k]

    fig = plt.figure(figsize=(10,3))
    ax = plt.axes()
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.gcf().set_size_inches(10, 3.5)
    
    if not colors:
        color = None

    dendro = hierarchy.dendrogram(Z, 
                                  leaf_label_func=llf, 
                                  no_labels=False,
                                  color_threshold=color_threshold,
                                  link_color_func=color,
                                  ax=ax)

    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show(block=True)
    # plt.savefig('images/dendro_centroids.png', dpi=600)
    plt.close()


def map_names():
    return {
        'Decisão_3':'Decision (3)',
        'Baixa Definitiva_22':'Discharge (22)',
        'Distribuição_26':'Distribution (26)',
        'Redistribuição_36':'Redistribution (36)',
        'Conclusão_51':'Conclusion (51)',
        'Juntada_67':'Attachment (67)',
        'Publicação_92':'Publication (92)',
        'Protocolo de Petição_118':'Petition (118)',
        'Remessa_123':'Referral (123)',
        'Recebimento_132':'Receival (132)',
        'Julgamento_193':'Trial (193)',
        'Entrega em carga/vista_493':'Records Delivery (493)',
        'Trânsito em julgado_848':'Res Judicata (848)',
        'Arquivamento_861':'Filing (861)',
        'Guarda Permanente_867':'Storage (867)',
        'Desarquivamento_893':'Unfiling (893)',
        'Recebimento_977':'Receival (977)',
        'Remessa_978':'Referral (978)',
        'Disponibilização no Diário da Justiça Eletrônico_1061':\
            'Made Available (1061)',
        'Despacho_11009':'Dispatch (11009)',
    }


def map_index():
    return {
        'cluster_0':'0',
        'cluster_39':'39',
        'cluster_26':'26',
        'cluster_67':'67',
        'cluster_31':'31',
        'cluster_29':'29',
        'cluster_32':'32',
        'cluster_37':'37',
        'cluster_30':'30',
        'cluster_40':'40',
    }


def map_cols(columns):
    rename = {}

    for i in range(len(columns)):
        rename[columns[i]] = columns[i][8:]

    return rename
    

def gen_heatmap(df_cent, limit, cluster_count, clusters):
    rc('font',**{'family':'serif','serif':['Times New Roman']})

    if limit:
        clusters = cluster_count.most_common(limit)
        clusters = ['cluster_' + str(c[0]) for c in clusters]
        df_cent = df_cent[clusters]
    
    # df_cent = df_cent.drop(index='cluster_label')
    df_cent = df_cent.rename(
        index=map_names(), columns=map_cols(df_cent.columns))
    fontsize = 14

    plt.gcf().set_size_inches(8, 6)
    s = sns.heatmap(df_cent, 
                    annot=True, 
                    cmap="Greens_r",
                    annot_kws={"size":12},
                    )
    s.set_xlabel('Cluster', fontsize=fontsize)
    s.set_ylabel('Activity', fontsize=fontsize)
    s.xaxis.labelpad = 15
    s.yaxis.labelpad = 15

    cbar = s.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=fontsize)

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    plt.tight_layout()
    plt.show(block=True)
    # plt.savefig('images/heatmap_centroids_final.png', dpi=800)
    plt.close()


def visualize_heatmap_same_clus(df_dur, 
                                df_gram, 
                                clus, 
                                thres, 
                                base_path):
    thres = 5
    df_gram_work = df_gram.copy(deep=True)
    df_gram_work = df_gram_work.drop(columns=['cluster_label'])

    df_temp = df_dur[df_dur['cluster_label'] == clus]

    df_subclus_1 = df_temp[df_temp['duration'] <= thres][['duration']]
    df_subclus_1['cluster_label'] = clus + 0.1 

    df_subclus_2 = df_temp[df_temp['duration'] > thres][['duration']]
    df_subclus_2['cluster_label'] = clus + 0.2

    df_subclus = pd.concat([df_subclus_1,df_subclus_2])
    df_subclus = df_subclus.drop(columns=['duration'])
    df_subclus = df_subclus.join(df_gram_work, how='left')

    my_cluster.get_centroids(df_subclus,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                None, 
                base_path + 'tests/files/centroids_db_same_clus.csv')

    df_cent = pd.read_csv(base_path + 'tests/files/centroids_db_same_clus.csv',
                          sep='\t')
    df_cent = df_cent.set_index('activity')
    df_cent = df_cent.rename(
        index={'Disponibilização no Diário da Justiça Eletrônico_1061':
               'Disponibilização Diário_1061'})

    clusters = ['cluster_' + str(clus + 0.1), 'cluster_' + str(clus + 0.2)]
    gen_heatmap(df_cent, None, None, clusters)

    print()


def visualize_time_same_clus(df_gram_norm, 
                             df, 
                             df_dur, 
                             df_assu, 
                             clus, 
                             eps1, 
                             eps2, 
                             min_pts):
    df_temp = df_dur[df_dur['cluster_label'] == clus]
    df_gram_work = df_gram_norm.drop(columns=['cluster_label'])
    df_gram_work = df_gram_work[df_gram_work.index.isin(df_temp.index)]

    cluster_labels = my_cluster.cluster_dbscan(df_gram_work, eps1, min_pts)
    cluster_count = Counter(cluster_labels)
    
    print('### clusters: ' + str(cluster_count))
    
    df_gram_work['cluster_label_new'] = cluster_labels

    df_dur = df_dur.join(df_gram_work[['cluster_label_new']])
    all_labels = set(cluster_labels)
    dict_series = {}

    for l in all_labels:
        dict_series['cluster_' + str(l)] = df_dur\
            [df_dur['cluster_label_new'] == l]['duration']

    visualize_distribs_dict(dict_series)

    df_work = df_dur[~df_dur['cluster_label_new'].isna()]
    df_work = df_work.drop(columns=['cluster_label'])
    df_work = df_work.rename(columns={'cluster_label_new':'cluster_label'})
    df_work = df_work.join(df.drop(columns=['cluster_label','duration']), 
                           how='left')

    df_work2 = df_dur[~df_dur['cluster_label_new'].isna()]
    df_work2 = df_work2.drop(columns=['cluster_label','duration'])
    df_work2 = df_work2.rename(columns={'cluster_label_new':'cluster_label'})

    df_assu_temp = df_assu[['id','assuntoCodigoNacional']]
    df_assu_temp = df_assu_temp.set_index('id', drop=True)

    df_work2 = df_work2.join(df_assu_temp, how='left')


    print()


def visualize_time_attr_clus(df, attr, val, limit):
    dict_show = {}
    df_work = df[df[attr].isin(val)]
    df_work2 = df_work.groupby('cluster_label')[attr].count()
    df_work2 = df_work2.sort_values(ascending=False)

    val_limit = df_work2.iloc[limit - 1]

    df_work2 = df_work2[df_work2 >= val_limit]
    clusters = df_work2.index.to_list()

    for c in clusters:
        dict_show['cluster_' + str(c)] = \
            df_work[df_work['cluster_label'] == c]['duration']

    visualize_distribs_dict(dict_show)


def visualize_corr_act_time(df_gram, df_dur):
    df_work = df_gram.drop(columns=['cluster_label'])
    df_work = df_work.sum(axis=1).to_frame(name='count_act')
    df_work = df_work.join(df_dur.drop(columns=['cluster_label']), 
                           how='left')

    sns.pairplot(df_work)
    
    plt.show(block=True)
    plt.close()


def visualize_corr_act_clu(df_gram):
    df_work = df_gram.drop(columns=['cluster_label'])
    df_work = df_work.sum(axis=1).to_frame(name='count_act')

    df_work2 = df_gram['cluster_label'].value_counts().\
                            to_frame(name='total_traces')

    df_final = df_gram[['cluster_label']]
    df_final = df_final.join(df_work, how='left')
    df_final = df_final.groupby('cluster_label').\
                        agg(total_acts=('count_act','sum'))

    df_final = df_final.join(df_work2)
    df_final['mean_acts'] = df_final['total_acts'] / \
                             df_final['total_traces']
    df_final = df_final.drop(columns=['total_acts'])
    df_final = df_final.sort_values('total_traces')

    sns.pairplot(df_final)
    
    plt.show(block=True)
    plt.close()

    print()


def get_zeroed_columns(df):
    df_work = df.sum()
    df_work = df_work[df_work == 0]

    return df_work.index.to_list()


def visualize_corr_1gram_time(df_gram, 
                              df_dur, 
                              df, 
                              cla_list, 
                              incl_outl):

    df_temp = df[['id','classeProcessual']]
    df_temp = df_temp.set_index('id')

    df_work = df_gram.copy(deep=True)

    if not incl_outl:
        df_work = df_work[df_work['cluster_label'] != -1]

    df_work = df_work.drop(columns=['cluster_label'])
    df_work = df_work.join(df_temp)
    df_work = df_work[df_work['classeProcessual'].isin(cla_list)]
    df_work = df_work.drop(columns=\
        get_zeroed_columns(df_work) + ['classeProcessual'])
    df_work = df_work.join(df_dur.drop(columns=['cluster_label']))
    df_work = (df_work - df_work.min())/ \
              (df_work.max() - df_work.min())

    corr = df_work.corr()
    corr = corr.round(2)

    rename_dict = {'Disponibilização no Diário da Justiça Eletrônico_1061':
               'Disponibilização_1061'}
    corr = corr.rename(
        index=rename_dict, columns=rename_dict)

    sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True)
    
    plt.show(block=True)
    plt.close()

    print()


def visualize_corr(df_gram, df_dur, columns):
    df_work = df_gram.drop(columns=['cluster_label'])
    df_work = df_work.join(df_dur[['duration']])
    df_work = df_work.drop(columns= \
        get_zeroed_columns(df_work))

    corr = df_work.corr()
    corr = corr.round(2)

    rename_dict = {'Disponibilização no Diário da Justiça Eletrônico_1061':
               'Disponibilização_1061'}
    corr = corr.rename(
        index=rename_dict, columns=rename_dict)

    if columns:
        corr = corr[columns]

    sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.index,
        annot=True)
    
    plt.show(block=True)
    plt.close()


def split_distrib(df_gram, df_dur, classe):
    rc('font',**{'family':'serif','serif':['Times New Roman']})
    

    df_out_work = df_dur[df_dur.index.isin(df_gram.index)]


    if 985 in classe:
        max_x = 120
        max_y = 1000

        name = 'Guarda Permanente'
        df_gram_temp = df_gram[(df_gram['Guarda Permanente_867'] > 0)]

    elif 1032 in classe:
        max_x = 50
        max_y = 1400

        name = 'Protocolo de Petição_118'
        df_gram_temp = df_gram[(df_gram['Protocolo de Petição_118'] > 0)]

    elif 1720 in classe:
        max_x = 30
        max_y = 450

        name = 'Redistribuição'
        df_gram_temp = df_gram[(df_gram['Redistribuição_36'] > 0)]

    else:
        raise Exception('### Invalid class!!')

    
    df_work = df_dur[df_dur.index.isin(df_gram_temp.index)]
    df_out_work = df_out_work[~df_out_work.index.isin(df_work.index)]

    print('size df_work: ' + str(len(df_work)))
    print('size df_out_work: ' + str(len(df_out_work)))

    fig = plt.figure(figsize=(5,10))
    
    fontsize = 14

    ax = plt.axes()
    ax.set_xlabel('Time in Months', fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.set_ylim([0, max_y])
    ax.set_xlim([0, max_x])
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    plt.gcf().set_size_inches(4, 3)
    plt.hist(df_work['duration'], bins=20, edgecolor='black')
    plt.tight_layout()
    plt.savefig('images/hist_1032_peticao.png', dpi=600)

    # plt.subplot(1, 2, 1)
    # plt.title('Possui "' + name + '"')
    
    # ax = plt.gca()
    # ax.set_ylim([0, max_y])
    # ax.set_xlim([0, max_x])

    # plt.hist(df_work['duration'], bins=20, edgecolor='black')
    
    # plt.subplot(1, 2, 2)
    # plt.title('Não possui "' + name + '"')

    # ax = plt.gca()
    # ax.set_ylim([0, max_y])
    # ax.set_xlim([0, max_x])

    # plt.hist(df_out_work['duration'], bins=20, edgecolor='black')
    
    # plt.show(block=True)
    # plt.close(fig)


    return df_work['duration'], df_out_work['duration']


def visualize_centroids(df_cent):
    df_cent = df_cent.rename(index={
        'Disponibilização no Diário da Justiça Eletrônico_1061':
        'Disponibilização Diário_1061'})

    gen_heatmap(df_cent, None, None, None)


def visualize_time_by_class(df, classes):
    df_work = df[['classeProcessual','duration']]
    duration_classes = {}

    for c in classes:
        name = 'Classe ' + str(c)
        duration_classes[name] = \
            df_work[df_work['classeProcessual'] == c]['duration']
    
    visualize_distribs_dict(duration_classes)


def visualize_mov_freq_by_class(df_mov, classes):
    df_work = df_mov[['classeProcessual','movimentoCodigoNacional']]

    for c in classes:
        s = df_work[df_work['classeProcessual'] == c]['movimentoCodigoNacional']
        s = s.astype(str)

        visualize_distribs2(s, 0.05, c)


    print()


def visualize_time_clus_class(df, class_list, clus_list):
    df_work = df[['classeProcessual','cluster_label','duration']]
    df_work = df_work[df_work['cluster_label'].isin(clus_list)]
    df_work = df_work[df_work['classeProcessual'].isin(class_list)]
    duration_classes = {}

    for c in class_list:
        name = 'Classe ' + str(c)
        duration_classes[name] = \
            df_work[df_work['classeProcessual'] == c]['duration']
    
    visualize_distribs_dict(duration_classes)
    print()


def visualize_most_frequent_assunto(df_assu,
                                    df,
                                    df_dur,
                                    class_list, 
                                    clus_list, 
                                    max_assu):

    df_work = df_assu.join(df_dur.drop(columns=['cluster_label']))
    df_work = df_work[['id',
                       'assuntoCodigoNacional',
                       'classeProcessual',
                       'cluster_label',
                       'duration']]
    df_work = df_work[df_work['cluster_label'].isin(clus_list)]
    df_work = df_work[df_work['classeProcessual'].isin(class_list)]
    
    for cla in class_list:
        distribs = {}
        df_temp = df_work[df_work['classeProcessual'] == cla]
        assuntos = df_temp['assuntoCodigoNacional'].\
            value_counts()[:max_assu].index.tolist()
        for assu in assuntos:
            df_temp2 = df_temp[df_temp['assuntoCodigoNacional'] == assu]
            df_temp2 = df_temp2.\
                drop_duplicates(subset=['id', 'assuntoCodigoNacional'])
            distribs['Assunto ' + str(assu)] = \
                df_temp2['duration']
        
        df_temp2 = df_temp[~df_temp['assuntoCodigoNacional'].isin(assuntos)]
        df_temp2 = df_temp2.\
                drop_duplicates(subset=['id', 'assuntoCodigoNacional'])
        distribs['Assunto Outros'] = \
                df_temp2['duration']

        visualize_distribs_dict(distribs, title='Classe ' + str(cla))
    
    print()


def visualize_time_by_clus_and_class(df, 
                                     class_list, 
                                     clus_list, 
                                     class_clus_list):

    df_work = df[['classeProcessual','cluster_label','duration']]
    df_work = df_work[df_work['cluster_label'].isin(clus_list)]
    df_work = df_work[df_work['classeProcessual'].isin(class_list)]

    for c in class_clus_list.keys():
        duration_classes = {}
        title = 'Classe ' + str(c)
        df_temp = df_work[df_work['classeProcessual'] == c]

        for clus in class_clus_list[c]:
            df_temp2 = df_temp[df_temp['cluster_label'] == clus]
            duration_classes['Cluster ' + str(clus)] = df_temp2['duration']

        visualize_distribs_dict(duration_classes, title=title)
        print()


def visualize_dfg_clusters(df_mov, 
                           df_code_mov_eng,
                           clusters, 
                           classes, 
                           noise_thres,
                           features_code,
                           saving_path=None):

    df_vis = df_mov.copy(deep=True)

    if features_code:
        df_vis = my_filter.rem_spec_act(df_vis, 
                                        rem_act=features_code,
                                        act_col='movimentoCodigoNacional')

    if clusters:
        df_vis = df_vis[df_vis['cluster_label'].isin(clusters)]
    
    if classes:
        df_vis = df_vis[df_vis['classeProcessual'].isin(classes)]

    log = convert_df_to_log(df_vis, df_code_mov_eng)

    dfg_freq, sa_freq, ea_freq = my_model.create_dfg(log,performance=False)

    # dfg_freq, sa_freq, ea_freq = my_model.rem_noise_dfg_percent(dfg_freq,
    #                                                             sa_freq,
    #                                                             ea_freq,
    #                                                             1 - noise_thres)

    # dfg_freq, sa_freq, ea_freq = my_model.rem_noise_dfg(dfg_freq,
    #                                                     sa_freq,
    #                                                     ea_freq,
    #                                                     noise_thres)

    dfg_freq, sa_freq, ea_freq = my_model.rem_noise_dfg_with_excep(
                                        dfg_freq,
                                        sa_freq,
                                        ea_freq,
                                        noise_thres)


    if saving_path:
        pm4py.save_vis_dfg(dfg_freq, sa_freq, ea_freq, saving_path)
    else:
        pm4py.view_dfg(dfg_freq, sa_freq, ea_freq)


def visualize_dfg_perf_clusters2(df_mov, 
                                df_code_mov_eng,
                                clusters, 
                                classes, 
                                noise_thres,
                                features_code,
                                saving_path=None):

    df_vis = df_mov.copy(deep=True)

    if features_code:
        df_vis = my_filter.rem_spec_act(df_vis, 
                                        rem_act=features_code,
                                        act_col='movimentoCodigoNacional')

    df_vis = df_vis[df_vis['cluster_label'].isin(clusters)]
    df_vis = df_vis[df_vis['classeProcessual'].isin(classes)]

    log = convert_df_to_log(df_vis, df_code_mov_eng)

    dfg_aux, sa_freq, ea_freq = my_model.create_dfg(log,performance=False)
    dfg_perf, sa_perf, ea_perf = my_model.create_dfg(log,performance=True)
    dfg_perf, sa_perf, ea_perf = my_model.rem_noise_dfg_perf(dfg_perf,
                                                             dfg_aux,
                                                             sa_perf,
                                                             ea_perf,
                                                             noise_thres)

    if saving_path:
        pm4py.save_vis_performance_dfg(dfg_perf, 
                                       sa_perf, 
                                       ea_perf, 
                                       saving_path)
    else:
        pm4py.view_performance_dfg(dfg_perf, 
                                   sa_perf, 
                                   ea_perf,
                                   aggregation_measure='median')


def visualize_dfg_perf_clusters(df_mov, 
                                df_code_mov_eng,
                                clusters, 
                                classes, 
                                noise_thres,
                                features_code,
                                saving_path=None):

    df_vis = df_mov.copy(deep=True)

    if features_code:
        df_vis = my_filter.rem_spec_act(df_vis, 
                                        rem_act=features_code,
                                        act_col='movimentoCodigoNacional')

    df_vis = df_vis[df_vis['cluster_label'].isin(clusters)]
    df_vis = df_vis[df_vis['classeProcessual'].isin(classes)]

    log = convert_df_to_log(df_vis, df_code_mov_eng)

    dfg_aux, sa_freq, ea_freq = my_model.create_dfg(log,performance=False)
    dfg_perf, sa_perf, ea_perf = my_model.create_dfg(log,performance=True)

    # dfg_perf, sa_perf, ea_perf = my_model.rem_noise_dfg_perf(dfg_perf,
    #                                                          dfg_aux,
    #                                                          sa_perf,
    #                                                          ea_perf,
    #                                                          noise_thres)

    dfg_perf, sa_perf, ea_perf = my_model.rem_noise_dfg_perf_with_excep(
                                                        dfg_aux,
                                                        sa_freq,
                                                        ea_freq,
                                                        dfg_perf,
                                                        sa_perf,
                                                        ea_perf,
                                                        noise_thres)


    if saving_path:
        pm4py.save_vis_performance_dfg(dfg_perf, 
                                       sa_perf, 
                                       ea_perf, 
                                       saving_path)
    else:
        pm4py.view_performance_dfg(dfg_perf, 
                                   sa_perf, 
                                   ea_perf,
                                   aggregation_measure='median')


def adjust_plot(fontsize, x_label, y_label):
    ax = plt.axes()
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.gcf().set_size_inches(4, 3)



def visualize_clusters_size_distrib(df_gram):
    rc('font',**{'family':'serif','serif':['Times New Roman']})

    df_vis = df_gram.copy(deep=True)
    df_vis = df_vis[df_vis['cluster_label'] != -1]

    size_distrib = df_gram.groupby('cluster_label').\
                    agg(count=('cluster_label','count')).\
                    sort_values('count', ascending=False)
    y = size_distrib['count'].to_list()
    x = list(range(len(y)))

    adjust_plot(14, 'Cluster', 'Number of Records')

    plt.plot(x,y)
    plt.tight_layout()
    plt.savefig('images/clusters_size.png', dpi=300)


def get_accum(l):
    l_accum = l.copy()

    for id in range(len(l_accum)):
        if id > 0:
            l_accum[id] = l_accum[id] + l_accum[id-1]
        else:
            l_accum[id] = l_accum[id]

    return l_accum


def visualize_clusters_accum(df_gram):
    rc('font',**{'family':'serif','serif':['Times New Roman']})

    df_vis = df_gram.copy(deep=True)
    df_vis = df_vis[df_vis['cluster_label'] != -1]

    size_distrib = df_gram.groupby('cluster_label').\
                    agg(count=('cluster_label','count')).\
                    sort_values('count', ascending=False)
    y = size_distrib['count'].to_list()
    y = get_accum(y)
    x = list(range(len(y)))

    adjust_plot(14, 'Cluster', 'Number of Records')

    plt.plot(x,y)
    plt.tight_layout()
    plt.savefig('images/clusters_size_accum.png', dpi=300)