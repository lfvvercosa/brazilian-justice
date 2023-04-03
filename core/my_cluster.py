from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from bisect import bisect_left
from nltk.util import ngrams

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import core.my_utils as my_utils


def find_k_kmeans(df_gram):
    X = df_gram.to_numpy()
    max_clust = min(len(X),40)
    n_clust = list(range(2,max_clust))
    wcss = []

    for k in n_clust:
        clustering = KMeans(n_clusters=k)
        cluster_labels = clustering.fit_predict(X)
        wcss.append(clustering.inertia_)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =", k,
            "The average silhouette_score is :", silhouette_avg,
        )
    
    sns.lineplot(x = n_clust, y = wcss)
    plt.show()


def get_centroids(df_gram, kmeans, my_path, save=True):
    if kmeans:
        centroids  = kmeans.cluster_centers_
        cols = list(df_gram.columns)
        cols.remove('cluster_label')

        df_centr = pd.DataFrame(
                        data=centroids[:,:],
                        columns=cols,
                   ).round(3)

        df_centr = df_centr.T

        if save:
            df_centr.to_csv(my_path, sep='\t', index_label='cluster')
            print('centroids saved!')
    else:
        center_list = df_gram['cluster_label'].drop_duplicates().tolist()
        df_centr = None

        for c in center_list:
            df_temp = df_gram[df_gram['cluster_label'] == c]
            # df_temp = df_temp.drop(columns=['cluster_label'])
            col_name = 'cluster_' + str(int(c))
            df_temp = df_temp.mean(axis=0)

            if df_centr is not None:
                df_centr[col_name] = df_temp
            else:
                df_centr = df_temp.to_frame(name=col_name)

        df_centr = df_centr.round(4)

        if save:
            df_centr.to_csv(my_path, sep='\t', index_label='activity')
            print('centroids saved!')    
    
    
    return df_centr
    

def cluster_kmeans(X, k):
    print('### clustering with kmeans...')
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(X)

    return kmeans,cluster_labels


def find_eps_dbscan(df, min_pts):
    print('### using Nearest Neighbors to find eps...')

    neigh = NearestNeighbors(n_neighbors=min_pts)
    nbrs = neigh.fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances_sort = np.sort(distances, axis=0)
    distances_sort = np.mean(distances_sort, axis=1)
    # distances_sort = distances_sort[:,1]
    distances_sort = list(distances_sort)
    distances_sort.reverse()

    plt.figure(figsize=(15,7))
    plt.plot(distances_sort)
    plt.title('K-distance Graph',fontsize=20)
    plt.xlabel('Data Points sorted by distance',fontsize=14)
    plt.ylabel('k-distance',fontsize=14)
    plt.show(block=True)


def cluster_dbscan(df, eps, min_pts):
    print('### clustering DBSCAN...')

    model = DBSCAN(eps = eps, min_samples = min_pts, metric='euclidean')
 

    return model.fit_predict(df)


def infreq_toofreq_cols(df_mov, min_perc, max_perc):
    df = create_1_gram(df_mov, 'id', 'movimentoCodigoNacional')
    all_cols = list(df.columns)
    df[df != 0] = 1
    total = len(df.index)
    cols_freq = df.sum()
    sel_cols = cols_freq

    if min_perc:
        min_limit = int(min_perc * total)
        sel_cols = sel_cols[(cols_freq > min_limit)]
    
    if max_perc:
        max_limit = int(max_perc * total)
        sel_cols = sel_cols[(cols_freq < max_limit)]
   
    sel_cols = sel_cols.index.tolist()
    rem_cols = [c for c in all_cols if c not in sel_cols]

    print('### Removed Acts by TF-IDF:')
    print(rem_cols)
    print()

    return rem_cols


def rem_infreq_feats(df_gram, min_percent):
    min_val = int(min_percent * len(df_gram))
    
    df_temp = df_gram.copy(deep=True)
    df_temp = df_temp.where(df_temp == 0, 1)
    df_temp = df_temp.sum()
    df_temp = df_temp[df_temp > min_val]
    
    freq_feats = df_temp.index.to_list()


    return df_gram[freq_feats]
   

def binary_search(a, x, lo=0, hi=None):
    if hi is None: hi = len(a)
    pos = bisect_left(a, x, lo, hi)                  # find insertion position
    return pos if pos != hi and a[pos] == x else -1  # don't walk off the end


def create_binary_array(mov_code, act_list, size):
    bin_arr = [0] * size
    pos = binary_search(act_list, mov_code, 0, size)

    if pos != -1:
        bin_arr[pos] = 1

    return bin_arr


def create_1_gram(df_mov, id, act):
    print('### creating 1-gram...')

    act_list = df_mov.drop_duplicates(subset=[act])[act].to_list()
    act_list.sort()
    size = len(act_list)

    print('act_list:')
    print(act_list)

    df_gram = df_mov[[id, act]]

    df_gram['n_gram'] = df_gram.apply(lambda df: create_binary_array(
                            df[act],
                            act_list,
                            size
                            ), axis=1
                        )

    df_gram[act_list] = pd.DataFrame(df_gram['n_gram'].tolist(), 
                                    index=df_gram.index)

    df_gram = df_gram[[id] + act_list]
    df_gram = df_gram.groupby([id], as_index=True).sum()


    return df_gram


def get_all_feats(df, id, act, n):
    df_traces = df.groupby(id).agg(trace=(act,list))
    df_traces = df_traces[df_traces['trace'].map(len) >= n]
    df_traces['n_gram'] = \
        df_traces['trace'].apply(lambda x: list(ngrams(x, n=n)))
    df_traces = df_traces.drop(columns=['trace'])

    flatten = [item for sublist in df_traces['n_gram'] for item in sublist]

    df_traces = df_traces.explode('n_gram')

    return df_traces,list(set(flatten))


def create_n_gram(df, id, act, n):
    if n == 1:
        return create_1_gram(df, id, act)

    print('### creating ' + str(n) + '-gram...')
    df_gram, act_list = get_all_feats(df, id, act, n)
    act_list.sort()
    size = len(act_list)

    df_gram['result'] = df_gram.apply(lambda df_gram: create_binary_array(
                            df_gram['n_gram'],
                            act_list,
                            size
                            ), axis=1
                        )

    df_gram[act_list] = pd.DataFrame(df_gram['result'].tolist(), 
                                     index=df_gram.index)

    df_gram = df_gram.groupby(id).sum()


    return df_gram


def print_cluster_traces(df_mov, df_code_mov, saving_path, file_path):
    print('### printing cluster traces...')

    df = my_utils.get_act_name(df_mov, df_code_mov)
    df = df.sort_values('movimentoDataHora')

    # df = df[df['cluster_label'] != -1]

    df = df.groupby('id').\
            agg({'movimentoCodigoNacional': lambda x: list(x),
                 'cluster_label':'min'})
    df['movimentoCodigoNacional'] = df['movimentoCodigoNacional'].astype(str)
    df = df.groupby('movimentoCodigoNacional', as_index=False).agg(
        count=('cluster_label','count'),
        cluster_label=('cluster_label','min'))

    df.to_csv(saving_path, sep='\t')

    cluster_labels = df['cluster_label'].drop_duplicates().tolist()

    with open(file_path, 'w+') as my_file:
        for c in cluster_labels:
            my_file.write('##############################\n')
            my_file.write('########## CLUSTER ' + str(c) + \
                          ' ##########\n')
            my_file.write('##############################\n\n')
            df_temp = df[df['cluster_label'] == c]
            i = 0

            for index, row in df_temp.iterrows():
                if i > len(df_temp):
                    break
                else:
                    my_file.write(str(row[0]))
                    my_file.write('\n')
                    my_file.write('count: ' + str(row[1]))
                    my_file.write('\n')
                    my_file.write('cluster: ' + str(row[2]))
                    my_file.write('\n\n')

            
    print()


def get_attr_distribution_cluster(df, col):
    df_distrib = df.groupby(['cluster_label',col]).\
        agg(count=('assuntoCodigoNacional','count'))
    df_distrib = df_distrib.reset_index(level=1)

    return df_distrib


def cluster_agglomerative(df, n):
    print('### clustering agglomerative...')

    df_work = df.drop(columns=['cluster_label'])
    X = df_work.to_numpy()
    clustering = AgglomerativeClustering(n_clusters = n, 
                                         linkage = 'average').fit(X)

    cluster_labels = list(clustering.labels_)
    cluster_labels = [c + 100 for c in cluster_labels]

    df_work['cluster_label_new'] = cluster_labels


    return df_work[['cluster_label_new']]




