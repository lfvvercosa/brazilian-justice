import pandas as pd
import pm4py


def create_petrinet_inductive(log, noise_thres):
    net, im, fm = pm4py.discover_petri_net_inductive(
                    log,
                    activity_key='concept:name', 
                    case_id_key='case:concept:name', 
                    timestamp_key='time:timestamp',
                    noise_threshold=noise_thres
                  )
    

    return net,im,fm


def create_tree_inductive(log, noise_thres):
    process_tree = pm4py.discover_process_tree_inductive(
                    log,
                    activity_key='concept:name', 
                    case_id_key='case:concept:name', 
                    timestamp_key='time:timestamp',
                    noise_threshold=noise_thres
                  )
    

    return process_tree


def create_dfg(log, performance=True):
    if performance:
        func = pm4py.discover_performance_dfg
    else:
        func = pm4py.discover_dfg

    dfg,sa,ea = func(log=log,
                     activity_key='concept:name',
                     timestamp_key='time:timestamp',
                     case_id_key='case:concept:name',
                    )
    
    return dfg,sa,ea


def update_sa_ea(dfg,sa,ea):
    if dfg:
        elements_temp = list(set(dfg.elements()))
        elements = set()
        elements_sa = list(sa.keys())
        elements_ea = list(ea.keys())

        for e in elements_temp:
            elements.add(e[0])
            elements.add(e[1])
        
        for e in elements_sa:
            if e not in elements:
                del sa[e]

        for e in elements_ea:
            if e not in elements:
                del ea[e]
    
    return sa,ea


def rem_noise_dfg(dfg, sa, ea, noise_thres):
    if dfg:
        my_max = max(dfg.most_common(1)[0][-1],
                     max(sa.values()))
        my_min = int(noise_thres * my_max)
        del_list = []
        elements = set(dfg.elements())

        for e in elements:
            if dfg[e] < my_min:
                del_list.append(e)
        
        for e in del_list:
            del dfg[e]
    
    sa = rem_noise_sa_ea(sa, noise_thres)
    ea = rem_noise_sa_ea(ea, noise_thres)
    sa,ea = update_sa_ea(dfg,sa,ea)


    return dfg,sa,ea


def rem_noise_sa_ea(d, noise_thres):
    my_max = max(d.values())
    my_min = int(noise_thres * my_max)
    elements = list(d.keys())

    for e in elements:
        if d[e] < my_min:
            del d[e]
    
    
    return d


def rem_noise_dfg_perf(dfg_perf, dfg_freq, sa, ea, noise_thres):
    my_max = max(dfg_freq.most_common(1)[0][-1],
                  max(sa.values()))   
    my_min = int(noise_thres * my_max)
    del_list = []
    elements = set(dfg_freq.elements())

    for e in elements:
        if dfg_freq[e] < my_min:
            del_list.append(e)
    
    for e in del_list:
        del dfg_freq[e]
        del dfg_perf[e]
    
    sa = rem_noise_sa_ea(sa, noise_thres)
    ea = rem_noise_sa_ea(ea, noise_thres)
    sa,ea = update_sa_ea(dfg_freq, sa, ea)


    return dfg_perf,sa,ea


def rem_noise_dfg_percent(dfg, sa, ea, perc_keep):
    all_edges = dfg.most_common()
    all_edges += [(('-',k), sa[k]) for k in sa]
    all_edges += [((k,'-'), ea[k]) for k in ea]
    all_edges = sorted(all_edges, key=lambda tup: tup[1])

    remove = []
    perc_remove = 1 - perc_keep
    n = int(perc_remove * len(all_edges))
    count = 0

    while(count < n):
        remove.append(all_edges[count][0])
        count += 1

    for edge in remove:
        if edge[0] == '-':
            del sa[edge[1]]
        elif edge[1] == '-':
            del ea[edge[0]]
        else:
            del dfg[edge]

    print()

    return dfg,sa,ea


def get_inbound_edges(edges, act):
    count = 0

    for e in edges:
        if e[1] == act:
            count += 1
    
    return count


def get_outbound_edges(edges, act):
    count = 0

    for e in edges:
        if e[0] == act:
            count += 1
    
    return count


def remove_in_dfg(edge, dfg, sa, ea):
    if edge[0] == '-':
            del sa[edge[1]]
    elif edge[1] == '-':
        del ea[edge[0]]
    else:
        del dfg[edge]


def rem_noise_dfg_with_excep(dfg, sa, ea, noise_thres):
    all_edges = dfg.most_common()
    all_edges += [(('-',k), sa[k]) for k in sa]
    all_edges += [((k,'-'), ea[k]) for k in ea]
    all_edges = sorted(all_edges, key=lambda tup: tup[1])

    my_max = all_edges[-1][1]
    my_min = int(noise_thres * my_max)
    del_list = []
    edges = [e[0] for e in all_edges]

    for e in all_edges:
        if e[1] < my_min:
            del_list.append(e[0])

    for e in del_list:
        if get_outbound_edges(edges, e[0]) > 1 and \
           get_inbound_edges(edges, e[1]) > 1:
                edges.remove(e)
                remove_in_dfg(e, dfg, sa, ea)


    return dfg,sa,ea
    

def rem_noise_dfg_perf_with_excep(dfg, 
                                  sa, 
                                  ea, 
                                  dfg_perf,
                                  sa_perf,
                                  ea_perf,
                                  noise_thres,
                                 ):
    all_edges = dfg.most_common()
    all_edges += [(('-',k), sa[k]) for k in sa]
    all_edges += [((k,'-'), ea[k]) for k in ea]
    all_edges = sorted(all_edges, key=lambda tup: tup[1])

    my_max = all_edges[-1][1]
    my_min = int(noise_thres * my_max)
    del_list = []
    edges = [e[0] for e in all_edges]

    for e in all_edges:
        if e[1] < my_min:
            del_list.append(e[0])

    for e in del_list:
        if get_outbound_edges(edges, e[0]) > 1 and \
           get_inbound_edges(edges, e[1]) > 1:
                edges.remove(e)
                remove_in_dfg(e, dfg, sa, ea)
                remove_in_dfg(e, dfg_perf, sa_perf, ea_perf)


    return dfg_perf, sa_perf, ea_perf
