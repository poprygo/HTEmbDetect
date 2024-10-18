# in this file I query exported data and draw plots

from cmath import inf
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandasql import sqldf

def query_how_many_distinguishable():

    df = pd.read_excel("metrics_all.xlsx")
    df_filtered = df[df["metrics"] != inf]

    query_mean = df_filtered.groupby(['name', 'class']).mean('metrics')
    query_std = df_filtered.groupby(['name', 'class']).std()
    query_max = df_filtered.groupby(['name', 'class'], as_index=False).max()

    q = """
        SELECT name, class, AVG(metrics) FROM df_filtered 
        GROUP BY name, class 
        """
    # print(sqldf(q, globals()))
    print(query_mean)
    print(query_std)

    # selecting all combinatorial safe metrics
    query_safe_c = df_filtered.loc[(df_filtered['class'] == 0) & (
        df_filtered['name'].str.contains('c'))]

    query_trojan_c = df_filtered.loc[(df_filtered['class'] == 1) & (
        df_filtered['name'].str.contains('c'))]

    print(query_safe_c['metrics'])

    print(query_max)

    safes = query_safe_c['metrics'].to_numpy()
    trojans = query_trojan_c['metrics'].to_numpy()

    # plt.hist(safes)
    # plt.hist(trojans)

    # plt.show()
    print(query_max)
    max_safes = query_max.loc[query_max['class'] == 0]['metrics'].to_numpy()
    max_trojans = query_max.loc[query_max['class'] == 1]['metrics'].to_numpy()

    q = """
        SELECT t1.metrics / t2.metrics as ratio,
        t1.name
            from (select metrics, name from query_max where class = 1) as t1
            join
            (select metrics, name from query_max where class = 0) as t2
            on
            t1.name = t2.name
        """
    ratios = sqldf(q, globals())
    c_ratios = ratios.loc[ratios['name'].str.contains('c')]
    s_ratios = ratios.loc[ratios['name'].str.contains('s')]
    print('Percent of detectable in c_rations: ', len(c_ratios[c_ratios['ratio'] > 1]) / len(c_ratios))
    print('Percent of detectable in s_rations: ', len(
        s_ratios[s_ratios['ratio'] > 1]) / len(s_ratios))

def query_tradeof_threshold():
    from matplotlib import pyplot as plt
    df = pd.read_excel("distances_to_center.xlsx")
    ht_distances = df['distances'].to_numpy()
    ht_distances = np.sort(ht_distances)

    distances = [i for i in np.arange(0.0, 1.0, 0.005)]
    ht_percentages  = np.searchsorted(ht_distances, distances)
    
    ht_percentages = [i / len(ht_distances) for i in ht_percentages]


    df2 = pd.read_excel("distances_all_embeddings_to_center3.xlsx")
    emb_distances = df2['distances'].to_numpy()
    emb_distances = np.sort(emb_distances)

    emb_percentages  = np.searchsorted(emb_distances, distances)
    emb_percentages = [len(emb_distances) / i for i in emb_percentages]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Fraction of HTs covered')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Search space reduction, times')
    ax1.plot(distances, ht_percentages, label = 'Fraction of HTs covered')
    
    ax2.plot(distances, emb_percentages, color = 'orange', label = 'Search space reduction')
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.set_ylim((0, 20))
    plt.xlim((0.00, 0.15))
    plt.legend(handles+handles2, labels+labels2,loc='center right')
    plt.show()
    


def query_fow_many_near_center():
    from matplotlib import pyplot as plt

    df = pd.read_excel("distances_to_center.xlsx")
    query_distances_main_c = df.loc[(df['bench_types'] == 1 ) & (df['centroid_ids'] == 0)]
    data1 = query_distances_main_c['distances'].to_numpy()
    query_distances_multimodal_c = df.loc[(
        df['bench_types'] == 1) & (df['centroid_ids'] != 0)]
    data2 = query_distances_multimodal_c['distances'].to_numpy()

    bins = np.linspace(0, 0.5, 100)
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.hist(data1, bins, alpha=0.5,
                label='Distance to the main embeddings distribution centroid')

    plt.hist(data2, bins, alpha=0.5,
             label='Distance to the secondary embeddings distribution centroid')
    plt.legend(loc='upper right')
    ax.set_title ('Distance from the HT embeddings centroid, combinatorial IC')

    plt.show()

    ########################################

    query_distances_main_c = df.loc[(df['bench_types'] == 0) & (df['centroid_ids'] == 0)]
    data1 = query_distances_main_c['distances'].to_numpy()
    query_distances_multimodal_c = df.loc[(
        df['bench_types'] == 0) & (df['centroid_ids'] != 0)]
    data2 = query_distances_multimodal_c['distances'].to_numpy()

    bins = np.linspace(0, 0.5, 100)
    ax = fig.add_subplot()
    plt.hist(data1, bins, alpha=0.5,
                label='Distance to the main embeddings distribution centroid')

    plt.hist(data2, bins, alpha=0.5,
             label='Distance to the secondary embeddings distribution centroid')
    plt.legend(loc='upper right')
    plt.title(label=
        'Distance from the HT embeddings centroid, sequential IC')

    plt.show()


def stability_analysis():
    df1 = pd.read_excel("stability_data1.xlsx")
    df2 = pd.read_excel("stability_data.xlsx")
    df_total = pd.concat([df1, df2])
    q = """
        SELECT COUNT(bench_prediction_with_ht) as n_detected, bench_type
        FROM df_total 
        WHERE bench_prediction_with_ht = 1
        GROUP BY bench_id
        """
    stats = sqldf(q, {'df_total': df_total})


    hist_comb = stats.loc[(stats['bench_type'] == 0)]['n_detected'].to_numpy()
    hist_seq = stats.loc[(stats['bench_type'] == 1)]['n_detected'].to_numpy()
    
    bins = np.linspace(0, 13, 14)
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.hist(hist_comb, bins, alpha=0.5,
             label='Combinational HTs detection over multiple trainings')

    plt.hist(hist_seq, bins, alpha=0.5,
             label='Sequentional HTs detection over multiple trainings')
    plt.legend(loc='upper left')
    plt.title('Detection stability analysis over 13 training runs')

    plt.show()

def stability_analysis_with_names():
    df1 = pd.read_excel("stability_data2.xlsx")
    df2 = pd.read_excel("stability_data3.xlsx")
    df_total = pd.concat([df1, df2])
    q = """
        SELECT COUNT(bench_prediction_with_ht) as n_detected, bench_type
        FROM df_total 
        WHERE bench_prediction_with_ht = 1
        GROUP BY bench_id
        """
    stats = sqldf(q, {'df_total': df_total})


    hist_comb = stats.loc[(stats['bench_type'] == 0)]['n_detected'].to_numpy()
    hist_seq = stats.loc[(stats['bench_type'] == 1)]['n_detected'].to_numpy()
    
    bins = np.linspace(0, 13, 14)
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.hist(hist_comb, bins, alpha=0.5,
             label='Combinational HTs detection over multiple trainings')

    plt.hist(hist_seq, bins, alpha=0.5,
             label='Sequentional HTs detection over multiple trainings')
    plt.legend(loc='upper left')
    plt.title('Detection stability analysis over 10 training runs')

    plt.show()

    q = """
        Select bench_id, bench_name, n_detected from
        (SELECT COUNT(bench_prediction_with_ht) as n_detected, bench_id, bench_name
        FROM df_total 
        WHERE bench_prediction_with_ht = 1
        AND bench_type = 1
        GROUP BY bench_id, bench_name)
        where n_detected < 4
        order by n_detected
        """
    
    outliers = sqldf(q, {'df_total': df_total})
    outliers.to_excel('outliers.xlsx')

#stability_analysis()
##stability_analysis_with_names()
#query_fow_many_near_center()
query_tradeof_threshold()