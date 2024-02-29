import csv
import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from collections import defaultdict
import scipy.stats
from collections import Counter,defaultdict
import pm4py.objects.log.importer.xes.importer  as xes_importer
import pm4py.objects.log.exporter.xes as csv_exporter
import statistics
from pyvis.network import Network
import seaborn as sns
import json
import pm4py

from simprocess.simulation_activity import Distribution

RESOURCE_NAME = 'org:resource'
# RESOURCE_NAME = 'Resource'

def get_input_file(file_path=None):
    resper = ResourcePerformace()
    event_log_address = file_path if file_path else input("Event Log:")
    log_format = event_log_address.split('.')[-1]

    if str(log_format) == 'csv':
        event_log = pd.read_csv(event_log_address)
    elif str(log_format) == 'xes':
        xes_log = xes_importer.apply(event_log_address)
        xes_log = resper.calculate_completed(xes_log)
        dataframe = pm4py.convert_to_dataframe(xes_log)
        event_log = dataframe

    return event_log

class ResourcePerformace:

    def calculate_completed(self, log):
        timetaken = {}
        for trace in log:
            length = len(trace)
            for index, event in enumerate(trace):
                if index < (length - 1):
                    next_event = trace[index + 1]
                    if "concept:name" in event:
                        attribute = event["concept:name"]
                        if "time:complete" in event:
                            if attribute not in timetaken:
                                timetaken[attribute] = [
                                    (event["time:complete"] - event["time:timestamp"]).total_seconds()]
                            else:
                                timetaken[attribute].append(
                                    (event["time:complete"] - event["time:timestamp"]).total_seconds())
                        else:
                            if "time:timestamp" in event:
                                time = event["time:timestamp"]
                            if "time:timestamp" in next_event:
                                next_time = next_event["time:timestamp"]
                            else:
                                next_time = time
                            event['time:complete'] = next_event["time:timestamp"]
                            if attribute not in timetaken:
                                timetaken[attribute] = [(next_time - time).total_seconds()]
                            else:
                                timetaken[attribute].append((next_time - time).total_seconds())
                else:
                    mean = statistics.mean(timetaken[attribute])
                    if "concept:name" in event:
                        attribute = event["concept:name"]
                    if attribute not in timetaken:
                        timetaken[attribute] = [mean]
                    else:
                        timetaken[attribute].append(mean)
                    mean_timedelta = timedelta(seconds=mean)
                    event['time:complete'] = event["time:timestamp"] + mean_timedelta
        return log

    def get_input_file(self, file_path=None):
        event_log_address = file_path if file_path else input("Event Log:")
        log_format = event_log_address.split('.')[-1]

        if str(log_format) == 'csv':
            event_log = pd.read_csv(event_log_address)
        elif str(log_format) == 'xes':
            xes_log = xes_importer.apply(event_log_address)
            xes_log = self.calculate_completed(xes_log)
            dataframe = pm4py.convert_to_dataframe(xes_log)
            event_log = dataframe

        return event_log

    def create_matrix_resource(self,event_log):

        event_log['time:complete'] = pd.to_datetime(event_log['time:complete'])
        event_log['time:timestamp'] = pd.to_datetime(event_log['time:timestamp'])
        event_duration = abs(event_log['time:complete'] - event_log['time:timestamp'])
        event_log['Event Duration'] = event_duration
        # create adjancy matrix of activities
        resource_matrix = pd.DataFrame(np.zeros(shape=(event_log[RESOURCE_NAME].nunique(), event_log[RESOURCE_NAME].nunique())),
                              columns=event_log[RESOURCE_NAME].unique(), index=event_log[RESOURCE_NAME].unique())
        temp_log = event_log.sort_values(['time:timestamp'], ascending=True).groupby('case:concept:name')
        trace = {}

        for case, casegroup in temp_log:
            trace.update({case: casegroup['Resource'].values})

        for key, val in trace.items():
            i = 0
            while i < (len(val) - 1):
                resource_matrix[val[i + 1]][val[i]] += 1
                i += 1

        return resource_matrix

    def remove_outliers(self, dataset, attribute):

        to_remove = []
        tasks = set(dataset["concept:name"].tolist())

        for task in tasks:
            filtered_dataset = dataset[dataset["concept:name"] == task]
            Q1 = filtered_dataset[attribute].quantile(0.25)
            Q3 = filtered_dataset[attribute].quantile(0.75)
            IQR = Q3 - Q1

            UpperWhisker = Q3 + 1.5 * IQR
            LowerWhisker = Q1 - 1.5 * IQR
            # filter = ((filtered_dataset[attribute] > LowerWhisker) & (filtered_dataset[attribute] < UpperWhisker))
            filter = filtered_dataset[attribute] < UpperWhisker
            result = filtered_dataset.loc[filter]

            to_remove.extend(result["case:concept:name"].tolist())

        to_remove = set(to_remove)
        result = dataset
        result = dataset[dataset["case:concept:name"].isin(to_remove) == False]
        return result

    def calculate_duration(self, event_log):
        tasks = set(event_log["concept:name"].tolist())

        dict = {}
        for task in tasks:
            dfr = event_log[event_log["concept:name"] == task]
            dfr = dfr["Event Duration"].apply(datetime.timedelta.total_seconds)
            dict[task] = dfr.tolist()
        dfn = pd.DataFrame.from_dict(dict, orient='index')
        dfr = dfn.transpose()
        dfr.dropna(inplace=True)
        new_timetaken = dfr.to_dict('list')
        distribution = {}
        for attribute in new_timetaken:
            try:
                dst = Distribution()
                distribution[attribute] = dst.Fit(new_timetaken[attribute])
            except:
                distribution = 'few data for distribution fitting'
            new_timetaken[attribute] = statistics.median(new_timetaken[attribute])

        min_value = min(i for i in new_timetaken.values() if i > 0)

        for key, value in new_timetaken.items():
            new_timetaken[key] = int(value/min_value)
        return new_timetaken

    def create_matrix(self, event_log):
        event_log['time:timestamp'] = pd.to_datetime(event_log['time:timestamp'], utc=True)
        event_log['time:complete'] = pd.to_datetime(event_log['time:complete'], utc=True)

        event_duration = abs(event_log['time:complete'] - event_log['time:timestamp'])
        event_log['Event Duration'] = event_duration
        event_log = self.remove_outliers(event_log, "Event Duration")
        act_dur_dict = {}
        temp_act_log = event_log.groupby(['concept:name'])
        for kact, vact in temp_act_log:
            act_dur_dict[kact] = vact['Event Duration'].mean()

        # create adjancy matrix of activities
        matrix = pd.DataFrame(np.zeros(shape=(event_log['concept:name'].nunique(), event_log['concept:name'].nunique())),
                              columns=event_log['concept:name'].unique(), index=event_log['concept:name'].unique())
        temp_log = event_log.sort_values(['time:timestamp'], ascending=True).groupby('case:concept:name')
        trace = {}

        for case, casegroup in temp_log:
            trace.update({case: casegroup['concept:name'].values})

        for key, val in trace.items():
            i = 0
            while i < (len(val)-1):
                matrix[val[i + 1]][val[i]] += 1
                i += 1
        return matrix, act_dur_dict

    def find_resource(self, event_log, remove_outliers=False):
        event_log['time:complete'] = pd.to_datetime(event_log['time:complete'])
        event_log['time:timestamp'] = pd.to_datetime(event_log['time:timestamp'])
        event_duration = abs(event_log['time:complete'] - event_log['time:timestamp'])
        event_log['Event Duration'] = event_duration
        if remove_outliers:
            event_log = self.remove_outliers(event_log, "Event Duration")
        freq_act_res_matrix = pd.DataFrame(np.zeros(shape=(len(event_log[RESOURCE_NAME].unique()), len(event_log['concept:name'].unique()))),
                              columns=event_log['concept:name'].unique(), index=event_log[RESOURCE_NAME].unique())
        dur_act_res_matrix = pd.DataFrame(np.zeros(shape=(len(event_log[RESOURCE_NAME].unique()), len(event_log['concept:name'].unique()))),
            columns=event_log['concept:name'].unique(), index=event_log[RESOURCE_NAME].unique())
        med_dur_act_res_matrix = pd.DataFrame(
            np.zeros(shape=(len(event_log[RESOURCE_NAME].unique()), len(event_log['concept:name'].unique()))),
            columns=event_log['concept:name'].unique(), index=event_log[RESOURCE_NAME].unique())

        act_groupy = event_log.groupby('concept:name')
        for name, group in act_groupy:
                resgroup = group.groupby(RESOURCE_NAME)['Event Duration']
                res_per_act_freq = resgroup.size()
                res_per_act_sum = resgroup.sum()
                median = resgroup.median()
                for res in res_per_act_freq.keys():
                    freq_act_res_matrix[name][res] = res_per_act_freq.get(res)
                    if res_per_act_freq.get(res) != 0 and res_per_act_sum.get(res) != 0:
                        dur_act_res_matrix[name][res] = pd.to_timedelta((res_per_act_sum.get(res))/res_per_act_freq.get(res)).seconds/1
                        med_dur_act_res_matrix[name][res] = pd.to_timedelta(median.get(res)).seconds/1
        max_values = freq_act_res_matrix.max(axis=0)
        filter = freq_act_res_matrix.iloc[:] <= 5
        med_dur_act_res_matrix_filtered = med_dur_act_res_matrix.copy()
        med_dur_act_res_matrix_filtered[filter] = 0
        for col in freq_act_res_matrix.columns:
            res = freq_act_res_matrix[col].idxmax()
            if (med_dur_act_res_matrix_filtered[col] == 0).all():
                med_dur_act_res_matrix_filtered[col].loc[res] = med_dur_act_res_matrix[col].loc[res]
        return freq_act_res_matrix, med_dur_act_res_matrix


def resource_discovery(event_log=None, file_path=None):
    resper = ResourcePerformace()
    if event_log is None:
        event_log = resper.get_input_file(file_path)
    #matrix, act_dur_dict = resper.create_matrix(event_log)
    freq_act_res_matrix, dur_act_res_matrix = resper.find_resource(event_log, False)
    dur_act_res_matrix_norm = dur_act_res_matrix.apply(lambda x: (x / x[x > 0].min()))
    resource_names, resource_el = _transform_resource_df(dur_act_res_matrix_norm)
    #duration = resper.calculate_duration(event_log)
    return resource_names, resource_el
    #return new_names, new_res_el

def _transform_resource_df(df):
    #df = df.apply(lambda x: (x / x[x > 0].min()))
    res_matrix_dict = df.to_dict('index')
    df.reset_index(level=0, inplace=True)
    resource_names = df[['index']].copy()
    resource_names = resource_names.to_dict('index')
    resource_el = dict()
    for key, value in resource_names.items():
        name = value['index']
        resource_names[key] = name
        res_matrix_dict[key] = res_matrix_dict[name]
        del res_matrix_dict[name]
    for k1, v1 in res_matrix_dict.items():
        for k2, v2 in v1.items():
            if v2 > 0:
                k2 = k2.replace(" ", "")
                if k2 not in resource_el.keys():
                    resource_el[k2] = {}
                resource_el[k2][k1] = int(v2)
    return resource_names, resource_el

if __name__ =="__main__":
    x = resource_discovery("hospital_test.csv")
    pass
    # resper = ResourcePerformace()
    # event_log = resper.get_input_file("hospital.csv")
    # matrix, act_dur_dict = resper.create_matrix(event_log)
    # freq_act_res_matrix, dur_act_res_matrix = resper.find_resource(event_log)
    # resper.find_roles(freq_act_res_matrix)
    # resper.draw_matrix(matrix, freq_act_res_matrix, dur_act_res_matrix, act_dur_dict)