import argparse
import os
import random
from pathlib import Path
from datetime import datetime, timedelta
import statistics
from pm4py.objects.log.importer.xes import importer as xes_import_factory
# from pm4py.objects.log.adapters.pandas import csv_import_adapter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as conversion_factory
import math
import warnings
from pm4py.util import xes_constants
from pm4py.util import constants
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

case_id_key = xes_constants.DEFAULT_TRACEID_KEY
activity_key = xes_constants.DEFAULT_NAME_KEY
timestamp_key = datetime.now()
results = []

warnings.filterwarnings('ignore')


def read_input_file_path():
    """
        Reads the input file path from the Command Line Interface and verifies if the file exists
        Returns
        --------------
        file.file_path
                The file path of the input event log file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=Path)
    file = parser.parse_args()
    print("File received: ", file.file_path)
    if file.file_path.exists():
        print("File exists")
    else:
        print("File does not exist. Please input correct file")
        exit()
    return str(file.file_path)


def import_xes(file_path):
    """
        Imports logs from the input xes file
        Parameters
        --------------
        file_path
            Path of the input event log file
        Returns
        --------------
        xes_log
            The input event logs in the form of a log
        """
    xes_log = xes_import_factory.apply(file_path)
    print("Import of xes successful,with {0} traces in total".format(len(xes_log)))
    return xes_log


def import_csv(file_path):
    """
            Imports logs from the input csv file
            Parameters
            --------------
            :param file_path:
                The path to the csv log file
            Returns
            --------------
            csv_log
                The input event logs in the form of a log
            """
    data_frame = pd.read_csv(
        os.path.join(file_path), sep=",")

    log_csv = dataframe_utils.convert_timestamp_columns_in_df(data_frame)
    log_csv = log_csv.sort_values('time:timestamp')
    event_log = log_converter.apply(log_csv)
    print("Import of csv successful,with {0} traces in total".format(len(event_log)))

    return event_log


def verify_extension_and_import(file_path=None):
    """
            This function verifies that the extension of the event log file is .xes or .csv and imports the
            logs from those files
            Returns
            --------------
            log
                The input event logs in the form of a log
            """

    # file_path ="Prozessmodel.xes"
    if not file_path:
        file_path = read_input_file_path()
    file_name, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.replace("'))", "")
    print("File Extension: ", file_extension)
    if file_extension == ".xes":
        log = import_xes(file_path)
        return log
    elif file_extension == ".csv":
        log = import_csv(file_path)
        return log
    else:
        print("Unsupported extension. Supported file extensions are .xes and .csv ONLY")
        exit()


def remove_outliers(dataset, attribute):
    Q1 = dataset[attribute].quantile(0.25)
    Q3 = dataset[attribute].quantile(0.75)
    IQR = Q3 - Q1

    UpperWhisker = Q3 + 1.5 * IQR
    LowerWhisker = Q1 - 1.5 * IQR

    filter = (dataset[attribute] >= LowerWhisker) & (dataset[attribute] <= UpperWhisker)

    result = dataset.loc[filter]

    return result


class Distribution(object):
    def __init__(self, dist_name_list=None):
        self.dist_names = ['norm', 'lognorm', 'expon']
        # self.dist_names = ['norm']
        self.dist_results = []
        self.params = {}

        self.DistributionName = ""
        self.PValue = 0
        self.Param = None

        self.isFitted = False

    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)
            self.params[dist_name] = param
            # Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param)
            self.dist_results.append((dist_name, p))
        # select the best fitted distribution
        sel_dist, p = (max(self.dist_results, key=lambda item: item[1]))
        # store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p

        self.isFitted = True
        # print("Best fitted distribution and the p value are:", self.DistributionName,self.PValue)

        return self.DistributionName, self.PValue


def create_methods(log, multiplier=1, file_path=None):
    """
                This function calculates the average time taken for each activity and writes methods to method.py file
                to be used for simulation
                """
    if not log:
        log = verify_extension_and_import(file_path)
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
                                abs((event["time:complete"] - event["time:timestamp"]).total_seconds())]
                        else:
                            timetaken[attribute].append(
                                abs((event["time:complete"] - event["time:timestamp"]).total_seconds()))
                    else:
                        if "time:timestamp" in event:
                            time = event["time:timestamp"]
                        if "time:timestamp" in next_event:
                            next_time = next_event["time:timestamp"]
                        else:
                            next_time = time
                        if attribute not in timetaken:
                            timetaken[attribute] = [abs((next_time - time).total_seconds())]
                        else:
                            timetaken[attribute].append(abs((next_time - time).total_seconds()))
            else:
                mean = abs(statistics.mean(timetaken[attribute]))
                if "concept:name" in event:
                    attribute = event["concept:name"]
                if attribute not in timetaken:
                    timetaken[attribute] = [mean]
                else:
                    timetaken[attribute].append(mean)

    dfn = pd.DataFrame.from_dict(timetaken, orient='index')
    dfr = dfn.transpose()
    dfr.dropna(inplace=True)

    cleaned = dfr
    for col in dfr.columns:
        cleaned = remove_outliers(cleaned, col)
    dfr = cleaned
    new_timetaken = dfr.to_dict('list')
    distribution = {}
    for attribute in new_timetaken:
        try:
            dst = Distribution()
            distribution[attribute] = dst.Fit(new_timetaken[attribute])
        except:
            distribution = 'few data for distribution fitting'
        new_timetaken[attribute] = statistics.median(new_timetaken[attribute])

    attributes = {}
    for trace in log:
        for event in trace:
            if "concept:name" in event:
                attribute = event["concept:name"]
                if attribute not in attributes:
                    attributes[attribute] = math.ceil(new_timetaken[attribute])
    print(attributes)

    attributes_dict = _prepare_methods_dict(attributes, multiplier)
    return attributes_dict, log


def _prepare_methods_dict(methods, multiplier=1):
    fixed_methods = {}
    smallest_time = math.inf
    for method, duration in methods.items():
        if smallest_time > duration > 0:
            smallest_time = duration
        fixed_methods[str(method).replace(" ", "")] = duration
    for method, duration in fixed_methods.items():
        fixed_methods[method] = multiplier * round(fixed_methods[method] / smallest_time) if round(fixed_methods[method] / smallest_time) < 300 else 2 * multiplier
    return fixed_methods

if __name__ == '__main__':
    create_methods()

