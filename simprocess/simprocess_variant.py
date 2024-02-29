import csv
import math
import statistics
from datetime import datetime
from itertools import compress

from pm4py.algo.discovery.inductive import algorithm as inductive_miner
import pandas as pd
from copy import copy, deepcopy
from pm4py.util import constants
import random
#from pm4py.objects.petri import semantics
from simprocess.ResourcePerformanceExtraction import resource_discovery, get_input_file
import pprint
from pm4py import filter_variants_by_coverage_percentage

import simprocess.simulation_activity as simulation_activity
diff_counter = 0
random.seed()

class ProcessCase:
    def __init__(self, case_id, process, variant, variant_expected_duration, variant_deadline, start_step):
        self.id = case_id
        self.variant = variant
        self.process = process
        self.task_start_step = None
        self.process_start_step = start_step
        self.task_deadline = None
        self.process_case_deadline = self.process_start_step + variant_deadline
        self.process_case_expected_duration = variant_expected_duration
        self.task_waiting_time = 0
        self.process_waiting_time = 0
        self.process_processing_time = 0
        self.process_response_time = -1
        self.time_length = 0
        self.next_task = None
        self.transition = None
        self.allocated_resource = None
        self.next_task_duration = None

    def is_terminal_state(self):
        return len(self.variant) == 0

    def pick_next_task(self, counter):
        if self.is_terminal_state():
            return False

        task_name = self.variant[0]
        del self.variant[0]

        #or self.process.task_duration[task_name] == 0
        while task_name not in self.process.tasks:
            if self.is_terminal_state():
                return False
            task_name = self.variant[0]
            del self.variant[0]
            self.next_task_duration = 0

        self.next_task = task_name
        self.next_task_duration = 0
        return True

    def is_resource_eligible(self, task, resource):
        return getattr(self.process.trace, 'resource_eligibility')(task, resource)

    def is_resource_eligible_new(self, method, resource):
        if method is None or resource is None:
            return None
        if method not in self.process.resource_eligibilities:
            return None
        if not (resource in self.process.resource_eligibilities[method].keys()):
            return None
        return self.process.resource_eligibilities[method][resource]

    def __str__(self):
        return self.next_task

    @property
    def task_slack_time(self):
        return self.task_deadline - self.task_start_step - self.next_task_duration

    @property
    def process_case_slack_time(self):
        return self.process_case_deadline - self.process_start_step - (self.process_case_expected_duration - self.time_length)


class Process:
    def __init__(self, id, methods, variants_dict):
        self.id = id
        self.resource_names = {}
        self.resource_eligibilities = {}
        self.trace = None
        self.tasks = methods
        self.expected_task_duration = {}
        self.task_deadline = {}
        self.expected_variant_duration = []
        self.variant_deadline = []
        #self.task_duration = methods
        self.variants = variants_dict

    def pick_variant(self):
        variant = random.choices(self.variants['variants'], weights=self.variants['weights'], k=1)[0].copy()
        variant_index = self.variants['variants'].index(variant)
        expected_duration = self.expected_variant_duration[variant_index]
        deadline = self.variant_deadline[variant_index]
        return variant, expected_duration, deadline


class Simulation:
    def __init__(self, available_resources, process_event_logs, num_of_initial_process_cases, queue_capacity_modifier, state_repr, logging=False):
        processes = {}
        self.nn_eligibilites = []
        self.task_counter = {}
        self.task_duration = {}
        self.resource_eligibilities = {}
        for ix, file_path in enumerate(process_event_logs):
            event_log = get_input_file(file_path)
            filtered_log = filter_variants_by_coverage_percentage(event_log, 0.1)
            distinct_resources = set()
            self.resource_names, resource_eligibilities = resource_discovery(event_log)
            self.resource_eligibilities.update(resource_eligibilities)
            for method in resource_eligibilities.values():
                for res in method.keys():
                    distinct_resources.add(res)
            self.distinct_resources = list(distinct_resources)
            log = simulation_activity.verify_extension_and_import(file_path)
            methods, variants_dict = discover_process_model(log)
            available_methods = list(resource_eligibilities.keys())
            x = sum(variants_dict["weights"])
            for k, v in resource_eligibilities.items():
                for k2, v2 in v.items():
                    resource_eligibilities[k][k2] = math.ceil(resource_eligibilities[k][k2])

            if 'Confirmationofreceipt' in resource_eligibilities.keys():
                resource_eligibilities['Confirmationofreceipt'] = {0: 3}
            if 'XConfirmationofreceipt' in resource_eligibilities.keys():
                resource_eligibilities['XConfirmationofreceipt'] = {0: 3}

            processes[ix] = Process(ix, available_methods, variants_dict)
            processes[ix].resource_names, processes[ix].resource_eligibilities = self.resource_names, resource_eligibilities

            if ix == 1:
                processes[ix].variants["weights"] = [120, 55, 25, 84, 14, 231, 36, 154, 92, 333, 73, 65, 10]

            for task, el in resource_eligibilities.items():
                processes[ix].expected_task_duration[task] = math.ceil(statistics.mean(el.values()))
                processes[ix].task_deadline[task] = max(el.values())
            for variant in processes[ix].variants['variants']:
                expected_duration = 0
                deadline = 0
                for task in variant:
                    expected_duration += processes[ix].expected_task_duration[task]
                    deadline += processes[ix].task_deadline[task]
                processes[ix].expected_variant_duration.append(math.ceil(expected_duration))
                processes[ix].variant_deadline.append(math.ceil(deadline))
            expected_duration_mean = statistics.mean([x for x in processes[ix].expected_variant_duration])
            deadline_mean = statistics.mean([x for x in processes[ix].variant_deadline])
            processes[ix].expected_variant_duration = list()
            processes[ix].variant_deadline = list()
            for _ in processes[ix].variants['variants']:
                processes[ix].expected_variant_duration.append(math.ceil(expected_duration_mean))
                processes[ix].variant_deadline.append(math.ceil(deadline_mean))
            self.resource_eligibilities.update(resource_eligibilities)

        self.task_counter = dict.fromkeys(self.resource_eligibilities.keys(), 0)

        self.task_duration = {k: [] for k in self.resource_eligibilities.keys()}
        self.task_resources = {k: [] for k in self.resource_eligibilities.keys()}
        self.all_resources = list(range(0, len(self.distinct_resources)))
        self.available_resources = self.all_resources.copy()
        self.processes = processes
        self.processes_counters = {p: 0 for p in processes}
        self.enabled_process_cases = list()
        self.current_process_cases = list()
        self.completed_process_cases = list()
        self.process_case_index = 0
        self.logging = logging
        if self.logging:
            self.logger_timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")


        self.completed_counter = {p: 0 for p in processes}
        self.task_count = sum([len(p.tasks) for p in processes.values()])
        self.action_space = [len(self.all_resources), self.task_count]
        self.state_repr = state_repr
        self.step_counter = 0
        self.process_case_probability = 0.007
        self.resource_utilization = {}
        self.idle_waiting_time = 0


        self.task_for_id = {}
        self.id_for_task = {}
        self.C = 1000.0
        i = 0
        for p in processes.values():
            for task in p.tasks:
                self.task_for_id[i] = task
                self.id_for_task[task] = i
                i += 1

        for task, el in self.resource_eligibilities.items():
            for resource in el.keys():
                self.nn_eligibilites.append([resource, self.id_for_task[task]])
        print(self.nn_eligibilites)

        self.queue_capacity = queue_capacity_modifier * 60 * len(self.all_resources)
        self.initialize_process_cases(num_of_initial_process_cases)

    def initialize_process_cases(self, num_of_process_cases):
        for i in range(num_of_process_cases):
            process = random.choice(list(self.processes.values()))
            id = process.id
            process_case_count = self.processes_counters[id]
            variant, expected_duration, deadline = process.pick_variant()
            new_process_case = ProcessCase(process_case_count, process, variant, expected_duration, deadline, self.step_counter)
            new_process_case.pick_next_task(self.task_counter)
            new_process_case.task_start_step = self.step_counter
            new_process_case.process_start_step = self.step_counter
            new_process_case.task_deadline = new_process_case.process.task_deadline[new_process_case.next_task] + new_process_case.task_start_step
            self.enabled_process_cases.append(new_process_case)
            self.task_counter[new_process_case.next_task] += 1
            self.processes_counters[id] += 1

    def get_action_from_int(self, action):
        if action == len(self.nn_eligibilites):
            return [-1, -1]
        return self.nn_eligibilites[action]

    def get_action_from_int_all(self, action):
        nmb_of_tasks = len(self.resource_eligibilities.keys())
        if action == 0:
            return [-1, -1]
        else:
            action = action - 1
            resource = math.floor(action / nmb_of_tasks)
            task = action % nmb_of_tasks
            return [resource, task]

    def step(self, action):
        reward = 0
        [resource, task_id] = action
        is_enabled = False
        factor = None

        if self._is_action_valid(action):
            is_enabled = self._task_enabled(resource, task_id)
            task = self.task_for_id[task_id]
            self.enabled_process_cases = sorted(self.enabled_process_cases, key=lambda x: x.time_length, reverse=True)
            chosen_process_case = next(filter(lambda t: t.next_task == task, self.enabled_process_cases), None)
            if chosen_process_case:
                factor = chosen_process_case.is_resource_eligible_new(task, resource)

        if self._is_action_valid(action) and is_enabled and (resource in self.available_resources) \
                and factor:
            reward = self._assign_resource(chosen_process_case, resource, factor)

        self._handle_completed_tasks()
        self._increase_waiting_time()
        if random.random() < self.process_case_probability and not self._is_queue_full:
            self.initialize_process_cases(1)
        self.increase_resources_utilization()
        self.step_counter += 1

        return self.state, reward

    def step_fifo(self):
        return self._step_heuristic("fifo")

    def step_spt(self):
        return self._step_heuristic("spt")

    def step_lst_task(self):
        return self._step_heuristic("lst_task")

    def step_lst_process_case(self):
        return self._step_heuristic("lst_process_case")

    def step_edf_task(self):
        return self._step_heuristic("edf_task")

    def step_edf_process_case(self):
        return self._step_heuristic("edf_process_case")

    @staticmethod
    def _is_action_valid(action):
        return action != [-1, -1]

    def _handle_completed_tasks(self):
        to_delete = self.current_process_cases[:]
        for process_case in self.current_process_cases:
            process_case.next_task_duration -= 1
            if process_case.next_task_duration <= 0:
                self.available_resources.append(process_case.allocated_resource)
                process_case.allocated_resource = None
                next = process_case.pick_next_task(self.task_counter)
                process_case.task_waiting_time = 0
                process_case.task_start_step = self.step_counter
                if not next:
                    self.completed_counter[process_case.process.id] += 1
                    self.completed_process_cases.append(process_case)
                else:
                    self.enabled_process_cases.append(process_case)
                    self.task_counter[process_case.next_task] += 1
                to_delete.remove(process_case)
        self.current_process_cases = to_delete

    def _assign_resource(self, process_case, resource, factor=1):
        C = self.C
        reward = 0
        if process_case.next_task == "Confirmationofreceipt" or process_case.next_task == "XConfirmationofreceipt":
            process_case.process_response_time = self.step_counter - process_case.task_start_step
        self.available_resources.remove(resource)
        self.enabled_process_cases.remove(process_case)
        self.task_counter[process_case.next_task] -= 1

        process_case.next_task_duration = math.ceil(factor)
        self.task_duration[process_case.next_task].append(process_case.next_task_duration)
        self.task_resources[process_case.next_task].append(resource)
        if len(process_case.variant) == 0:
            reward = C
            # reward = C/math.sqrt(self.step_counter) if self.step_counter else C
            # reward += (C / process_case.time_length) if process_case.time_length else C
            # reward += (C / self._process_case_queue_waiting_time) if self._process_case_queue_waiting_time else C
        process_case.allocated_resource = resource
        self.current_process_cases.append(process_case)
        if self.logging:
            with open(str(self.logger_timestamp) + '.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                start_timestamp = self._prepare_timestamp(self.step_counter)
                end_timestamp = self._prepare_timestamp(self.step_counter + factor)
                writer.writerow([process_case.id, process_case.next_task, start_timestamp, end_timestamp, process_case.allocated_resource])
        # reward += ((C * 0.2) / process_case.time_length) if process_case.time_length else (C * 0.2)
        return reward

    def _prepare_timestamp(self, timestamp):
        timestamp_over = timestamp // 1000
        timestamp_under = str(timestamp % 1000)
        timestamp_over = str(timestamp_over).zfill(2)
        return timestamp_over + ":" + timestamp_under

    def reset(self):
        self.processes_counters = {p: 0 for p in self.processes}
        self.completed_counter = {p: 0 for p in self.processes}
        self.task_counter = dict.fromkeys(self.resource_eligibilities.keys(), 0)
        self.task_duration = {k: [] for k in self.resource_eligibilities.keys()}
        self.task_resources = {k: [] for k in self.resource_eligibilities.keys()}
        self.enabled_process_cases.clear()
        self.current_process_cases.clear()
        self.completed_process_cases.clear()
        self.available_resources = list(self.all_resources)
        self.resource_utilization = dict()
        self.step_counter = 0
        self.idle_waiting_time = 0
        random.seed()
        if self.logging:
            self.logger_timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")
        return self.state

    def soft_reset(self):
        random.seed()
        self.processes_counters = {p: 0 for p in self.processes}
        self.completed_counter = {p: 0 for p in self.processes}
        #self.task_counter = dict.fromkeys(self.resource_eligibilities.keys(), 0)
        self.resource_utilization = dict()
        self.step_counter = 0
        self.idle_waiting_time = 0
        if self.logging:
            self.logger_timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")
        return self.state


    @property
    def state(self):
        if self.state_repr == "std":
            return self._state_std()
        elif self.state_repr == "a0":
            return self._state_a0()
        elif self.state_repr == "a1":
            return self._state_a1()
        # elif self.state_repr == "a2":
        #     return self._state_a2()
        elif self.state_repr == "a10":
            return self._state_a10()

    def _state_a0(self):
        num_of_resources = len(self.all_resources)
        state_resource_list = [0] * num_of_resources
        state_task_list = [0] * self.task_count
        for case in self.current_process_cases:
            state_resource_list[case.allocated_resource] = 1

        state_task_list = [1 if value > 0 else 0 for key, value in self.task_counter.items()]
        # for i in range(self.task_count):
        #     task = self.task_for_id[i]
        #     no_of_task_instances = self.task_counter.get(task, 0)
        #     if no_of_task_instances == 0:
        #         state_task_list[i] = 0
        #     else:
        #         state_task_list[i] = 1
        # return pd.Series(state_resource_list + state_task_list)
        state_resource_list.extend(state_task_list)
        return state_resource_list

    def _state_a10(self):
        num_of_resources = len(self.all_resources)
        state_resource_list = [-1] * num_of_resources
        state_task_list = [0] * self.task_count
        for case in self.current_process_cases:
            state_resource_list[case.allocated_resource] = self.id_for_task[case.next_task]
        state_task_list = [round(value/len(self.enabled_process_cases)) if value > 0 else 0 for key, value in self.task_counter.items()]
        state_resource_list.extend(state_task_list)
        return state_resource_list


    def _state_a1(self):
        num_of_resources = len(self.all_resources)
        state_resource_list = [-1] * num_of_resources
        state_task_list = [0] * self.task_count
        for case in self.current_process_cases:
            state_resource_list[case.allocated_resource] = self.id_for_task[case.next_task]
        state_task_list = [value if value > 0 else 0 for key, value in self.task_counter.items()]
        state_resource_list.extend(state_task_list)
        return state_resource_list

    def first_visit_viable(self):
        return self._task_enabled(None, 3) and any(x in [0, 4, 8, 11] for x in self.available_resources)

    def count_processes(self):
        task_dict = {id: 0 for id in range(len(self.id_for_task.keys()))}
        for task in self.enabled_process_cases:
            id = self.id_for_task[task.next_task]
            task_dict[id] += 1
        for task in self.current_process_cases:
            id = self.id_for_task[task.next_task]
            task_dict[id] += 1
        return list(task_dict.values())

    def is_action_useful(self, action):

        def is_resource_eligible(process, method, resource):
            if method is None or resource is None:
                return None
            if method not in process.resource_eligibilities:
                return None
            if not (resource in process.resource_eligibilities[method].keys()):
                return None
            return process.resource_eligibilities[method][resource]
        # This method was added to determine how much the neuron network can be
        # optimized to achieve better results. Returned value is an integer with
        # possible values as follows:
        # 0 - there is at least one task in the queue and at least one not occupied resource that
        #     can process it but the action given as parameter does not change the environment state
        # 1 - chosen action properly assigns task from queue to the not occupied resource that can handle it
        # 2 - there is no pair of action and available resource in the queues that can be matched
        task_dict = {num: 0 for num in range(len(self.id_for_task.keys()))}
        [resource, task_id] = action
        #chosen_task = next(filter(lambda t: t.id == task_id, self.enabled_tasks), None)
        best_eligibility_env = math.inf
        action_eligibility = None

        chosen_process_case_id = None
        is_enabled = False
        if self._is_action_valid(action):
            is_resource_available = (resource in self.available_resources)
            is_enabled = self._task_enabled(resource, task_id)
            task = self.task_for_id[task_id]
            chosen_process_case = next(filter(lambda t: t.next_task == task, self.enabled_process_cases), None)
            chosen_process_case_id = chosen_process_case.id if chosen_process_case else None
            action_eligibility = is_resource_eligible(self.processes[0], task, resource)
            #action_eligibility = chosen_process_case.is_resource_eligible_new(task, resource) if chosen_process_case else 0
            action_eligibility = action_eligibility if action_eligibility else 0
        else:
            is_resource_available = None
            is_enabled = None


        for case in self.enabled_process_cases:
            id = self.id_for_task[case.next_task]
            task_dict[id] += 1
        # is_enabled = False
        # if self._is_action_valid(action):
        #     is_enabled = self._task_enabled(resource, task_id)

        any_assignment_available = False
        for case in self.enabled_process_cases:
            for res in self.available_resources:
                task = case.next_task
                if case.is_resource_eligible_new(task, res):
                    any_assignment_available = True
                    if case.is_resource_eligible_new(task, res) < best_eligibility_env:
                        best_eligibility_env = case.is_resource_eligible_new(task, res)

        return [self.step_counter, chosen_process_case_id, action, any_assignment_available, best_eligibility_env, action_eligibility, is_resource_available,
                is_enabled, self.available_resources, *task_dict.values(), self.state.values]

    def action_exists(self):
        tasks = [key for (key, value) in self.task_counter.items() if value > 0]
        if not tasks or not self.available_resources:
            return False
        for task in tasks:
            for res in self.available_resources:
                if res in self.resource_eligibilities[task].keys():
                    return True
        return False

    def continue_without_action(self):
        self._handle_completed_tasks()
        self._increase_waiting_time()
        if random.random() < self.process_case_probability and not self._is_queue_full:
            self.initialize_process_cases(1)
        self.increase_resources_utilization()
        self.step_counter += 1
        return self.state

    def get_processes_counters(self):
        return self.processes_counters

    def get_simulation_copy(self):
        return deepcopy(self)

    def _task_enabled(self, resource, task_id):
        if self.state_repr == "std":
            return self.state.loc[resource, task_id] == 0
        elif self.state_repr == "a0" or self.state_repr == "a1" or self.state_repr == "a2" or self.state_repr == "a10":
            return self.task_counter[self.task_for_id[task_id]] > 0
            #return self.state.iloc[len(self.all_resources) + task_id] > 0

    def _step_heuristic(self, heuristic_name):
        sort_attr = ""
        filename = heuristic_name + "_log.csv"
        if heuristic_name == "fifo":
            sort_attr = "process_start_step"
        elif heuristic_name == "spt":
            sort_attr = "task_start_step"
        elif heuristic_name == "lst_task":
            sort_attr = "task_slack_time"
        elif heuristic_name == "lst_process_case":
            sort_attr = "process_case_slack_time"
        elif heuristic_name == "edf_task":
            sort_attr = "task_deadline"
        elif heuristic_name == "edf_process_case":
            sort_attr = "process_case_deadline"

        global diff_counter
        reward = 0
        action = []

        if random.random() < self.process_case_probability and not self._is_queue_full:
            self.initialize_process_cases(1)

        is_looping = True
        if self.enabled_process_cases:
            for process_case in sorted(self.enabled_process_cases, key=lambda x: getattr(x, sort_attr)):
                random.shuffle(self.available_resources)
                for resource in self.available_resources:
                    factor = process_case.is_resource_eligible_new(process_case.next_task, resource)
                    if factor:
                        action = [resource, self.id_for_task[process_case.next_task]]
                        # eligibility_row = self.is_action_useful(action)
                        # with open(filename, 'a', newline='') as file:
                        #     writer = csv.writer(file)
                        #     writer.writerow(eligibility_row)
                        reward = self._assign_resource(process_case, resource, factor)
                        is_looping = False
                        break
                if not is_looping:
                    break

        self._handle_completed_tasks()
        self._increase_waiting_time()
        self.increase_resources_utilization()
        self.step_counter += 1
        if not action:
            action = [-1, -1]
        return self.state, reward, action

    @property
    def _is_queue_full(self):
        return self.queue_capacity <= len(self.enabled_process_cases)

    @property
    def _task_queue_waiting_time(self):
        time = 0
        for process_case in self.enabled_process_cases:
            time = time + process_case.task_waiting_time
        return time

    @property
    def _process_case_queue_waiting_time(self):
        time = 0
        for process_case in self.enabled_process_cases:
            time = time + process_case.process_waiting_time
        return time

    def _increase_waiting_time(self):
        for process_case in self.enabled_process_cases:
            process_case.task_waiting_time += 1
            process_case.process_waiting_time += 1
            process_case.time_length += 1
            if process_case.next_task == "confirmationofreceipt":
                process_case.process_response_time += 1
        for process_case in self.current_process_cases:
            process_case.process_processing_time += 1
            process_case.time_length += 1
        if len(self.current_process_cases) == 0:
            self.idle_waiting_time += 1

    def _state_std(self):
        pass

    @property
    def enabled_processes_length(self):
        x = []
        for p in self.enabled_process_cases:
            x.append(p.time_length)
        return x

    def increase_resources_utilization(self):
        used_resources = set(self.all_resources) - set(self.available_resources)
        for r in used_resources:
            if r not in self.resource_utilization:
                self.resource_utilization[r] = 1
            else:
                self.resource_utilization[r] += 1


def discover_process_model(log):
    filter = True
    import pm4py
    multiplier = 1
    k = 10
    min_coverage_percentage = 0.006
    methods, _ = simulation_activity.create_methods(log, multiplier)
    if filter:
        log = filter_variants_by_coverage_percentage(log, min_coverage_percentage)

    variants = pm4py.get_variants(log)

    variants_dict_2 = {'variants': [key.replace(" ", "").split(',') for key in variants.keys()], 'weights': [len(value) for value in variants.values()]}
    return methods, variants_dict_2


if __name__ == "__main__":
    pass
