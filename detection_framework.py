from gettext import find
from turtle import distance
import numpy as np
import matplotlib.pyplot as plt

from openTSNE import TSNE

import pandas as pd

import re
import os
import random

from kbc import process_datasets

from kbc.datasets import Dataset
from kbc.models import CP, ComplEx
from kbc.regularizers import F2, N3
from kbc.optimizers import KBCOptimizer
from kbc.process_datasets import prepare_dataset

from pathlib import Path
import pkg_resources
from regex import P

#from sklearn.manifold import TSNE

import shutil
from sympy import degree

import torch
from torch import optim

import tqdm

import os
from glob import glob

import functools

from scipy.spatial.distance import pdist, squareform

class DetectionFramework:
    def __init__(self) -> None:
        self.parser = None
        self.embedder = None

    @staticmethod
    def trojan_metric_distance_from_center(neighbors_embeddings):
        return np.mean(np.sqrt(np.sum(np.square(neighbors_embeddings))))


    @staticmethod
    def trojan_metric(neighbors_embeddings):
        """ Measures multi-dimensional "standard deviation" and distance from main cluster of neighbors to outlier
        args: 
        neighbors_embedding - np array of size axb, a - # of neighbors, b - # of embedding dimensions
        """

        #calculating the geometric center
        center = np.mean(neighbors_embeddings, axis=0)
        #print(center.shape)
        #looking for the furthest point

        # shame
        #dist = np.vectorize(lambda a, b: np.sqrt(np.sum(np.square(a-b))))
        #distances = dist(neighbors_embeddings, center)
        distances = []
        for i in range(neighbors_embeddings.shape[0]):
            neigh = neighbors_embeddings[i]
            distances.append(np.sqrt(np.sum(np.square(neigh-center))))
        distances = np.array(distances)

        #print(distances.shape)
        outlier_index = np.argmax(distances)
        outlier = neighbors_embeddings[outlier_index]
        cluster_center = np.mean(np.delete(neighbors_embeddings, np.s_[
                                outlier_index], axis=0), axis=0)
        distance_to_outlier = np.sqrt(np.sum(np.square(cluster_center-outlier)))

        distances_to_cluster_center = []
        for i in range(neighbors_embeddings.shape[0]):
            neigh = neighbors_embeddings[i]
            distances_to_cluster_center.append(
                np.sqrt(np.sum(np.square(neigh-cluster_center))))
        distances_to_cluster_center = np.array(distances_to_cluster_center)

        cluster_std = np.mean(np.delete(distances_to_cluster_center, np.s_[
                            outlier_index], axis=0), axis=0)
        #print(cluster_std)

        return cluster_std, distance_to_outlier, cluster_center

    @staticmethod
    def get_location_of_embeddings(embeddings_ids, embeddings):
        import numpy as np
        #print("embeddings_shape = ",embeddings.shape)
        #print(embeddings_ids)
        target_embeddings = embeddings[embeddings_ids, :]
        mean = np.mean(target_embeddings, axis=0)
        std = np.std(target_embeddings, axis=0)
        return mean, std

    @staticmethod
    def filter_embeddings_in_range(embeddings, mask, center, radii):
        """takes embeddings, return indices of embeddings with range radii from coordinates center
        args: embeddings - array of all embeddings, mask - only embeddings with indices in array 'mask' are considered,
        center - centor position, radii - radii 
        """
        #filtered_mask = np.where(center - embeddings[mask,:] < radii, mask)
        filtered_mask = list(
            filter(lambda x: (np.abs(center - embeddings[x]) < radii).all(), mask))
        print(len(filtered_mask))
        print(filtered_mask)
        return filtered_mask

    def map_idx_to_id(self, idx):
        for i in range(len(self.all_idx)):
            if idx == self.all_idx[i] or idx == tuple(self.all_idx[i]):
                return i
        return -1

    def map_idxs_to_ids(self, idxs):
        ids = []
        for idx in idxs:
            ids.append(self.map_idx_to_id(idx))
        return ids

    # @staticmethod
    # def map_idx_to_id(idx, all_idx):
    #     for i in range(len(all_idx)):
    #         if idx == all_idx[i]:
    #             return i
    #     return -1

    # @staticmethod
    # def map_idxs_to_ids(idxs, all_idx):
    #     ids = []
    #     for idx in idxs:
    #         ids.append(DetectionFramework.map_idx_to_id(idx, all_idx))
    #     return ids

    def build_maps(self):
        
        self.neighbors = {}

        for idx in self.all_idx:
            neighs = self.find_neighbors(idx)
            neighs = [(x[0], x[1], x[2]) for x in neighs]
            self.neighbors[self.map_idx_to_id(idx)] = self.map_idxs_to_ids(neighs)
        
        self.distance_map_reverse = {}
        for k, v in self.distance_map.items():
            self.distance_map_reverse.setdefault(v, []).append(k)
        
        
        # looking for trojan ids that are connected to safe ids:
        # node distance should be 0 and neighbors distances should be at least one 1
        def check_if_any(y):
            flag = False
            for x in y:
                if x in self.distance_map_reverse[1]:
                    flag = True
            return flag

        self.first_order = list(filter(lambda x: check_if_any(self.neighbors[x]) , self.distance_map_reverse[0]))

        # and looking for safe ids that are NOT connected to trojans

        self.super_safes = []
        for n in self.safe_ids:
            flag = False
            if self.distance_map[n] <= 2:
                continue # we don't need anything else
            for neigh in self.neighbors[n]:
                if self.distance_map[neigh] == 0:
                    flag = True
            if flag == False:
                self.super_safes.append(n)
        
        pass

    def sample_n_close_to_node(self, node_idx, max_n):
        max_distance = 1000

        distance_map_idx = {(node_idx[0], node_idx[1], node_idx[2]): 0}

        current = [(node_idx[0], node_idx[1], node_idx[2])]

        while len(current) > 0 and len(distance_map_idx) < max_n + 1:
            key = current.pop(0)
            node = (key[0], key[1], key[2])
            distance = distance_map_idx[key]

            if distance >= max_distance:
                continue
            
            neighs = self.find_neighbors(node)
            neighs = [(x[0], x[1], x[2]) for x in neighs]
            for n in neighs:
                if n in distance_map_idx:
                    if distance + 1 < distance_map_idx[n]:
                        distance_map_idx[n] = distance + 1
                else:
                    distance_map_idx[n] = distance + 1
                    current.append(n)
        
        return distance_map_idx

    def shortest_distance_to_trojan(self):
        max_distance = 1000

        distance_map_idx = {(signal[0], signal[1], signal[2]):0 for signal in self.trojan_signals_idx}

        current = list(filter(lambda x: distance_map_idx[x] == 0, distance_map_idx.keys()))
        

        while len(current) > 0:
            key = current.pop(0)
            node = (key[0], key[1], key[2])
            distance = distance_map_idx[key]
            
            if distance >= max_distance:
                continue

            neighs = self.find_neighbors(node)
            neighs = [(x[0], x[1], x[2]) for x in neighs]
            for n in neighs:
                if n in distance_map_idx:
                    if distance + 1 < distance_map_idx[n]:
                        distance_map_idx[n] = distance + 1
                else:
                    distance_map_idx[n] = distance + 1
                    current.append(n)
        
        self.distance_map = {}
        self.trojan_ids = []
        self.safe_ids = []
        for key, value in distance_map_idx.items():
            self.distance_map[self.map_idx_to_id(key)] = value
            if value == 0:
                self.trojan_ids.append(self.map_idx_to_id(key))
            else:
                self.safe_ids.append(self.map_idx_to_id(key))

        pass

    def find_neighbors(self, signal):
        """finds neighbors of 'signal' node in 'all_idx' graph, assiming that wire relation is 'wire_id' 
        """
        frontier = [signal]
        neighbors = []
        while len(frontier) > 0:
            current = frontier.pop(0)

            for (e1, r, e2) in self.all_idx:
                if e1 == current[2]:
                    if r == self.wire_id:
                        frontier.append((e1, r, e2))
                    else:
                        neighbors.append([e1, r, e2])
                if e2 == current[0]:
                    if r == self.wire_id:
                        frontier.append((e1, r, e2))
                    else:
                        neighbors.append([e1, r, e2])
        return neighbors


    def extract_input_output_pairs_trojan(self, filename):
        """
        idx - graph elements, 3-tuples like (100,10,100). Each of them stand for one embedding
        """
        original_v = ''
        with open(filename, 'rt') as f:
            original_v = f.read()

        signal_lists = []

        trojan_signals = []
        safe_signals = []
        triggering_events_signals = []

        first_order_triggering_events = []

        trojan_gate_name = r'.*troj.*'

        output_strings = []

        entities_set = set()
        relations_set = set()

        safe_ids = []
        trojan_ids = []

        matches = re.findall(
            r'\n\s*([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s*\((.*)\).*;', original_v)
        for m in matches:
            #print(m[1])
            gate_type = m[0]
            gate_name = m[1]
            signal_list = re.split(r'[\s\(\),]+', m[2])
            signal_list = list(filter(lambda x: len(x) > 0, signal_list))
            input_list = signal_list[::2]
            signal_list = signal_list[1::2]

            relations = list([gate_type + x for x in input_list])

            #print(signal_list)
            gate_output = signal_list[0]
            signal_list = signal_list[1:]

            tab = '\t'

            for signal, relation in zip(signal_list, relations):
                output_string = f"{gate_name}{tab}{relation}{tab}{signal}\n"
                output_strings.append(output_string)

                entities_set.add(gate_name)
                entities_set.add(signal)
                relations_set.add(relation)

                if re.match(trojan_gate_name, gate_name):
                    trojan_signals.append((gate_name, relation, signal))
                else:
                    safe_signals.append((gate_name, relation, signal))

            if bool(gate_output.find('troj') != -1) != bool(gate_name.find('troj') != -1):
                first_order_triggering_events.append(
                    (gate_name, relation, signal))

            output_strings.append(f"{gate_output}{tab}wire{tab}{gate_name}\n")
            if re.match(trojan_gate_name, gate_name):
                trojan_signals.append((gate_output, 'wire', gate_name))
            else:
                safe_signals.append((gate_output, 'wire', gate_name))
            entities_set.add(gate_name)
            entities_set.add(gate_output)
            relations_set.add('wire')

        entities_to_id = {x: i for (i, x) in enumerate(sorted(entities_set))}
        relations_to_id = {x: i for (i, x) in enumerate(sorted(relations_set))}

        wire_id = relations_to_id['wire']
        self.wire_id = wire_id

        trojan_signals_idx = [[entities_to_id[e1], relations_to_id[r],
                               entities_to_id[e2]] for (e1, r, e2) in trojan_signals]
        safe_signals_idx = [[entities_to_id[e1], relations_to_id[r],
                            entities_to_id[e2]] for (e1, r, e2) in safe_signals]
        first_order_idx = [[entities_to_id[e1], relations_to_id[r], entities_to_id[e2]] for (
            e1, r, e2) in first_order_triggering_events]

        join = [(trojan[2], safe[0], safe[1], safe[2])
                for safe in safe_signals_idx for trojan in trojan_signals_idx if safe[2] == trojan[0]]


        all_idx = safe_signals_idx + trojan_signals_idx

        print('safe idx size ', len(safe_signals_idx), 'trojan idx size', len(trojan_signals_idx) )

        self.all_idx = all_idx

        neighbors = []
        for i in first_order_idx:
            neighbors.append(self.find_neighbors(i,))

        neighbors_ids = []
        for n in neighbors:
            neighbors_ids.append(
                [i for i in range(len(all_idx)) if all_idx[i] in n])

        print('neighbors:', neighbors)
        print('neighbors ids:', neighbors_ids)

        first_order_ids = [i for i in range(
            len(all_idx)) if all_idx[i] in first_order_idx]

        with open('./all_idx.txt', 'w') as f:
            for item in all_idx:
                f.write("%s\n" % item)

        #print(len(safe_signals_idx), len(trojan_signals_idx))

        
        self.safe_signals_idx = safe_signals_idx
        self.trojan_signals_idx = trojan_signals_idx
        self.first_order_ids = first_order_ids
        

        return output_strings, all_idx, safe_signals_idx, trojan_signals_idx, first_order_ids, neighbors_ids, wire_id

    def train_netlist_embeddings(self, filename, num_epochs=30):
        """ trains embeddings from the given file
        """
        strings, all_idx, safe_idx, trojan_idx, first_order_ids, neighbors_ids, wire_id = self.extract_input_output_pairs_trojan(
            filename)

        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')

        with open('./tmp/train', 'wt') as f:
            f.writelines(strings)
        with open('./tmp/valid', 'wt') as f:
            f.writelines(strings)
        with open('./tmp/test', 'wt') as f:
            f.writelines(strings)

        DATA_PATH = pkg_resources.resource_filename('kbc', 'data/')
        if os.path.exists(Path(DATA_PATH) / filename):
            shutil.rmtree(Path(DATA_PATH) / filename)

        prepare_dataset('./tmp/', filename)

        dataset = Dataset(filename)

        model = ComplEx(dataset.get_shape(), 200)

        device = 'cuda'
        model.to(device)

        regularizer = N3(0.00001)
        examples = torch.from_numpy(dataset.get_train().astype('int64'))

        optimizer = KBCOptimizer(model, regularizer, optim.Adagrad(
            model.parameters(), lr=0.1), batch_size=256, verbose=False)

        cur_loss = 0
        curve = {'train': [], 'valid': [], 'test': []}

        for e in tqdm.tqdm(range(num_epochs)):
            cur_loss = optimizer.epoch(examples)

        print(len(self.all_idx))
        embeddings = model.get_queries(torch.tensor(all_idx).to("cuda"))
        embeddings = embeddings.cpu().detach().numpy()
        self.embeddings = embeddings
        return embeddings, first_order_ids, neighbors_ids, wire_id, all_idx, safe_idx

    def tSNE(self, n_iter = 1000, perplexity = 100, n_components = 2):
        tsne = TSNE(n_components=n_components, learning_rate='auto',
                    perplexity=perplexity, n_iter = n_iter, n_jobs = 2)
        self.lowdim = tsne.fit(self.embeddings)


    def visualize_embeddings(self, ids, color_code = True):
        plt.scatter(self.lowdim.transpose()[0], self.lowdim.transpose()[1], alpha=0.01)

        colors_code = {0: 'red', 1: 'orange',
                       2: 'yellow', 3: 'green', 4: 'blue', 5: 'blue', 6:'blue',
                        7:'blue', 8:'blue', 9:'blue', 10:'blue', 11:'blue'}

        colors = []
        for i in ids:
            colors.append(colors_code[self.distance_map[i]])

    
        plt.scatter(self.lowdim[ids,:].transpose()[0], self.lowdim[ids,:].transpose()[1], c = colors)
        plt.show(block=True)


    def get_embeddings_detected_with_simple_policies(self, close = False):
        # this is needed to make the training set as balanced as possible
        # we take all trojan nodes and only  

        trojan_ids = self.first_order_ids

        trojan_metrics = []

        for n in self.first_order_ids:
            neighs = self.neighbors[n]
            if len(neighs) > 1:
                #std, dist = trojan_metric(n, embeddings[neighs,:] + [n])
                std, dist, _ = self.trojan_metric(
                    self.embeddings[neighs + [n], :])
                # position = self.trojan_metric_distance_from_center(self.lowdim[neighs + [n],:])
                metric = (dist+0.25)/(std+0.25)  # + 0.1/position

                _, _, centroid = self.trojan_metric(
                    self.lowdim[neighs + [n], :])
                if close:
                    if np.mean(centroid) > 0.25:
                        continue
                trojan_metrics.append(metric)
        
        median_trojan = np.median(trojan_metrics)

        #sampling safe nodes
        outliers = []
        outliers_metrics = []
        safe_metrics = []
        for safe_id in self.super_safes:
            #sampling
            sampling_rate = 1.0
            if random.uniform(0, 1) > sampling_rate: continue 
            if len(neighs) > 1:
                safe_neighbors = self.neighbors[safe_id]
                # std, dist = trojan_metric(safe_id, embeddings[safe_neighbors,:] + [n])
                # std, dist = self.trojan_metric(self.embeddings[safe_neighbors + [safe_id],:])
                
                std, dist, _ = self.trojan_metric(
                    self.embeddings[safe_neighbors + [safe_id], :])
                _,_, centroid = self.trojan_metric(
                    self.lowdim[safe_neighbors + [safe_id], :])
                if close:
                    if np.mean(centroid) > 0.25:
                        continue
                metric = (dist+0.25)/(std+0.25) # + 0.1/position

                safe_metrics.append(metric)
                if metric > median_trojan:
                    outliers.append(safe_id)
        
        safes_ids = outliers

        return trojan_ids, safes_ids
        
    def get_metrics(self):
        trojan_metrics = []

        for n in self.first_order_ids:
            neighs = self.neighbors[n]
            if len(neighs) > 1:
                metric = metric2(self.lowdim[neighs + [n], :])
                _, _, centroid = self.trojan_metric(
                    self.lowdim[neighs + [n], :])
                if np.mean(centroid) > 0.25:
                    continue

                trojan_metrics.append(metric)

        safe_metrics = []
        for safe_id in self.super_safes:
            #sampling
            sampling_rate = 0.1
            if random.uniform(0, 1) > sampling_rate:
                continue
            if len(neighs) > 1:
                safe_neighbors = self.neighbors[safe_id]
                # std, dist = trojan_metric(safe_id, embeddings[safe_neighbors,:] + [n])
                # std, dist = self.trojan_metric(self.embeddings[safe_neighbors + [safe_id],:])

                _, _, centroid = self.trojan_metric(
                    self.lowdim[safe_neighbors + [safe_id], :])
                if np.mean(centroid) > 0.25:
                    continue
                metric = metric2(
                    self.lowdim[safe_neighbors + [safe_id], :])

                safe_metrics.append(metric)

        return safe_metrics, trojan_metrics

    def get_average_metrics(self):
        trojan_metrics = []

        for n in self.first_order_ids:
            neighs = self.neighbors[n]
            if len(neighs) > 1:
                metric = metric2(self.lowdim[neighs + [n], :])
                _, _, centroid = self.trojan_metric(
                    self.lowdim[neighs + [n], :])
                if np.mean(centroid) > 0.25:
                    continue

                trojan_metrics.append(metric)
        mean_trojan = np.median(trojan_metrics)

        safe_metrics = []
        for safe_id in self.super_safes:
            #sampling
            sampling_rate = 1.0
            if random.uniform(0, 1) > sampling_rate:
                continue
            if len(neighs) > 1:
                safe_neighbors = self.neighbors[safe_id]
                # std, dist = trojan_metric(safe_id, embeddings[safe_neighbors,:] + [n])
                # std, dist = self.trojan_metric(self.embeddings[safe_neighbors + [safe_id],:])

                _, _, centroid = self.trojan_metric(
                    self.lowdim[safe_neighbors + [safe_id], :])
                if np.mean(centroid) > 0.25:
                    continue
                metric = metric2(
                    self.lowdim[safe_neighbors + [safe_id], :])

                safe_metrics.append(metric)
        mean_safe = np.median(safe_metrics)

        # separable = np.min(trojan_metrics) / np.max(safe_metrics)

        return mean_safe, mean_trojan

    def analyze_safes_and_trojans(self):
        trojan_metrics = []

        for n in self.first_order_ids:
            neighs = self.neighbors[n]
            if len(neighs) > 1:
                metric = metric2(self.lowdim[neighs + [n], :])
                _,_, centroid = self.trojan_metric(
                    self.lowdim[neighs + [n], :])
                if np.mean(centroid) > 0.25:
                    continue

                trojan_metrics.append(metric)

        print('Trojan metrics: ', trojan_metrics)                

        median_trojan = np.median(trojan_metrics)

        #sampling safe nodes
        outliers = []
        outliers_metrics = []
        safe_metrics = []
        for safe_id in self.super_safes:
            #sampling
            sampling_rate = 1.0
            if random.uniform(0, 1) > sampling_rate: continue 
            if len(neighs) > 1:
                safe_neighbors = self.neighbors[safe_id]
                # std, dist = trojan_metric(safe_id, embeddings[safe_neighbors,:] + [n])
                # std, dist = self.trojan_metric(self.embeddings[safe_neighbors + [safe_id],:])
                
                _,_, centroid = self.trojan_metric(
                    self.lowdim[safe_neighbors + [safe_id], :])
                if np.mean(centroid) > 0.25:
                    continue
                metric = metric2(
                    self.lowdim[safe_neighbors + [safe_id], :])

                safe_metrics.append(metric)
                if metric > median_trojan:
                    outliers.append(safe_id)
                    outliers_metrics.append(metric)

        print('False positives metrics: ', outliers_metrics)

        #histogram

        # bins = np.linspace(0, 4, 40)
        # plt.hist(trojan_metrics, bins, alpha=1, label='trojan dist/std')
        # plt.hist(safe_metrics, bins, alpha=0.5, label='safe dist/std')
        # #plt.hist(safe_metrics)
        # #plt.hist(y, bins, alpha=0.5, label='y')
        # plt.legend(loc='upper right')
        # plt.show(block=True)
        
        for i in self.first_order_ids + self.trojan_ids:
            self.visualize_embeddings(self.neighbors[i] + [i])
        

        pass



    def generate_trainset(self, add_raw_features = False, positions = False):
        trojan_ids, safe_ids = self.get_embeddings_detected_with_simple_policies(close = False)
        X = []
        Y = []
        n_features = 20
        if positions is True:
            n_features = 30
        for t in trojan_ids:
            if positions:
                field, idx = self.build_perceptive_field_positions_tsne(t, 10)
            else:    
                field, idx = self.build_perceptive_field(t, 10, 0.1, add_raw_features)
            if len(field.shape) == 0 or field.shape[0] != n_features:
                continue
            X.append(field)
            Y.append(1)
        for s in safe_ids:
            if positions:
                field, idx = self.build_perceptive_field_positions_tsne(s, 10)
            else:
                field, idx = self.build_perceptive_field(s, 10, 0.1, add_raw_features)
            if len(field.shape) == 0 or field.shape[0] != n_features:
                continue
            X.append(field)
            Y.append(0)
    
        #X = np.array(X, dtype=object)
        #Y = np.array(Y, dtype=object)
        #print(f'X shape: {X.shape}, Y shape: {Y.shape}')
        
        # print('X:')
        # print(X)
        # print('Y:')
        # print(Y)



        X = np.stack(X, axis = 0)
        Y = np.array(Y)
        return X, Y
        pass

    def check_receptive_field(self):
        trojans = self.first_order_ids
        for t in trojans:
            field, idx = self.build_perceptive_field(t, 10, 0.1)
            print(f'Node {t}, Receptive field: {field}, neigh idxs: {idx}')

    def build_perceptive_field_positions_tsne(self, id, field_size= 10):
        neighs_map = self.sample_n_close_to_node(self.all_idx[id], field_size)
        del neighs_map[(self.all_idx[id][0], self.all_idx[id]
                        [1], self.all_idx[id][2])]

        if len(neighs_map) == 0:
            return np.array([0.0, 0.0])

        neigh_idxs, jumps = list(zip(*neighs_map.items()))
        neigh_ids = self.map_idxs_to_ids(neigh_idxs)

        def distance(x, y): return np.linalg.norm(x - y)

        #sort neighs by their distance from
        #1) original node
        #2) in low-dim space
        original_node_vector = self.lowdim[id]
        distances = [distance(original_node_vector, self.lowdim[x]) for x in neigh_ids]
        
        positions = [self.lowdim[id]] + [self.lowdim[x] for x in neigh_ids]


        distances_sorted, positions_sorted, neigh_ids_sorted, jumps_sorted = list(
            zip(* sorted(zip(distances, positions, neigh_ids, jumps))))


       
        
        return np.concatenate((np.array(positions_sorted[:field_size]).flatten(), jumps_sorted[:field_size])), neigh_ids_sorted[:field_size]


        pass

    def build_perceptive_field(self, id, field_size = 10, randomize = 0.0, raw_features = False):
        # sample neighbors of the original graph
        # omitting wires

        #instead, sampling neighbors in Djikstra manner until we have 

        neighs_map = self.sample_n_close_to_node(self.all_idx[id], field_size)
        del neighs_map[(self.all_idx[id][0], self.all_idx[id][1], self.all_idx[id][2])]

        if len(neighs_map) == 0:
            return np.array([0.0, 0.0])

        neigh_idxs, jumps = list(zip(*neighs_map.items()))
        neigh_ids = self.map_idxs_to_ids(neigh_idxs)

        distance = lambda x, y: np.linalg.norm(x - y)
    

        #sort neighs by their distance from
        #1) original node
        #2) in low-dim space
        original_node_vector = self.lowdim[id]
        distances = [distance(original_node_vector, self.lowdim[x]) * random.uniform(1.0 - randomize, 1.0 + randomize)  for x in neigh_ids]
        
        distances_raw = []

        if raw_features:
            distances_raw = [distance(self.embeddings[id], self.embeddings[x]) * random.uniform(1.0 - randomize, 1.0 + randomize)  for x in neigh_ids]

        if raw_features is False:
            distances_sorted, neigh_ids_sorted, jumps_sorted = list(zip(* sorted(zip(distances, neigh_ids, jumps))))
            return np.array(distances_sorted[:field_size] + jumps_sorted [:field_size]), neigh_ids_sorted[:field_size]
        else:
            distances_sorted, distances_raw_sorted, neigh_ids_sorted, jumps_sorted = list(
                zip(* sorted(zip(distances, distances_raw, neigh_ids, jumps))))
            return np.array(distances_sorted[:field_size] + distances_raw_sorted[:field_size] + jumps_sorted [:field_size]), neigh_ids_sorted[:field_size]


    def distances_from_all_embeddings_to_center(self):
        """
        Returns relative (to the cloud diameter) distances from every embedding to
        the closest large cluster center
        """
        from sklearn.cluster import OPTICS
        from sklearn.neighbors import NearestCentroid

        # if non-tsne needed - change self.lowdim to self.embeddings
        D1 = pdist(self.lowdim)
        D1 = squareform(D1)
        diagonal = np.nanmax(D1)

        trojan_centroid = np.mean(self.lowdim[self.trojan_ids], axis=0)

        

        clustering = OPTICS(min_samples=20).fit(self.lowdim)
        labels = clustering.labels_

        #removing noise
        # embeddings_no_noise = self.lowdim[labels != -1]
        # labels_no_noise = labels[labels != -1]
        labels_no_noise = np.concatenate((labels, [1000]))
        embeddings_no_noise = np.concatenate((self.lowdim, [(1000.0, 1000.0)]))

        clf = NearestCentroid()
        clf.fit(embeddings_no_noise, labels_no_noise)
        centroids = clf.centroids_

        distances_result = []

        for e in self.lowdim:
            distances = np.linalg.norm(centroids-e, axis = 1)
            min_distance = np.min(distances)
            distances_result.append(min_distance / diagonal)


        return distances_result

    def distance_to_center(self):
        """
        Returns relative (to the cloud diameter) distance to the closest large cluster center
        """

        from sklearn.cluster import OPTICS
        from sklearn.neighbors import NearestCentroid

        # if non-tsne needed - change self.lowdim to self.embeddings
        D1 = pdist(self.lowdim)
        D1 = squareform(D1)
        diagonal = np.nanmax(D1)

        trojan_centroid = np.mean(self.lowdim[self.trojan_ids], axis=0)

        clustering = OPTICS(min_samples=20).fit(self.lowdim)
        labels = clustering.labels_

        #removing noise
        # embeddings_no_noise = self.lowdim[labels != -1]
        # labels_no_noise = labels[labels != -1]
        labels_no_noise = labels
        embeddings_no_noise = self.lowdim

        clf = NearestCentroid()
        clf.fit(embeddings_no_noise, labels_no_noise)
        centroids = clf.centroids_

        distances = np.linalg.norm(centroids-trojan_centroid, axis = 1)
        min_distance = np.min(distances)

        cluster_number = np.argmin(distances)

        return min_distance / diagonal, cluster_number



    pass


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def expand_training_data(points, n_rotations = 16, scale_range_low = 0.9, scale_range_high = 1.1, scale_step = 0.1):
    # datapoints are rotated around (0,0) n_rotation times
    extended_set = []
    for i in range(n_rotations):
        points_rotated = rotate(points, degree = i * 360/n_rotations)
        for scale in range(scale_range_low, scale_range_high, scale_step):
            points_rotated_scaled = points_rotated * scale
            extended_set.append(points_rotated_scaled)
    
    return extended_set


def generate_trainset_from_benchmarks(folder = './TRIT-TC', out_filename = 'trainset', timer = 100000, feature_position = False, feature_distance = True, n_rotations = 1):
    import time

    #timer_intermediate = 30


    timer_intermediate = 60*60

    start_time = time.time()
    previous_timer = start_time

    benchmarks = [y for x in os.walk(folder)
              for y in glob(os.path.join(x[0], '*.v'))]
    
    random.shuffle(benchmarks)

    #print(benchmarks)

    # for debug, sample first ten
    #benchmarks = benchmarks[:3]

    Xs = []
    Ys = []

    benchmark_classes = []

    benchmark_names = []

    benchmark_ids = []

    pbar = tqdm.tqdm(benchmarks)
    counter_benchmark_id = 0
    for b in pbar:
        pbar.set_description('Processing benchmarks')
        if b.find('s') != -1:
            bench_class = 1 # state
        else:
            bench_class = 0 # combinational
        
        bench_name = b

        framework = DetectionFramework()
        framework.train_netlist_embeddings(b, 100)
        framework.shortest_distance_to_trojan()
        framework.build_maps()
        framework.tSNE(n_iter = 500, perplexity = 35) ####
        #framework.analyze_safes_and_trojans()
        
        for i in range(n_rotations):
            #framework.lowdim = rotate(framework.lowdim, degrees = i * 360/n_rotations)
            X = []
            Y = []
            #framework.check_receptive_field()
            if feature_distance:
                X, Y = framework.generate_trainset()
            if feature_position:
                X, Y = framework.generate_trainset(positions=True)
            Xs.append(X)
            Ys.append(Y)
            benchmark_classes.append(np.ones(Y.shape) * bench_class)
            benchmark_ids.append(np.ones(Y.shape) * counter_benchmark_id)
            
            counter_benchmark_id += 1

            for j in range(Y.shape[0]):
                benchmark_names.append(bench_name)

            if (time.time() - start_time) > timer:
                break


            if (time.time() - previous_timer) > timer_intermediate:
                
                X_full = np.concatenate(Xs, axis=0)
                #functools.reduce(lambda acc, a: np.stack(acc, a), Xs)
                Y_full = np.concatenate(Ys, axis=0)
                #functools.reduce(lambda acc, a: np.stack(acc, a), Ys)
                ids_full = np.concatenate(benchmark_ids, axis=0)
                bench_classes_full = np.concatenate(benchmark_classes, axis=0)

                np.savetxt(f'{out_filename}_intermediate_X_bench_names.csv',
                           benchmark_names, delimiter=',', fmt='%s')

                np.savetxt(f'{out_filename}_intermediate_X.csv', X_full, delimiter=',')
                np.savetxt(f'{out_filename}_intermediate_Y.csv', Y_full, delimiter=',')
                np.savetxt(f'{out_filename}_intermediate_ids.csv', ids_full, delimiter=',')
                np.savetxt(f'{out_filename}_intermediate_bench_classes.csv',
                           bench_classes_full, delimiter=',')
                previous_timer = time.time()
    
    X_full = np.concatenate(Xs, axis=0)
    #functools.reduce(lambda acc, a: np.stack(acc, a), Xs)
    Y_full = np.concatenate(Ys, axis=0)
    #functools.reduce(lambda acc, a: np.stack(acc, a), Ys)
    ids_full = np.concatenate(benchmark_ids, axis = 0)

    np.savetxt(f'{out_filename}_X.csv', X_full, delimiter = ',')
    np.savetxt(f'{out_filename}_Y.csv', Y_full, delimiter = ',')
    np.savetxt(f'{out_filename}_ids.csv', ids_full, delimiter=',')
    np.savetxt(f'{out_filename}_intermediate_bench_classes.csv',
               bench_classes_full, delimiter=',')
    pass


def test():
    framework = DetectionFramework()
    #framework.train_netlist_embeddings('benches/c2670_T001.v', 100)
    #framework.train_netlist_embeddings('TRIT-TC/c5315_T039/c5315_T039.v', 100)
    framework.train_netlist_embeddings('TRIT-TC/c2670_T004/c2670_T004.v', 300)
    framework.shortest_distance_to_trojan()
    framework.build_maps()
    framework.tSNE(perplexity = 300)
    framework.analyze_safes_and_trojans()

    #framework.check_receptive_field()
    #framework.generate_trainset()

def validate_benchmarks(filename, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #once we heve two classes, threshold is accounted
    #as predictes[1] > threshold + predicted[0]
    # thresold is supposed to be [-0.5 - 0.5]

    valid_dict = {
        'bench_id': [],
        'bench_name': [],
        'bench_type' : [],
        'bench_prediction_with_ht': [],
        'bench_prediction_without_ht': [],
                }

    threshold = 0.0
    

    min_HT_number = 1

    x_data = np.genfromtxt(f'{filename}_X.csv', delimiter = ',')
    y_data = np.genfromtxt(f'{filename}_Y.csv', delimiter = ',')
    ids = np.genfromtxt(f'{filename}_ids.csv', delimiter=',')
    bench_classes = np.genfromtxt(
        f'{filename}_bench_classes.csv', delimiter=',')
    
    names_file = open(f'{filename}_X_bench_names.csv')
    bench_names = names_file.readlines()
    #min_id = np.min(ids)
    #max_id = np.max(ids)

    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0

    ids_s = ids[np.where(bench_classes == 1)]

    ids_c = ids[np.where(bench_classes == 0)]

    ########
    # combinatorial

    c_benches = np.unique(ids_c)
    for i in c_benches:
        #for i in range(int(min_id), int(max_id) + 1):
        indexes = np.where(ids == i)

        #sampling features of current benchmark
        x_data_benchmark = x_data[indexes]
        y_data_benchmark = y_data[indexes]

        y_pred = model(torch.FloatTensor(x_data_benchmark).to(device))
        y_pred = y_pred.detach().cpu().numpy()

        total = 0
        for j in range(y_pred.shape[0]):
            if y_pred[j][1] > threshold + y_pred[j][0]:
                total += 1
        
        valid_dict['bench_id'].append(i)
        valid_dict['bench_type'].append(0)
        #sketchy
        valid_dict['bench_name'].append(bench_names[indexes[0][0]])

        if total >= min_HT_number:
            valid_dict['bench_prediction_with_ht'].append(1)
            true_positives += 1
        else:
            valid_dict['bench_prediction_with_ht'].append(0)
            false_negatives += 1

        #sampling safe version of this benchmark

        safe_indexes = np.where(y_data_benchmark == 0)
        x_data_safe = x_data_benchmark[safe_indexes]

        y_pred = model(torch.FloatTensor(x_data_safe).to(device))
        y_pred = y_pred.detach().cpu().numpy()

        total = 0
        for j in range(y_pred.shape[0]):
            if y_pred[j][1] > threshold + y_pred[j][0]:
                total += 1
        if total >= min_HT_number:
            false_positives += 1
            valid_dict['bench_prediction_without_ht'].append(1)
        else:
            true_negatives += 1
            valid_dict['bench_prediction_without_ht'].append(0)

    if (true_positives + false_negatives) == 0:
        c_r  = 0
    else:
        c_r = true_positives / (true_positives + false_negatives)
    if (true_positives + false_positives) == 0:
        c_p = 0
    else:
        c_p = true_positives / (true_positives + false_positives)

    print(f'combinatorial recall is {c_r}')
    print(f'combinatorial prec is {c_p}')



    ##################
    #sequentional

    true_positives = 0
    false_positives = 0

    for i in np.unique(ids_s):
        #for i in range(int(min_id), int(max_id) + 1):
        indexes = np.where(ids == i)

        #sampling features of current benchmark

        x_data_benchmark = x_data[indexes]
        y_data_benchmark = y_data[indexes]

        y_pred = model(torch.FloatTensor(x_data_benchmark).to(device))
        y_pred = y_pred.detach().cpu().numpy()

        total = 0
        for j in range(y_pred.shape[0]):
            if y_pred[j][1] > threshold + y_pred[j][0]:
                total += 1

        valid_dict['bench_id'].append(i)
        valid_dict['bench_type'].append(1)
        #sketchy
        valid_dict['bench_name'].append(bench_names[indexes[0][0]])
        
        if total >= min_HT_number:
            valid_dict['bench_prediction_with_ht'].append(1)
            true_positives += 1
        else:
            valid_dict['bench_prediction_with_ht'].append(0)
            false_negatives += 1

        #sampling safe version of this benchmark

        safe_indexes = np.where(y_data_benchmark == 0)
        x_data_safe = x_data_benchmark[safe_indexes]

        y_pred = model(torch.FloatTensor(x_data_safe).to(device))
        y_pred = y_pred.detach().cpu().numpy()

        total = 0
        for j in range(y_pred.shape[0]):
            if y_pred[j][1] > threshold + y_pred[j][0]:
                total += 1
        if total >= min_HT_number:
            false_positives += 1
            valid_dict['bench_prediction_without_ht'].append(1)
        else:
            true_negatives += 1
            valid_dict['bench_prediction_without_ht'].append(0)

    if (true_positives + false_negatives) == 0:
        s_r = 0
    else:
        s_r = true_positives / (true_positives + false_negatives)
    if (true_positives + false_positives) == 0:
        s_p = 0
    else:
        s_p = true_positives / (true_positives + false_positives)

    print(f'sequentional recall is {s_r}')
    print(f'sequentional prec is {s_p}')

    frame = pd.DataFrame(data = valid_dict)

    return c_r, c_p, s_r, s_p, frame
    #return true_positives, true_negatives, false_positives, false_negatives

def test_mlp(filename_base = 'bench3', n_epochs = 4000000, intermediate_log = True):

  

    x_data = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                      [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                      [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                      [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                      [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]])
    
    y_data = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[1.0],[1.0],[1.0],[1.0],[1.0]])
    


    # x_data = np.genfromtxt('trainset_X.csv', delimiter = ',')
    # y_data = np.genfromtxt('trainset_Y.csv', delimiter = ',')
    
    # x_data = np.genfromtxt('trainset_X_intermediate.csv', delimiter = ',')
    # y_data = np.genfromtxt('trainset_Y_intermediate.csv', delimiter = ',')
    x_data = np.genfromtxt(f'{filename_base}_X.csv', delimiter = ',')
    y_data = np.genfromtxt(f'{filename_base}_Y.csv', delimiter=',')

    

    #converting labels from [N]-shape tensor into [N,1]-shape tensor
    y_data = np.expand_dims(y_data, axis = -1)

    #expand from (n, 1) labels to (n, 2), where each output stands for each class
    # [1, 0] is safe class, [0, 1] is HT
    y_data_safe_label = 1-y_data
    y_data = np.concatenate((1-y_data, y_data), axis = 1)

    #y_data = np.array([0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0])
    
    if intermediate_log:
        eval_file = filename_base
    else:
        eval_file = None

    model = pytorch_mlp(x_data, y_data, n_epochs=n_epochs, filename_base=eval_file)
    _, _, _, _, frame = validate_benchmarks(filename_base, model)

    return frame
    pass

def BCELoss_ClassWeights(input, target, class_weights):
    # input (n, d)
    # target (n, d)
    # class_weights (1, d)
    input = torch.clamp(input, min=1e-7, max=1-1e-7)
    bce = - target * torch.log(input) - (1 - target) * torch.log(1 - input)
    weighted_bce = (bce * class_weights).sum(axis=1) / \
        class_weights.sum(axis=1)[0]
    final_reduced_over_batch = weighted_bce.mean(axis=0)
    return final_reduced_over_batch

def metric2(vectors):
    """
    calculates a metric that is high for HTs and low for safe.
    HT have high density (# of points) / diameter^dimension of center
    and low density of outlier.

    Metric is density(vectors_sorted_by_distance_to_center[:-1] ) /
                    density( vectors_sorted_by_distance_to_center[-2:])

    Eyeballing shows that HTs are tend to be blob-shaped, therefore
    this density will be overrated on HT and underrated on safes, that
    leads to even better results
    """

    vectors = np.unique(vectors, axis=0)

    centroid = vectors.mean(axis=0)
    vectors_sorted_by_distance_to_center = sorted(
        vectors, key=lambda v: np.linalg.norm(v - centroid))
    
    D1 = pdist(vectors_sorted_by_distance_to_center[:-1])
    D1 = squareform(D1)
    diagonal1 = np.nanmax(D1)
    density1 = (len(vectors_sorted_by_distance_to_center) - 1) / diagonal1**2

    D2 = pdist(vectors_sorted_by_distance_to_center[-2:])
    D2 = squareform(D2)
    diagonal2 = np.nanmax(D2)
    density2 = 2 / diagonal2**2

    return density1 / density2


def pytorch_mlp(x_data, y_data, n_epochs = 4004, filename_base = None):

    intermediate_period = 5000
    intermediate_s_r = []
    intermediate_s_p = []
    intermediate_c_r = []
    intermediate_c_p = []
    intermediate_n_epochs = []
    intermediate_loss = []

    from sklearn.model_selection import train_test_split
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class_weights = [0.3, 3.0]

    x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2)
    class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.relu2 = torch.nn.ReLU()
            self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size * 2)
            self.relu3 = torch.nn.ReLU()
            #each output for each class
            self.fc4 = torch.nn.Linear(self.hidden_size * 2, 2)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.relu2(output)
            output = self.fc3(relu)
            output = self.relu3(output)
            output = self.fc4(output)
            output = self.sigmoid(output)
            return output

    model = Feedforward(20, 40).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #x_train_pytorch = torch.FloatTensor(x_train)
    #y_train_pytorch = torch.FloatTensor(y_train) 
    x_train_pytorch = torch.FloatTensor(x_train).to(device)
    y_train_pytorch = torch.FloatTensor(y_train).to(device) 

    losses = []

    model.train()

    print(f'Training {n_epochs} epochs...')

    for epoch in tqdm.tqdm(range(n_epochs)):

        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train_pytorch)
        # Compute Loss
        loss = criterion(y_pred, y_train_pytorch)
        #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        losses.append(loss.detach().cpu().numpy())
        # Backward pass
        loss.backward()

        if epoch > 0 and (epoch % intermediate_period == 0) and (filename_base is not None):
            #evaluating
            model.eval()
            c_r, c_p, s_r, s_p, _ = validate_benchmarks(filename_base, model)
            intermediate_s_r.append(s_r)
            intermediate_s_p.append(s_p)
            intermediate_c_r.append(c_r)
            intermediate_c_p.append(c_p)
            intermediate_n_epochs.append(epoch)
            intermediate_loss.append(loss.detach().cpu().numpy())
            model.train()


        optimizer.step()

    if filename_base is not None:

        fig, ax = plt.subplots()

        ax.plot(intermediate_n_epochs, intermediate_c_r, color='green', label='Combinational Recall')
        ax.plot(intermediate_n_epochs, intermediate_c_p,
                color='red', label='Combinational Precision')

        ax.plot(intermediate_n_epochs, intermediate_s_r, color='green',
                linestyle='dashed', label='Sequential Recall')
        ax.plot(intermediate_n_epochs, intermediate_s_p, color='red',
                linestyle='dashed', label='Sequential Precision')

        ax.set_ylim(ymin = 0, ymax = 1)

        ax2 = ax.twinx()
        ax2.plot(intermediate_n_epochs, intermediate_loss, color = 'blue', label = 'Loss Function')
        ax2.yaxis.tick_right()
        ax.legend(loc = 'upper left')
        ax2.legend()
        plt.show()

        plt.plot(losses)
        plt.title('Model performance')
        plt.xlabel('Epochs')
        #plt.yscale('log')
        plt.show()

    model.eval()
    y_pred = model(torch.FloatTensor(x_test).to(device))
    after_train = criterion(y_pred, torch.FloatTensor(y_test).to(device))
    print('Test loss after Training' , after_train.item())

    y_pred = y_pred.detach().cpu().numpy()
    #from sklearn.metrics import PrecisionRecallDisplay
    #display = PrecisionRecallDisplay.from_predictions(y_test, y_pred, name="Precision-Recall display")
    #_ = display.ax_.set_title("2-class Precision-Recall curve")
    #display.plot()
    #plt.show()

    return model


def evaluate_simple_methods_on_different_benches(dirname):

    import pandas as pd

    #lets say that hw is detectable if metric for safe is 10 times smaller than for compromised
    subfolders = [f.name for f in os.scandir(dirname) if f.is_dir()]
    means_safe = []
    means_trojan = []
    separables = [] 

    metrics = []
    classes = []
    bench_name = []

    for folder in subfolders:
        filename = dirname + '/' + folder + '/' + folder + '.v'
        framework = DetectionFramework()
        #framework.train_netlist_embeddings('benches/c2670_T001.v', 100)
        #framework.train_netlist_embeddings('TRIT-TC/c5315_T039/c5315_T039.v', 100)
        framework.train_netlist_embeddings(filename, 100)
        framework.shortest_distance_to_trojan()
        framework.build_maps()
        framework.tSNE(perplexity=300)
        mean_safe, mean_trojan = framework.get_average_metrics()
        means_safe.append(mean_safe)
        means_trojan.append(mean_trojan)
        # separables.append(separable)
        safe_metric, trojan_metrics = framework.get_metrics()
        for s_m in safe_metric:
            metrics.append(s_m)
            classes.append(0)
            bench_name.append(folder)
        for t_m in trojan_metrics:
            metrics.append(t_m)
            classes.append(1)
            bench_name.append(folder)
            


    df = pd.DataFrame({'name': subfolders, 'mean_safe': means_safe, 'mean_trojan': means_trojan})
    df.to_excel('benchmarks_averages.xlsx')

    df2 = pd.DataFrame({'name': bench_name, 'metrics': metrics,
                       'class': classes})
    df2.to_excel('metrics_all.xlsx')

def test_stability(filename, n_runs = 4, n_epochs = 1000000):
    total_df = pd.DataFrame()
    for run in range(n_runs):
        print(f'Run {run + 1} out of {n_runs}')
        frame = test_mlp(filename, n_epochs, False)
        frame['train_id'] = run
        total_df = total_df.append(frame)
    
        total_df.to_excel('stability_data1.xlsx')


    pass

def check_embeddings_distance_to_centers(dirname):
    import pandas as pd

    subfolders = [f.name for f in os.scandir(dirname) if f.is_dir()]

    total_distances = []
    bench_names = []
    bench_types = []

    for folder in subfolders:
        filename = dirname + '/' + folder + '/' + folder + '.v'
        framework = DetectionFramework()
        #framework.train_netlist_embeddings('benches/c2670_T001.v', 100)
        #framework.train_netlist_embeddings('TRIT-TC/c5315_T039/c5315_T039.v', 100)
        framework.train_netlist_embeddings(filename, 100)
        framework.shortest_distance_to_trojan()
        framework.build_maps()
        framework.tSNE(perplexity=300)
        distances = framework.distances_from_all_embeddings_to_center()
        total_distances += distances
        
        bench_names += [folder]*len(distances)
        if(folder.find('c')):
            bench_types += [0]*len(distances)
        else:
            bench_types += [0]*len(distances)

    df = pd.DataFrame(
        {'names': bench_names, 'distances': total_distances, 'bench_types': bench_types})
    df.to_excel('distances_all_embeddings_to_center.xlsx')

def check_trojan_embeddings_near_centers(dirname):
    import pandas as pd

    #lets say that hw is detectable if metric for safe is 10 times smaller than for compromised
    subfolders = [f.name for f in os.scandir(dirname) if f.is_dir()]
    
    distances = []
    centroid_ids = []
    bench_names = []
    bench_types = []

    for folder in subfolders:
        filename = dirname + '/' + folder + '/' + folder + '.v'
        framework = DetectionFramework()
        #framework.train_netlist_embeddings('benches/c2670_T001.v', 100)
        #framework.train_netlist_embeddings('TRIT-TC/c5315_T039/c5315_T039.v', 100)
        framework.train_netlist_embeddings(filename, 100)
        framework.shortest_distance_to_trojan()
        framework.build_maps()
        framework.tSNE(perplexity=300)
        distance, centroid_id = framework.distance_to_center()
        distances.append(distance)
        centroid_ids.append(centroid_id)
        bench_names.append(folder)
        if(folder.find('c')):
            bench_types.append(0)
        else:
            bench_types.append(1)

    df = pd.DataFrame(
        {'names': bench_names, 'distances': distances, 'centroid_ids': centroid_ids, 'bench_types': bench_types})
    df.to_excel('distances_to_center.xlsx')
        
    pass

#test()


#test_mlp('bench5_large_random_positions_labeled_intermediate')
#test_stability('bench5_large_random_positions_labeled_intermediate', 20, 500000)

####test_stability('bench5_large_random_positions_labeled_names_intermediate', 20, 10000)


#generate_trainset_from_benchmarks('./TRIT-TC', out_filename='bench5_large_random_positions',feature_distance= False, feature_position= True, timer = 12*60*60)

#generate_trainset_from_benchmarks('./TRIT-TC', out_filename='bench5_large_random_positions_labeled_names',
#                                 feature_distance=True, feature_position=False, timer=12*60*60)

#evaluate_simple_methods_on_different_benches('./TRIT-TC')
#check_trojan_embeddings_near_centers('./TRIT-TC')
check_embeddings_distance_to_centers('./TRIT-TC')
