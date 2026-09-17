"""Build the joint user-entity graph and load entity-indexed image features."""

import os
import random
import collections

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from data_loader.loader_base import DataLoaderBase


def load_image_embeddings(path):
    """Read tab-separated original IDs, remapped entity IDs, and feature vectors."""
    id2emb = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # org_id remap_id embedding
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            _, remap_id_str, emb_str = parts[0], parts[1], parts[2]
            remap_id = int(remap_id_str)
            vec = np.array([float(x) for x in emb_str.strip().split()], dtype=np.float32)
            id2emb[remap_id] = vec
    return id2emb


class DataLoaderKGAT(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = pd.read_csv(
            self.kg_file,
            sep=r"\s+",
            header=None,
            names=["h", "r", "t"],
            engine="python",
            dtype=str,  # Read as strings before validating numeric IDs.
        )
        # Strip surrounding whitespace and require numeric values.
        for c in ["h", "r", "t"]:
            kg_data[c] = pd.to_numeric(kg_data[c].str.strip(), errors="raise")

        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

        self.id2emb = None
        if args.image_emb_path:
            self.id2emb = load_image_embeddings(args.image_emb_path)

        # Initialize entity features with CLIP vectors where available and zeros otherwise.
        self.entity_init_features = None
        if self.id2emb is not None:
            d_clip = len(next(iter(self.id2emb.values())))
            init = np.zeros((self.n_entities, d_clip), dtype=np.float32)
            for eid, vec in self.id2emb.items():
                if 0 <= eid < self.n_entities:
                    init[eid] = vec
            self.entity_init_features = torch.from_numpy(init)  # [n_entities, d_clip]

    def construct_data(self, kg_data):
        # Require integer IDs before constructing graph tensors.
        kg_data["h"] = pd.to_numeric(kg_data["h"], errors="raise").astype(np.int64)
        kg_data["r"] = pd.to_numeric(kg_data["r"], errors="raise").astype(np.int64)
        kg_data["t"] = pd.to_numeric(kg_data["t"], errors="raise").astype(np.int64)

        # add inverse kg data
        n_relations = max(kg_data["r"]) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({"h": "t", "t": "h"}, axis="columns")
        inverse_kg_data["r"] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # Reserve relations 0 and 1 for user-item interactions and their inverses.
        # Original KG relation IDs shift by +2; has_image_rel_id uses this space.
        kg_data["r"] += 2
        self.n_relations = max(kg_data["r"]) + 1
        self.n_entities = max(max(kg_data["h"]), max(kg_data["t"])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        # Entity rows precede user rows in the shared embedding table.
        self.cf_train_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(
                np.int32
            ),
            self.cf_train_data[1].astype(np.int32),
        )
        self.cf_test_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(
                np.int32
            ),
            self.cf_test_data[1].astype(np.int32),
        )

        self.train_user_dict = {
            k + self.n_entities: np.unique(v).astype(np.int32)
            for k, v in self.train_user_dict.items()
        }
        self.test_user_dict = {
            k + self.n_entities: np.unique(v).astype(np.int32)
            for k, v in self.test_user_dict.items()
        }

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(
            np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=["h", "r", "t"]
        )
        cf2kg_train_data["h"] = self.cf_train_data[0]
        cf2kg_train_data["t"] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(
            np.ones((self.n_cf_train, 3), dtype=np.int32), columns=["h", "r", "t"]
        )
        inverse_cf2kg_train_data["h"] = self.cf_train_data[1]
        inverse_cf2kg_train_data["t"] = self.cf_train_data[0]

        self.kg_train_data = pd.concat(
            [kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True
        )
        self.kg_train_data[["h", "r", "t"]] = self.kg_train_data[["h", "r", "t"]].astype(np.int64)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def convert_coo2tensor(self, coo):
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1] * len(rows)
            adj = sp.coo_matrix(
                (vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities)
            )
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):
        def symmetric_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return norm_adj.tocoo()

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))

            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.laplacian_type == "symmetric":
            norm_lap_func = symmetric_norm_lap
        elif self.laplacian_type == "random-walk":
            norm_lap_func = random_walk_norm_lap
        else:
            raise NotImplementedError

        self.laplacian_dict = {}
        for r, adj in self.adjacency_dict.items():
            self.laplacian_dict[r] = norm_lap_func(adj)

        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

    def print_info(self, logging):
        if logging:
            logging.info("n_users:           %d" % self.n_users)
            logging.info("n_items:           %d" % self.n_items)
            logging.info("n_entities:        %d" % self.n_entities)
            logging.info("n_users_entities:  %d" % self.n_users_entities)
            logging.info("n_relations:       %d" % self.n_relations)

            logging.info("n_h_list:          %d" % len(self.h_list))
            logging.info("n_t_list:          %d" % len(self.t_list))
            logging.info("n_r_list:          %d" % len(self.r_list))

            logging.info("n_cf_train:        %d" % self.n_cf_train)
            logging.info("n_cf_test:         %d" % self.n_cf_test)

            logging.info("n_kg_train:        %d" % self.n_kg_train)
