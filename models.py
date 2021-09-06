# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from torch.nn import functional as F, Parameter
import torch
from torch import nn


class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)

                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    if (self.name == "SimplE" or self.name == "SimplE_o" or self.name == "ED_SimplE_old" or self.name == "ED_SimplE_openke" or self.name == "ED_SimplE_openke_new"):
                        q = self.get_queries(these_queries)
                        q_inv = self.get_queries_inv(these_queries)
                        scores = (q @ rhs + q_inv @ rhs) / 2
                    else:
                        q = self.get_queries(these_queries)
                        scores = q @ rhs

                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class InversE(KBCModel):

    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(InversE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.name = "InversE"

        self.ent_re = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel_inv_re = nn.Embedding(sizes[1], rank, sparse=True)
        self.rel_re = nn.Embedding(sizes[1], rank, sparse=True)

        self.ent_im = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel_inv_im = nn.Embedding(sizes[1], rank, sparse=True)
        self.rel_im = nn.Embedding(sizes[1], rank, sparse=True)

        self.ent_re.weight.data *= init_size
        self.rel_inv_re.weight.data *= init_size
        self.rel_re.weight.data *= init_size

        self.ent_im.weight.data *= init_size
        self.rel_inv_im.weight.data *= init_size
        self.rel_im.weight.data *= init_size


    def score(self, x):

        lhs_re = self.ent_re(x[:, 0])
        rel_re = self.rel_re(x[:, 1])
        rhs_re = self.ent_re(x[:, 2])
        rel_inv_re = self.rel_inv_re(x[:, 1])

        lhs_im = self.ent_im(x[:, 0])
        rel_im = self.rel_im(x[:, 1])
        rhs_im = self.ent_im(x[:, 2])
        rel_inv_im = self.rel_inv_im(x[:, 1])

        return (torch.sum(
            (lhs_re * rel_re - lhs_im * rel_im) * rhs_re +
            (lhs_re * rel_im + lhs_im * rel_re) * rhs_im,
            1, keepdim=True
        )   +  torch.sum(
            (lhs_re * rel_inv_re - lhs_im * rel_inv_im) * rhs_re +
            (lhs_re * rel_inv_im + lhs_im * rel_inv_re) * rhs_im,
            1, keepdim=True
        )
                  )/2

    def forward(self, x):

        lhs_re = self.ent_re(x[:, 0])
        rel_re = self.rel_re(x[:, 1])
        rhs_re = self.ent_re(x[:, 2])
        rel_inv_re = self.rel_inv_re(x[:, 1])

        lhs_im = self.ent_im(x[:, 0])
        rel_im = self.rel_im(x[:, 1])
        rhs_im = self.ent_im(x[:, 2])
        rel_inv_im = self.rel_inv_im(x[:, 1])

        return ((
            (lhs_re * rel_re - lhs_im * rel_im) @ self.ent_re.weight.transpose(0, 1) +
            (lhs_re * rel_im + lhs_im * rel_re) @ self.ent_im.weight.transpose(0, 1)
        ) + (
            (lhs_re * rel_inv_re - lhs_im * rel_inv_im) @ self.ent_re.weight.transpose(0, 1) +
            (lhs_re * rel_inv_im + lhs_im * rel_inv_re) @ self.ent_im.weight.transpose(0, 1)
        ))/2, (
            torch.sqrt(lhs_re ** 2 + lhs_im ** 2),
            torch.sqrt(rel_re ** 2 + rel_im ** 2),
            torch.sqrt(rhs_re ** 2 + rhs_im ** 2),
            torch.sqrt(rel_inv_re ** 2 + rel_inv_im ** 2),
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        rhs_comb = torch.cat([self.ent_re.weight.data, self.ent_im.weight.data], dim=1)
        return rhs_comb[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)
        # return self.rhs_re.weight.data[
        #     chunk_begin:chunk_begin + chunk_size
        # ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):

        lhs_re = self.ent_re(queries[:, 0])
        rel_re = self.rel_re(queries[:, 1])
        lhs_im = self.ent_im(queries[:, 0])
        rel_im = self.rel_im(queries[:, 1])

        return torch.cat([
            lhs_re * rel_re - lhs_im * rel_im,
            lhs_re * rel_im + lhs_im * rel_re
        ], 1)

    def get_queries_inv(self, queries: torch.Tensor):

        lhs_re = self.ent_re(queries[:, 0])
        rel_inv_re = self.rel_inv_re(queries[:, 1])
        lhs_im = self.ent_im(queries[:, 0])
        rel_inv_im = self.rel_inv_im(queries[:, 1])

        return torch.cat([
            lhs_re * rel_inv_re - lhs_im * rel_inv_im,
            lhs_re * rel_inv_im + lhs_im * rel_inv_re
        ], 1)
