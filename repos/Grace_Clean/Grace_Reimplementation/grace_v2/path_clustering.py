"""
Path clustering for hierarchical routing (route-style options).

Features match the design doc:
- hop_count
- stretch_to_shortest (hop stretch)
- overlap_to_shortest (edge overlap ratio)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

from . import config

# Avoid MKL thread-related leaks on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")


def nsfnet_edges() -> List[Tuple[int, int, float]]:
    return [
        (0, 1, 10),
        (0, 2, 10),
        (0, 3, 10),
        (1, 2, 10),
        (1, 7, 10),
        (2, 5, 10),
        (3, 4, 10),
        (3, 8, 10),
        (4, 5, 10),
        (4, 6, 10),
        (5, 12, 10),
        (5, 13, 10),
        (6, 7, 10),
        (7, 10, 10),
        (8, 9, 10),
        (8, 11, 10),
        (9, 10, 10),
        (9, 12, 10),
        (10, 11, 10),
        (10, 13, 10),
        (11, 12, 10),
        (12, 13, 10),
    ]


def build_nsfnet_graph() -> nx.Graph:
    g = nx.Graph()
    for u, v, _ in nsfnet_edges():
        g.add_edge(u, v, weight=1.0)
    return g


@dataclass
class ClusterInfo:
    paths: List[List[int]]
    mask: Dict[int, List[int]]  # node -> allowed next hops
    summary: Dict[str, float]


class PathClusterManager:
    def __init__(self, graph: nx.Graph | None = None):
        self.graph = graph or build_nsfnet_graph()
        self.dist_matrix: Dict[int, Dict[int, int]] = dict(
            nx.all_pairs_shortest_path_length(self.graph)
        )
        self.network_diameter = float(
            max(max(d.values()) for d in self.dist_matrix.values())
        )
        self.clusters: Dict[Tuple[int, int], Dict[int, ClusterInfo]] = {}
        self.candidate_masks: Dict[Tuple[int, int], Dict[int, List[int]]] = {}
    def k_shortest_paths(self, src: int, dst: int, k: int) -> List[List[int]]:
        try:
            gen = nx.shortest_simple_paths(self.graph, src, dst, weight="weight")
            return list(islice(gen, k))
        except nx.NetworkXNoPath:
            return []

    def _feature_vector(
        self, path: List[int], shortest_edges: set, shortest_hops: int
    ) -> List[float]:
        hop_count = len(path) - 1
        stretch = hop_count / float(shortest_hops) if shortest_hops > 0 else 1.0
        edges = set((path[i], path[i + 1]) for i in range(len(path) - 1))
        edges |= set((b, a) for a, b in edges)
        overlap = len(edges & shortest_edges) / float(len(shortest_edges) or 1)
        return [hop_count, stretch, overlap]

    def _make_mask(self, paths: List[List[int]]) -> Dict[int, List[int]]:
        mask: Dict[int, set] = {}
        for p in paths:
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                mask.setdefault(u, set()).add(v)
        return {k: sorted(v) for k, v in mask.items()}

    def build_clusters_for_pair(
        self, src: int, dst: int, *, k: int = config.K_PATHS, n_clusters: int = config.N_CLUSTERS
    ) -> Dict[int, ClusterInfo]:
        paths = self.k_shortest_paths(src, dst, k)
        if not paths:
            return {}
        self.candidate_masks[(src, dst)] = self._make_mask(paths)

        shortest = paths[0]
        shortest_edges = set((shortest[i], shortest[i + 1]) for i in range(len(shortest) - 1))
        shortest_edges |= set((b, a) for a, b in shortest_edges)
        shortest_hops = len(shortest) - 1

        feats = np.array([self._feature_vector(p, shortest_edges, shortest_hops) for p in paths])
        k_eff = min(n_clusters, len(paths))
        if k_eff <= 1:
            labels = np.zeros(len(paths), dtype=int)
            centers = np.mean(feats, axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=k_eff, random_state=42, n_init=10)
            labels = km.fit_predict(feats)
            centers = km.cluster_centers_

        clusters: Dict[int, ClusterInfo] = {}
        for cid in range(k_eff):
            idxs = np.where(labels == cid)[0]
            cluster_paths = [paths[i] for i in idxs]
            if not cluster_paths:
                continue
            mask = self._make_mask(cluster_paths)
            center = centers[cid] if cid < len(centers) else np.mean(feats[idxs], axis=0)
            clusters[cid] = ClusterInfo(
                paths=cluster_paths,
                mask=mask,
                summary={
                    "avg_hops": float(np.mean([len(p) - 1 for p in cluster_paths])),
                    "center_hop": float(center[0]),
                    "center_stretch": float(center[1]),
                    "center_overlap": float(center[2]),
                },
            )

        # Sort clusters by avg_hops to get deterministic IDs (short -> long)
        sorted_items = sorted(clusters.items(), key=lambda kv: kv[1].summary["avg_hops"])
        remapped: Dict[int, ClusterInfo] = {}
        for new_id, (_, info) in enumerate(sorted_items):
            remapped[new_id] = info
        self.clusters[(src, dst)] = remapped
        return remapped

    def build_all(self, flow_pairs: List[Tuple[int, int]]) -> Dict[Tuple[int, int], Dict[int, ClusterInfo]]:
        for src, dst in flow_pairs:
            self.build_clusters_for_pair(src, dst)
        return self.clusters

    def get_allowed_next_hops(self, src: int, dst: int, cluster_id: int, node: int) -> List[int]:
        pair = (src, dst)
        if pair not in self.clusters:
            return []
        cluster = self.clusters[pair].get(cluster_id)
        if not cluster:
            return []
        return cluster.mask.get(node, [])
    
    def get_candidate_next_hops(self, src: int, dst: int, node: int) -> List[int]:
      """Union-of-k-shortest-paths next-hop mask (no clustering / no bandit)."""
      pair = (src, dst)
      if pair in self.candidate_masks:
          return self.candidate_masks[pair].get(node, [])
      
        # Derive from loaded clusters if available
      if pair in self.clusters and self.clusters[pair]:
          all_paths: List[List[int]] = []
          for info in self.clusters[pair].values():
              all_paths.extend(info.paths)
          self.candidate_masks[pair] = self._make_mask(all_paths)
          return self.candidate_masks[pair].get(node, [])

      # Fallback: compute directly
      paths = self.k_shortest_paths(src, dst, config.K_PATHS)
      self.candidate_masks[pair] = self._make_mask(paths)
      return self.candidate_masks[pair].get(node, [])
    

    def save(self, path: Path) -> None:
        serializable = {}
        for (src, dst), cluster_map in self.clusters.items():
            serializable[f"{src}-{dst}"] = {
                str(cid): {
                    "paths": info.paths,
                    "mask": {str(k): v for k, v in info.mask.items()},
                    "summary": info.summary,
                }
                for cid, info in cluster_map.items()
            }
        payload = {
            "clusters": serializable,
            "dist_matrix": self.dist_matrix,
            "network_diameter": self.network_diameter,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "PathClusterManager":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        mgr = cls()
        mgr.dist_matrix = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in payload["dist_matrix"].items()}
        mgr.network_diameter = float(payload.get("network_diameter", mgr.network_diameter))
        mgr.clusters = {}
        for pair_key, cluster_map in payload["clusters"].items():
            src, dst = map(int, pair_key.split("-"))
            mgr.clusters[(src, dst)] = {}
            for cid_str, info in cluster_map.items():
                cid = int(cid_str)
                mask = {int(k): list(map(int, v)) for k, v in info["mask"].items()}
                mgr.clusters[(src, dst)][cid] = ClusterInfo(
                    paths=[list(map(int, p)) for p in info["paths"]],
                    mask=mask,
                    summary={k: float(v) for k, v in info["summary"].items()},
                )
        return mgr
