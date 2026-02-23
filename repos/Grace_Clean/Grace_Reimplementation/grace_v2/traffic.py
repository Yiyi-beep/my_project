"""
Traffic generators for multi-main-flow training.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from ns.packet.packet import Packet

from . import config


class FlowController:
    """Holds the current cluster assignment for a flow."""

    def __init__(self, flow_id: int, initial_cluster: int = 0, mode: str = "light"):
        self.flow_id = flow_id
        self.cluster_id = initial_cluster
        self.mode = mode

    def set_cluster(self, cid: int):
        self.cluster_id = cid

    def set_mode(self, mode: str):
        self.mode = mode


class FlowGenerator:
    def __init__(
        self,
        env,
        element_id: int,
        dst_id: int,
        controller: FlowController,
        *,
        pps: float,
        duration_ms: float,
        measure_start: float,
        measure_end: float,
        window_ms: float,
        packet_size: int = config.PACKET_SIZE_BYTES,
        arrival_process: str = "poisson",
        metrics=None,
    ):
        self.env = env
        self.element_id = element_id
        self.dst_id = dst_id
        self.controller = controller
        self.packet_size = packet_size
        self.measure_start = measure_start
        self.measure_end = measure_end
        self.window_ms = window_ms
        self.metrics = metrics

        self.flow_id = controller.flow_id
        self.packet_counter = 0
        self.out = None
        self.packets_sent = 0

        mean_iat = 1000.0 / float(pps)
        if arrival_process == "const":
            self.arrival_dist = lambda: float(mean_iat)
        else:
            self.arrival_dist = lambda: float(np.random.exponential(mean_iat))

        self.finish = duration_ms
        self.action = env.process(self.run())

    def run(self):
        while self.env.now < self.finish:
            yield self.env.timeout(self.arrival_dist())
            if self.env.now >= self.finish:
                break
            self.packet_counter += 1
            pkt = Packet(
                self.env.now,
                float(self.packet_size),
                self.packet_counter,
                src=self.element_id,
                dst=self.dst_id,
                flow_id=self.flow_id,
            )
            pkt.packet_id = self.packet_counter
            pkt.ttl = config.TTL_HOPS
            pkt.cluster_id = self.controller.cluster_id
            pkt.mode = self.controller.mode

            in_measure = self.measure_start <= self.env.now < self.measure_end
            pkt.generated_in_measure = in_measure
            if in_measure:
                pkt.window_id = int((self.env.now - self.measure_start) // self.window_ms)
                if self.metrics:
                    self.metrics.on_generate(pkt, self.env.now)
            else:
                pkt.window_id = None

            if self.out:
                self.out.put(pkt)
                self.packets_sent += 1


class TrafficManager:
    def __init__(self, env):
        self.env = env
        self.generators: List[FlowGenerator] = []
        self.controllers: List[FlowController] = []

    def add_flow(
        self,
        flow_id: int,
        src: int,
        dst: int,
        *,
        pps: float,
        duration_ms: float,
        measure_start: float,
        measure_end: float,
        window_ms: float,
        initial_cluster: int,
        mode: str,
        metrics=None,
    ) -> Tuple[FlowGenerator, FlowController]:
        controller = FlowController(flow_id, initial_cluster, mode=mode)
        gen = FlowGenerator(
            self.env,
            src,
            dst,
            controller,
            pps=pps,
            duration_ms=duration_ms,
            measure_start=measure_start,
            measure_end=measure_end,
            window_ms=window_ms,
            metrics=metrics,
        )
        self.generators.append(gen)
        self.controllers.append(controller)
        return gen, controller

