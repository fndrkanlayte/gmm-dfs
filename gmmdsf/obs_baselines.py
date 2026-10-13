"""Observation / reward functions for the PressLight- and MPLight-style baselines.

Simplified re-implementations inside sumo-rl (not line-by-line ports):

PressLight (Wei et al., KDD 2019): state = current phase + vehicle counts on
  every incoming and outgoing lane; reward = -|intersection pressure|.
MPLight (Chen et al., AAAI 2020): state = current phase + per-PHASE movement
  pressure (FRAP-style phase features, computed from the same lane-pair
  movement map MaxPressure uses); reward = -|intersection pressure|; trained
  with full parameter sharing (no per-intersection identity embedding).

Lane lists are SORTED: sumo-rl's out_lanes comes from a set, whose order
changes between processes (string hash randomisation).
"""
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction

LANE_CAP = 50.0  # count normaliser (veh / lane), matches MAX_QUEUE_PER_LANE


def _sorted_lanes(ts):
    return sorted(dict.fromkeys(ts.lanes)), sorted(set(ts.out_lanes))


class PressLightObservation(ObservationFunction):
    """[phase one-hot | in-lane counts | out-lane counts] / LANE_CAP."""

    def __init__(self, ts):
        super().__init__(ts)
        self._in, self._out = None, None

    def _init(self):
        if self._in is None:
            self._in, self._out = _sorted_lanes(self.ts)

    def __call__(self):
        self._init()
        s = self.ts.sumo
        ph = [1.0 if self.ts.green_phase == i else 0.0 for i in range(self.ts.num_green_phases)]
        cin = [s.lane.getLastStepVehicleNumber(l) / LANE_CAP for l in self._in]
        cout = [s.lane.getLastStepVehicleNumber(l) / LANE_CAP for l in self._out]
        return np.array(ph + cin + cout, dtype=np.float32)

    def observation_space(self):
        n_in = len(dict.fromkeys(self.ts.lanes))
        n_out = len(set(self.ts.out_lanes))
        d = self.ts.num_green_phases + n_in + n_out
        return spaces.Box(low=0.0, high=np.inf, shape=(d,), dtype=np.float32)


class MPLightObservation(ObservationFunction):
    """[phase one-hot | per-phase movement pressure] (pressure / LANE_CAP)."""

    def __init__(self, ts):
        super().__init__(ts)
        self._mv = None

    def _init(self):
        if self._mv is None:
            from eval_maxpressure import phase_movements
            self._mv = phase_movements(self.ts.green_phases,
                                       self.ts.sumo.trafficlight.getControlledLinks(self.ts.id))

    def __call__(self):
        self._init()
        s = self.ts.sumo
        cnt = {}

        def c(l):
            if l not in cnt:
                cnt[l] = s.lane.getLastStepVehicleNumber(l)
            return cnt[l]
        ph = [1.0 if self.ts.green_phase == i else 0.0 for i in range(self.ts.num_green_phases)]
        pr = [sum(c(i) - c(o) for i, o in pairs) / LANE_CAP for pairs in self._mv]
        return np.array(ph + pr, dtype=np.float32)

    def observation_space(self):
        d = 2 * self.ts.num_green_phases
        return spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32)


PRESSURE_SCALE = 10.0  # reward scaling only (optimal policy unchanged)


def abs_pressure_reward(ts):
    """PressLight / MPLight reward: -|#veh approaching - #veh leaving| / 10."""
    return -abs(ts.get_pressure()) / PRESSURE_SCALE


class DownstreamObservation(ObservationFunction):
    """sumo-rl default observation + outgoing-lane density (downstream info).

    [phase one-hot | min-green flag | in-lane density | in-lane queue |
     out-lane density]; the first four blocks are byte-identical to
    DefaultObservationFunction, out-lane density uses sumo-rl's own formula on
    SORTED out lanes. This gives the controller the same downstream occupancy
    MaxPressure / PressLight / MPLight use.
    """

    def __init__(self, ts):
        super().__init__(ts)
        self._out = None

    def __call__(self):
        from sumo_rl.environment.observations import DefaultObservationFunction
        if self._out is None:
            self._out = sorted(set(self.ts.out_lanes))
        base = DefaultObservationFunction(self.ts)()
        s = self.ts.sumo
        dens = [min(1.0, s.lane.getLastStepVehicleNumber(l)
                    / (self.ts.lanes_length[l] / (self.ts.MIN_GAP + s.lane.getLastStepLength(l))))
                for l in self._out]
        return np.concatenate([base, np.array(dens, dtype=np.float32)]).astype(np.float32)

    def observation_space(self):
        d = (self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)
             + len(set(self.ts.out_lanes)))
        return spaces.Box(low=0.0, high=1.0, shape=(d,), dtype=np.float32)


OBS_CLASSES = {'presslight': PressLightObservation, 'mplight': MPLightObservation,
               'downstream': DownstreamObservation}
