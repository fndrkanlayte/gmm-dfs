"""Optional observation functions.

``DownstreamObservation`` = sumo-rl default + outgoing-lane density (the
downstream-observation ablation arm, --obs downstream). Lane lists are SORTED
(sumo-rl's out_lanes comes from a set with hash-randomised order).
"""
import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction

LANE_CAP = 50.0  # count normaliser (veh / lane), matches MAX_QUEUE_PER_LANE


def _sorted_lanes(ts):
    return sorted(dict.fromkeys(ts.lanes)), sorted(set(ts.out_lanes))


class DownstreamObservation(ObservationFunction):
    """sumo-rl default observation + outgoing-lane density (downstream info).

    [phase one-hot | min-green flag | in-lane density | in-lane queue |
     out-lane density]; the first four blocks are byte-identical to
    DefaultObservationFunction, out-lane density uses sumo-rl's own formula on
    SORTED out lanes. This gives the controller downstream occupancy
    information, which the default observation lacks.
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


OBS_CLASSES = {'downstream': DownstreamObservation}
