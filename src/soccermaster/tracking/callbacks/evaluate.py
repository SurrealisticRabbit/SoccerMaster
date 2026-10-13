from soccermaster.tracking.core import Evaluator
from soccermaster.tracking.callbacks import Callback
from soccermaster.tracking.engine import TrackingEngine

import logging

log = logging.getLogger(__name__)


class Evaluate(Callback):
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def on_dataset_track_end(self, engine: "TrackingEngine"):
        if engine.tracker_state.detections_gt is not None:
            log.info("Starting evaluation.")
            self.evaluator.run(engine.tracker_state)
        else:
            log.warning(
                "Skipping evaluation because there's no ground truth detection."
            )