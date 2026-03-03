"""
EvalForge main package.
Why did the model cross the road? To optimize its health score :)
"""

from evalforge.blind_spots import map_blind_spots
from evalforge.fragility import calculate_adversarial_fragility
from evalforge.mismatch import detect_confidence_accuracy_mismatch
from evalforge.stability import compute_stability_from_scores, evaluate_seed_stability

__all__ = [
	"map_blind_spots",
	"calculate_adversarial_fragility",
	"detect_confidence_accuracy_mismatch",
	"compute_stability_from_scores",
	"evaluate_seed_stability",
]
