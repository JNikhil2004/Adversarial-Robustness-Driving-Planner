"""
ambiguity_detector.py - Backward-compatible wrapper around unified instruction_analyzer.py.
"""
from backend.robustness.instruction_analyzer import get_instruction_analyzer, InstructionAnalyzer


class AmbiguityDetector:
    def __init__(self):
        self._analyzer = get_instruction_analyzer()

    def analyze(self, instruction: str):
        return self._analyzer.evaluate_ambiguity(instruction)


_detector_instance = None


def get_ambiguity_detector() -> AmbiguityDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AmbiguityDetector()
    return _detector_instance
