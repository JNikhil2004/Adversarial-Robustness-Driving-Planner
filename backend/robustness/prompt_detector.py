"""
prompt_detector.py - Backward-compatible wrapper around unified instruction_analyzer.py.
"""
from backend.robustness.instruction_analyzer import get_instruction_analyzer, InstructionAnalyzer


class PromptDetector:
    def __init__(self):
        self._analyzer = get_instruction_analyzer()

    def predict(self, instruction: str):
        return self._analyzer.evaluate_prompt_risk(instruction)


_detector_instance = None


def get_prompt_detector() -> PromptDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PromptDetector()
    return _detector_instance
