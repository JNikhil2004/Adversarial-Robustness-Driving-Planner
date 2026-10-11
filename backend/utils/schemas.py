"""
Pydantic schemas describing the API's response contracts.

These are used purely for response validation / OpenAPI documentation --
the evaluation pipeline builds plain dicts internally (see decision_engine.py)
which are validated against these models before being returned to the client.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class PromptRiskAssessment(BaseModel):
    label: str = Field(..., description="trusted | suspicious | high_risk")
    confidence: float
    trigger_terms: List[str] = []
    score: float = Field(..., description="0-100, higher = more trustworthy")


class InstructionClarity(BaseModel):
    is_ambiguous: bool
    ambiguity_reasons: List[str] = []
    score: float = Field(..., description="0-100, higher = clearer instruction")


class SceneUnderstanding(BaseModel):
    detected_objects: List[str] = []
    detection_count: int
    mean_confidence: float
    attack_applied: Optional[str] = None
    score: float = Field(..., description="0-100 camera pipeline health score")


class RadarMotionAnalysis(BaseModel):
    total_tracks: int
    anomalous_tracks: int
    anomalies: List[str] = []
    attack_applied: Optional[str] = None
    score: float = Field(..., description="0-100 radar pipeline health score")


class CrossModalVerification(BaseModel):
    matched_objects: List[str] = []
    unmatched_instruction_reference: bool
    consistency_score: float = Field(..., description="0-100")


class RobustnessBreakdown(BaseModel):
    prompt_robustness: float
    camera_robustness: float
    radar_robustness: float
    overall_robustness: float


class EvaluationResult(BaseModel):
    prompt_risk: PromptRiskAssessment
    instruction_clarity: InstructionClarity
    scene_understanding: SceneUnderstanding
    radar_motion: RadarMotionAnalysis
    cross_modal: CrossModalVerification
    robustness: RobustnessBreakdown
    overall_safety_score: float
    risk_level: str
    recommendation: str
    explanation: List[str]
