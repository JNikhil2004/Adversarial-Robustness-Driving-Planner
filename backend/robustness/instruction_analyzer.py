"""
instruction_analyzer.py
-----------------------
Unified natural language verification for passenger instructions:
1. Prompt Risk Assessment (Deep Transformer Classifier):
   Detects adversarial prompt injections, safety overrides, jailbreaks, and trigger terms.
   Computes token-level attention saliency.
2. Linguistic Ambiguity Assessment (Semantic Groundability Net):
   Detects deictic vagueness ('there', 'it', 'that thing'), missing spatial anchors,
   and ungrounded pronouns.
"""

from typing import Dict, List, Any
import torch

from backend.preprocessing.preprocessor import clean_text, tokenize, extract_object_keywords
from backend.robustness.neural_modules import (
    TransformerPromptClassifier,
    SemanticGroundabilityNet,
    NeuralTextTokenizer,
    _shared_tokenizer,
    DEVICE,
)
from backend.utils.config import (
    TRANSFORMER_PROMPT_PATH,
    SEMANTIC_GROUNDABILITY_PATH,
    VAGUE_REFERENTS,
    VAGUE_QUANTIFIERS,
    MIN_INSTRUCTION_TOKENS,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["trusted", "suspicious", "high_risk"]

HIGH_RISK_LEXICON = [
    "ignore previous", "ignore your", "override", "developer mode",
    "disable", "forget your", "forget you are", "no matter what",
    "don't stop", "do not stop", "don't brake", "crash", "run the red",
    "run a red", "sidewalk", "maximum speed", "as fast as possible",
    "disregard", "pretend the", "take control", "safety checks",
    "emergency braking", "bypass", "jailbreak", "unrestricted",
]

SUSPICIOUS_LEXICON = [
    "always", "no one will notice", "quickly", "don't check", "whatever you think",
    "go there", "near it", "that thing", "somewhere",
]

DIRECTIONAL_TERMS = {
    "left", "right", "straight", "forward", "behind", "front", "ahead",
    "near", "next", "beside", "before", "after", "corner", "intersection",
    "past", "across", "between",
}


class InstructionAnalyzer:
    """Analyzes passenger instructions for adversarial prompt injection and linguistic ambiguity."""

    def __init__(self):
        self.tokenizer: NeuralTextTokenizer = _shared_tokenizer
        self.prompt_model: TransformerPromptClassifier = None
        self.groundability_model: SemanticGroundabilityNet = None
        self._load_models()

    def _load_models(self) -> None:
        # 1. Load Transformer Prompt Classifier
        try:
            self.prompt_model = TransformerPromptClassifier().to(DEVICE)
            if TRANSFORMER_PROMPT_PATH.exists():
                state = torch.load(TRANSFORMER_PROMPT_PATH, map_location=DEVICE, weights_only=True)
                self.prompt_model.load_state_dict(state)
                self.prompt_model.eval()
                logger.info(f"Loaded Transformer Prompt Classifier from {TRANSFORMER_PROMPT_PATH.name}")
            else:
                self.prompt_model.eval()
        except Exception as exc:
            logger.error(f"Failed to load TransformerPromptClassifier: {exc}")
            self.prompt_model = None

        # 2. Load Semantic Groundability Net
        try:
            self.groundability_model = SemanticGroundabilityNet().to(DEVICE)
            if SEMANTIC_GROUNDABILITY_PATH.exists():
                state = torch.load(SEMANTIC_GROUNDABILITY_PATH, map_location=DEVICE, weights_only=True)
                self.groundability_model.load_state_dict(state)
                self.groundability_model.eval()
                logger.info(f"Loaded SemanticGroundabilityNet from {SEMANTIC_GROUNDABILITY_PATH.name}")
            else:
                self.groundability_model.eval()
        except Exception as exc:
            logger.error(f"Failed to load SemanticGroundabilityNet: {exc}")
            self.groundability_model = None

    # --------------------------------------------------------------------------
    # Prompt Risk Classification
    # --------------------------------------------------------------------------

    def evaluate_prompt_risk(self, instruction: str) -> Dict[str, Any]:
        """Classify instruction into trusted, suspicious, or high_risk with attention saliency."""
        try:
            cleaned = clean_text(instruction)
            lowered = instruction.lower()
            trigger_terms = [
                t for t in HIGH_RISK_LEXICON + SUSPICIOUS_LEXICON
                if t in lowered or clean_text(t) in cleaned
            ]

            if not cleaned:
                return {
                    "label": "high_risk" if trigger_terms else "trusted",
                    "confidence": 0.5,
                    "trigger_terms": trigger_terms,
                    "score": 50.0,
                    "token_saliency": [],
                }

            if self.prompt_model is not None:
                probs, saliency_list = self.prompt_model.get_token_saliency(cleaned, self.tokenizer)
                probs_list = probs.cpu().tolist()
                best_idx = int(torch.argmax(probs).item())
                label = CLASS_NAMES[best_idx]
                confidence = float(probs_list[best_idx])
            else:
                label = "high_risk" if any(t in HIGH_RISK_LEXICON for t in trigger_terms) else "trusted"
                confidence = 0.8
                saliency_list = [{"token": w, "saliency": 0.5, "is_high_risk": False} for w in cleaned.split()]

            # Guardrail safety-net for exact lexical injection triggers
            high_risk_hits = [t for t in trigger_terms if t in HIGH_RISK_LEXICON]
            if high_risk_hits and label != "high_risk":
                label = "high_risk"
                confidence = max(confidence, 0.88)
                for s in saliency_list:
                    if any(t in s["token"].lower() for t in high_risk_hits):
                        s["saliency"] = max(s["saliency"], 0.95)
                        s["is_high_risk"] = True
            elif trigger_terms and label == "trusted":
                label = "suspicious"
                confidence = max(confidence, 0.65)

            base = {"trusted": 92.0, "suspicious": 55.0, "high_risk": 15.0}[label]
            score = float(max(0.0, min(100.0, base + (confidence - 0.5) * 16.0)))

            return {
                "label": label,
                "confidence": round(confidence, 3),
                "trigger_terms": trigger_terms,
                "score": round(score, 1),
                "token_saliency": saliency_list,
            }
        except Exception as exc:
            logger.error(f"Prompt risk analysis failed: {exc}")
            return {
                "label": "suspicious",
                "confidence": 0.0,
                "trigger_terms": [],
                "score": 50.0,
                "token_saliency": [],
            }

    # --------------------------------------------------------------------------
    # Ambiguity Analysis
    # --------------------------------------------------------------------------

    def evaluate_ambiguity(self, instruction: str) -> Dict[str, Any]:
        """Assess semantic clarity, completeness, and deictic vagueness."""
        try:
            cleaned = clean_text(instruction)
            tokens = tokenize(instruction)
            reasons: List[str] = []

            vague_hits = [t for t in tokens if t in VAGUE_REFERENTS]
            hedge_hits = [q for q in VAGUE_QUANTIFIERS if q in cleaned]
            objects = extract_object_keywords(instruction)
            has_direction = any(t in DIRECTIONAL_TERMS for t in tokens)

            if len(tokens) < MIN_INSTRUCTION_TOKENS:
                reasons.append(f"Instruction is underspecified ({len(tokens)} token(s)) -- lacks clear driving intent.")
            if vague_hits:
                reasons.append(f"Contains ambiguous deictic referent(s): {sorted(set(vague_hits))}.")
            if hedge_hits:
                reasons.append(f"Contains non-committal hedging language: {hedge_hits}.")
            if not objects and not has_direction:
                reasons.append("Missing concrete visual target and directional anchor for ego-planner.")
            if vague_hits and not objects:
                reasons.append("Dangling pronoun/referent without resolvable visual antecedent.")

            neural_clarity = 90.0
            if self.groundability_model is not None and cleaned:
                indices, _ = self.tokenizer.encode(cleaned)
                x = torch.tensor([indices], dtype=torch.long, device=DEVICE)
                with torch.no_grad():
                    clarity_out, def_out = self.groundability_model(x)
                    neural_clarity = float(clarity_out.item())
                    deficiencies = def_out[0].cpu().tolist()
                    if deficiencies[0] > 0.6 and not vague_hits:
                        reasons.append("Deep Groundability Net flagged high referent ambiguity.")

            penalty = len(reasons) * 22.0 + (12.0 if len(tokens) < MIN_INSTRUCTION_TOKENS else 0.0)
            calibrated_score = max(0.0, min(100.0, min(neural_clarity, 100.0 - penalty)))

            return {
                "is_ambiguous": len(reasons) > 0 or calibrated_score < 60.0,
                "ambiguity_reasons": reasons,
                "vague_tokens": list(set(vague_hits)),
                "score": round(calibrated_score, 1),
            }
        except Exception as exc:
            logger.error(f"Ambiguity analysis failed: {exc}")
            return {
                "is_ambiguous": True,
                "ambiguity_reasons": ["Internal error during linguistic analysis."],
                "vague_tokens": [],
                "score": 50.0,
            }


_analyzer_instance = None


def get_instruction_analyzer() -> InstructionAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = InstructionAnalyzer()
    return _analyzer_instance
