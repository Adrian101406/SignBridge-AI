import json
from pathlib import Path


class TextToBIM:
    """Maps structured MCIE medical concepts to the current 40-class BIM gloss vocabulary."""

    def __init__(self, vocabulary_path=None):
        if vocabulary_path is None:
            vocabulary_path = Path(__file__).with_name("medical_bim_vocabulary.json")
        self.mapping = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))

    def check_concept_support(self, mcie_output: dict) -> dict:
        supported, unsupported, glosses = [], [], []

        for concept in mcie_output.get("medical_concepts", []):
            if concept in self.mapping:
                supported.append(concept)
                glosses.extend(self.mapping[concept])
            else:
                unsupported.append(concept)

        return {
            "supported_concepts": supported,
            "unsupported_concepts": unsupported,
            "bim_glosses": glosses
        }

    def build_output(self, mcie_output: dict) -> dict:
        if not mcie_output.get("ready_for_translation", False):
            return {
                "success": False,
                "status": "NOT_READY",
                "message": "MCIE output is not ready for translation.",
                "bim_glosses": []
            }

        result = self.check_concept_support(mcie_output)
        supported = result["supported_concepts"]
        unsupported = result["unsupported_concepts"]
        glosses = result["bim_glosses"]

        if unsupported:
            return {
                "success": False,
                "status": "UNSUPPORTED_CONCEPT",
                "intent": mcie_output.get("intent"),
                "supported_concepts": supported,
                "unsupported_concepts": unsupported,
                "bim_glosses": glosses,
                "message": "One or more medical concepts are not currently supported by Text-to-BIM."
            }

        return {
            "success": True,
            "status": "SUPPORTED",
            "intent": mcie_output.get("intent"),
            "supported_concepts": supported,
            "unsupported_concepts": [],
            "bim_glosses": glosses,
            "message": "Medical concepts are supported by the current BIM vocabulary."
        }

    def build_avatar_payload(self, mcie_output: dict) -> dict:
        result = self.build_output(mcie_output)

        if not result["success"]:
            return {
                "success": False,
                "status": result["status"],
                "avatar_ready": False,
                "bim_gloss_sequence": [],
                "unsupported_concepts": result.get("unsupported_concepts", [])
            }

        return {
            "success": True,
            "status": "READY_FOR_AVATAR",
            "source": "text_to_bim",
            "intent": result["intent"],
            "medical_concepts": result["supported_concepts"],
            "bim_gloss_sequence": result["bim_glosses"],
            "grammar_validated": False,
            "avatar_ready": True
        }
