import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import json
from typing import Dict, List, Tuple
import warnings
import os
import importlib.util

warnings.filterwarnings("ignore")


class AITextDetector:
    def __init__(self):
        """Initialize AI text detector with multiple detection methods"""
        self.models = {}
        self.tokenizers = {}
        self._models_loaded_flags = {
            "roberta": False,
            "gpt2_detector": False,
            # "ai_classifier": False, # Example if you add more
        }

    def _ensure_models_loaded(self):
        """Load all defined models if they haven't been loaded yet."""
        if not self._models_loaded_flags["roberta"]:
            if self.load_roberta_detector():
                self._models_loaded_flags["roberta"] = True
        if not self._models_loaded_flags["gpt2_detector"]:
            if self.load_gpt2_detector():
                self._models_loaded_flags["gpt2_detector"] = True
        # Add more model loading checks here if needed

    def load_roberta_detector(self):
        """Load RoBERTa-based AI detector (OpenAI detector)"""
        try:
            model_name = "roberta-base-openai-detector"
            print(f"Loading {model_name}...")
            self.tokenizers["roberta"] = AutoTokenizer.from_pretrained(model_name)
            self.models["roberta"] = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            print("RoBERTa detector loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading RoBERTa detector: {e}")
            return False

    def load_gpt2_detector(self):
        """Load GPT-2 output detector"""
        try:
            model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
            print(f"Loading {model_name}...")
            self.tokenizers["gpt2_detector"] = AutoTokenizer.from_pretrained(model_name)
            self.models["gpt2_detector"] = (
                AutoModelForSequenceClassification.from_pretrained(model_name)
            )
            print("GPT-2 detector loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading GPT-2 detector: {e}")
            return False

    # Commented out ai_classifier as it was not used in the original main logic
    # def load_ai_text_classifier(self):
    #     """Load general AI text classifier"""
    #     try:
    #         model_name = "Hello-SimpleAI/chatgpt-detector-roberta-chinese"
    #         print(f"Loading {model_name}...")
    #         self.tokenizers["ai_classifier"] = AutoTokenizer.from_pretrained(model_name)
    #         self.models["ai_classifier"] = (
    #             AutoModelForSequenceClassification.from_pretrained(model_name)
    #         )
    #         print("AI text classifier loaded successfully!")
    #         return True
    #     except Exception as e:
    #         print(f"Error loading AI text classifier: {e}")
    #         return False

    def detect_with_model(self, text: str, model_name: str) -> float:
        """Detect AI-generated content using a specific model"""
        if model_name not in self.models:
            print(f"Model {model_name} not loaded.")
            return -1

        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Assuming class 1 is AI, class 0 is Human. This might vary per model.
            # For roberta-base-openai-detector: 0 -> Real, 1 -> Fake (AI)
            # For Hello-SimpleAI/chatgpt-detector-roberta: 0 -> Human, 1 -> ChatGPT
            ai_probability = predictions[0][1].item()

            return ai_probability

        except Exception as e:
            print(f"Error during detection with {model_name}: {e}")
            return -1

    def chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """Split text into chunks for better analysis"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 0:
                chunks.append(chunk)
        return chunks

    def analyze_text_content(
        self, text_content: str, source_description: str = "Input Text"
    ) -> Dict:
        """Analyze text content (string) for AI-generated content"""
        self._ensure_models_loaded()

        if not text_content.strip():
            return {"error": "Input text is empty"}

        if not self.models:  # No models could be loaded
            return {
                "error": "No AI detection models are available. Please check loading errors."
            }

        results = {
            "source_description": source_description,
            "total_characters": len(text_content),
            "total_words": len(text_content.split()),
            "models_used": list(self.models.keys()),
            "full_text_analysis": {},
            "chunk_analysis": {},
            "overall_ai_probability": 0,
        }

        # Analyze full text (or a representative part if very long)
        text_to_analyze_full = text_content[:2048]  # Limit for some models
        for model_name in self.models:
            ai_prob = self.detect_with_model(text_to_analyze_full, model_name)
            if ai_prob >= 0:
                results["full_text_analysis"][model_name] = {
                    "ai_probability": ai_prob,
                    "ai_percentage": f"{ai_prob * 100:.2f}%",
                }

        # Analyze chunks
        chunks = self.chunk_text(text_content)
        chunk_results = []

        for i, chunk in enumerate(chunks):
            chunk_scores = {}
            for model_name in self.models:
                ai_prob = self.detect_with_model(chunk, model_name)
                if ai_prob >= 0:
                    chunk_scores[model_name] = ai_prob

            if chunk_scores:
                avg_score = np.mean(list(chunk_scores.values())) if chunk_scores else 0
                chunk_results.append(
                    {
                        "chunk_id": i + 1,
                        "chunk_text_preview": chunk[:100] + "...",
                        "scores": chunk_scores,
                        "average_ai_probability": avg_score,
                    }
                )

        results["chunk_analysis"]["total_chunks"] = len(chunks)
        results["chunk_analysis"]["chunks"] = chunk_results

        all_scores = []
        for analysis in results["full_text_analysis"].values():
            all_scores.append(analysis["ai_probability"])
        for chunk_data in chunk_results:
            all_scores.append(chunk_data["average_ai_probability"])

        if all_scores:
            overall_probability = np.mean(all_scores)
            results["overall_ai_probability"] = overall_probability
            results["overall_ai_percentage"] = f"{overall_probability * 100:.2f}%"
            if overall_probability < 0.3:
                results["classification"] = "Likely Human-written"
            elif overall_probability < 0.7:
                results["classification"] = "Mixed/Uncertain"
            else:
                results["classification"] = "Likely AI-generated"
        else:
            results["classification"] = "Could not determine (no scores)"

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a readable report from analysis results"""
        if "error" in results:
            return f"Error: {results['error']}"

        report = f"""
AI Text Detection Report
========================

Source: {results.get('source_description', 'N/A')} 
Total Words: {results.get('total_words', 'N/A')}
Total Characters: {results.get('total_characters', 'N/A')}

Overall Results:
---------------
AI Probability: {results.get('overall_ai_percentage', 'N/A')}
Classification: {results.get('classification', 'N/A')}

Models Used: {results.get('models_used', [])}

Model-Specific Full Text Analysis:
-----------------------------------"""

        if results.get("full_text_analysis"):
            for model, scores in results["full_text_analysis"].items():
                report += f"\n  {model}: {scores['ai_percentage']} AI probability"
        else:
            report += "\n  No full text analysis available."

        report += f"""

Chunk Analysis:
---------------
Total Chunks Analyzed: {results.get('chunk_analysis', {}).get('total_chunks', 'N/A')}
"""
        if results.get("chunk_analysis", {}).get("chunks"):
            sorted_chunks = sorted(
                results["chunk_analysis"]["chunks"],
                key=lambda x: x.get("average_ai_probability", 0),
                reverse=True,
            )[:5]

            report += "\nTop 5 Most AI-like Chunks (if available):\n"
            if sorted_chunks:
                for chunk in sorted_chunks:
                    report += f"\n  Chunk {chunk['chunk_id']}: {chunk.get('average_ai_probability', 0)*100:.2f}% AI probability"
                    report += f"\n  Preview: {chunk['chunk_text_preview']}\n"
            else:
                report += "  No chunks to display.\n"
        else:
            report += "  No chunk analysis performed or available.\n"
        return report


# Alternative implementation using GPT-2 perplexity-based detection
class PerplexityBasedDetector:
    def __init__(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print("Loading GPT-2 for perplexity-based detection...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text using GPT-2"""
        encodings = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()

    def detect_ai_by_perplexity(self, text: str) -> Dict:
        """Detect AI text based on perplexity (lower perplexity suggests AI)"""
        perplexity = self.calculate_perplexity(text)

        # These thresholds are approximate and may need tuning
        if perplexity < 20:
            ai_probability = 0.9
            classification = "Very likely AI-generated"
        elif perplexity < 50:
            ai_probability = 0.7
            classification = "Likely AI-generated"
        elif perplexity < 100:
            ai_probability = 0.5
            classification = "Possibly AI-generated"
        elif perplexity < 200:
            ai_probability = 0.3
            classification = "Likely human-written"
        else:
            ai_probability = 0.1
            classification = "Very likely human-written"

        return {
            "perplexity": perplexity,
            "ai_probability": ai_probability,
            "ai_percentage": f"{ai_probability * 100:.2f}%",
            "classification": classification,
        }


# --- CLI Integration ---
def _process_func(
    detector: AITextDetector, text_content: str, extra_kwargs: Dict
) -> str:
    source_desc = extra_kwargs.get(
        "source", "Input Text"
    )  # Get source from kwargs from generic_cli
    analysis_results = detector.analyze_text_content(
        text_content, source_description=source_desc
    )
    report = detector.generate_report(analysis_results)

    # Optionally save detailed results to JSON if a flag is passed or by default
    # For simplicity, this is commented out for now but can be added as an extra_arg
    # if extra_kwargs.get("save_json_report", False):
    #     output_filename = f"ai_detection_results_{source_desc.replace(' ', '_').lower()}.json"
    #     try:
    #         with open(output_filename, "w") as f:
    #             json.dump(analysis_results, f, indent=2)
    #         report += f"\n\nDetailed results also saved to '{output_filename}'"
    #     except Exception as e:
    #         report += f"\n\nError saving JSON report: {e}"
    return report


if __name__ == "__main__":
    import sys

    current_dir = os.path.dirname(__file__)
    utils_path = os.path.join(current_dir, "humanize_cli_utils.py")
    spec = importlib.util.spec_from_file_location("humanize_cli_utils", utils_path)

    if spec is None:
        print(
            f"Error: Could not load humanize_cli_utils.py from {utils_path}. Ensure it exists."
        )
        sys.exit(1)

    cli_utils = importlib.util.module_from_spec(spec)
    sys.modules["humanize_cli_utils"] = cli_utils
    spec.loader.exec_module(cli_utils)

    cli_utils.generic_main_cli(
        description="Advanced AI Text Detector (Model-based)",
        humanizer_class=AITextDetector,  # This is AITextDetector from this file
        process_func=_process_func,
        extra_args=None,  # Can add args for model selection, perplexity, json output etc.
        extra_setup=lambda args: {
            "verbose": args.verbose,
            "source": getattr(
                args, "file", getattr(args, "folder", "command-line_text")
            ),
        },
    )
