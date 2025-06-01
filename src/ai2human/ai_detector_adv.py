import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Any, Optional, Annotated
import warnings
import os
import typer

# Import generic_main_cli from the local utils file
from .humanize_cli_utils import generic_main_cli

warnings.filterwarnings("ignore")

# Ensure NLTK data is available for chunking/sentence tokenization if used by helper funcs (though not directly in AITextDetector)
# This is good practice if any part of the code (even indirectly) might need it.
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt", quiet=True)


class AITextDetector:
    def __init__(self, model_name: Optional[str] = None, verbose: bool = True):
        """Initialize AI text detector with multiple detection methods"""
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self._models_loaded_flags = {
            "roberta": False,
            "gpt2_detector": False,
        }
        self.verbose = verbose
        self.cli_model_name = model_name

    def _ensure_models_loaded(self):
        """Load all defined models if they haven't been loaded yet."""
        if not self._models_loaded_flags["roberta"]:
            if self.load_roberta_detector():
                self._models_loaded_flags["roberta"] = True
        if not self._models_loaded_flags["gpt2_detector"]:
            if self.load_gpt2_detector():
                self._models_loaded_flags["gpt2_detector"] = True

    def load_roberta_detector(self) -> bool:
        """Load RoBERTa-based AI detector (OpenAI detector)"""
        try:
            model_name = "roberta-base-openai-detector"
            if self.verbose:
                typer.echo(f"Loading {model_name}...", color=typer.colors.BLUE)
            self.tokenizers["roberta"] = AutoTokenizer.from_pretrained(model_name)
            self.models["roberta"] = AutoModelForSequenceClassification.from_pretrained(
                model_name
            )
            if self.verbose:
                typer.echo(
                    "RoBERTa detector loaded successfully!", color=typer.colors.GREEN
                )
            return True
        except Exception as e:
            typer.echo(
                f"Error loading RoBERTa detector ({model_name}): {e}",
                err=True,
                color=typer.colors.RED,
            )
            return False

    def load_gpt2_detector(self) -> bool:
        """Load GPT-2 output detector"""
        try:
            model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
            if self.verbose:
                typer.echo(f"Loading {model_name}...", color=typer.colors.BLUE)
            self.tokenizers["gpt2_detector"] = AutoTokenizer.from_pretrained(model_name)
            self.models["gpt2_detector"] = (
                AutoModelForSequenceClassification.from_pretrained(model_name)
            )
            if self.verbose:
                typer.echo(
                    "GPT-2 detector loaded successfully!", color=typer.colors.GREEN
                )
            return True
        except Exception as e:
            typer.echo(
                f"Error loading GPT-2 detector ({model_name}): {e}",
                err=True,
                color=typer.colors.RED,
            )
            return False

    def detect_with_model(self, text: str, model_name: str) -> float:
        """Detect AI-generated content using a specific model"""
        if model_name not in self.models or model_name not in self.tokenizers:
            if self.verbose:
                typer.echo(
                    f"Model or tokenizer for {model_name} not loaded.",
                    err=True,
                    color=typer.colors.YELLOW,
                )
            return -1.0

        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            ai_probability = predictions[0][1].item()
            return ai_probability
        except Exception as e:
            typer.echo(
                f"Error during detection with {model_name}: {e}",
                err=True,
                color=typer.colors.RED,
            )
            return -1.0

    def chunk_text(
        self, text: str, chunk_size: int = 400, overlap: int = 50
    ) -> List[str]:
        """Split text into potentially overlapping chunks for better analysis."""
        if not text.strip():
            return []
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            words = text.split()
            sentences = [
                " ".join(words[i : i + chunk_size // 10])
                for i in range(0, len(words), chunk_size // 10)
            ]

        chunks = []
        current_chunk_words = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            if not sentence_words:
                continue

            if (
                current_word_count + len(sentence_words) > chunk_size
                and current_chunk_words
            ):
                chunks.append(" ".join(current_chunk_words))
                overlap_words_count = min(
                    len(current_chunk_words),
                    overlap
                    // (
                        len(sentence_words) / len(sentence_words)
                        if sentence_words
                        else 1
                    ),
                )
                current_chunk_words = (
                    current_chunk_words[-int(overlap_words_count) :]
                    if overlap_words_count > 0
                    else []
                )
                current_word_count = len(current_chunk_words)

            current_chunk_words.extend(sentence_words)
            current_word_count += len(sentence_words)

        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))

        return [c for c in chunks if c.strip()]

    def analyze_text_content(
        self, text_content: str, source_description: str = "Input Text"
    ) -> Dict:
        """Analyze text content (string) for AI-generated content"""
        self._ensure_models_loaded()

        if not text_content or not text_content.strip():
            return {"error": "Input text is empty or contains only whitespace."}

        if not self.models:
            return {
                "error": "No AI detection models are loaded or available. Cannot perform analysis."
            }

        results: Dict[str, Any] = {
            "source_description": source_description,
            "total_characters": len(text_content),
            "total_words": len(text_content.split()),
            "models_attempted_load": list(self._models_loaded_flags.keys()),
            "models_successfully_loaded": list(self.models.keys()),
            "full_text_analysis": {},
            "chunk_analysis": {"total_chunks": 0, "chunks": []},
            "overall_ai_probability": 0.0,
            "classification": "Error in analysis or no models loaded",
        }
        if not results["models_successfully_loaded"]:
            results["error"] = "No models were successfully loaded. Check logs."
            return results

        text_to_analyze_full = text_content[:2500]

        full_text_scores_by_model: Dict[str, float] = {}
        for model_name in self.models.keys():
            ai_prob = self.detect_with_model(text_to_analyze_full, model_name)
            if ai_prob >= 0:
                full_text_scores_by_model[model_name] = ai_prob
                results["full_text_analysis"][model_name] = {
                    "ai_probability": ai_prob,
                    "ai_percentage": f"{ai_prob * 100:.2f}%",
                }

        chunks = self.chunk_text(text_content, chunk_size=300, overlap=50)
        chunk_results_list: List[Dict[str, Any]] = []
        results["chunk_analysis"]["total_chunks"] = len(chunks)

        for i, chunk_text_item in enumerate(chunks):
            if not chunk_text_item.strip():
                continue
            chunk_model_scores: Dict[str, float] = {}
            for model_name in self.models.keys():
                ai_prob_chunk = self.detect_with_model(chunk_text_item, model_name)
                if ai_prob_chunk >= 0:
                    chunk_model_scores[model_name] = ai_prob_chunk

            if chunk_model_scores:
                avg_chunk_score = np.mean(list(chunk_model_scores.values()))
                chunk_results_list.append(
                    {
                        "chunk_id": i + 1,
                        "chunk_text_preview": chunk_text_item[:100] + "...",
                        "scores_by_model": {
                            k: f"{v*100:.2f}%" for k, v in chunk_model_scores.items()
                        },
                        "average_ai_probability": avg_chunk_score,
                        "average_ai_percentage": f"{avg_chunk_score*100:.2f}%",
                    }
                )
        results["chunk_analysis"]["chunks"] = chunk_results_list

        all_valid_scores: List[float] = []
        all_valid_scores.extend(full_text_scores_by_model.values())
        for chunk_data in chunk_results_list:
            all_valid_scores.append(chunk_data["average_ai_probability"])

        if all_valid_scores:
            overall_probability = np.mean(all_valid_scores)
            results["overall_ai_probability"] = overall_probability
            results["overall_ai_percentage"] = f"{overall_probability * 100:.2f}%"
            if overall_probability < 0.3:
                results["classification"] = "Likely Human-written"
            elif overall_probability < 0.7:
                results["classification"] = "Potentially AI-assisted or Mixed"
            else:
                results["classification"] = "Likely AI-generated"
        else:
            results["overall_ai_probability"] = 0.0
            results["overall_ai_percentage"] = "0.00%"
            results["classification"] = (
                "Could not determine (no valid scores from models)"
            )
            if not self.models:
                results["classification"] = "Error: No models loaded."

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a readable report from analysis results using Typer styling."""
        if "error" in results and results["error"]:
            return typer.style(
                f"Error: {results['error']}", fg=typer.colors.RED, bold=True
            )

        report_parts: List[str] = []
        source_desc = results.get("source_description", "N/A")
        report_parts.append(
            typer.style(
                "AI Text Detection Report (Advanced)",
                fg=typer.colors.BRIGHT_BLUE,
                bold=True,
            )
        )
        report_parts.append(
            typer.style("===================================", fg=typer.colors.BLUE)
        )
        report_parts.append(f"Source: {source_desc}")
        report_parts.append(
            f"Total Words: {results.get('total_words', 'N/A')}, Total Characters: {results.get('total_characters', 'N/A')}"
        )

        loaded_models_str = (
            ", ".join(results.get("models_successfully_loaded", [])) or "None"
        )
        report_parts.append(f"Models Successfully Loaded: {loaded_models_str}")
        attempted_models_str = (
            ", ".join(results.get("models_attempted_load", [])) or "None"
        )
        if loaded_models_str != attempted_models_str:
            report_parts.append(
                typer.style(
                    f"Models Attempted to Load: {attempted_models_str}",
                    fg=typer.colors.YELLOW,
                )
            )

        report_parts.append(typer.style("\nOverall Results:", bold=True))
        report_parts.append("---------------")
        overall_perc = results.get("overall_ai_percentage", "N/A")
        classification = results.get("classification", "N/A")
        prob_color = typer.colors.GREEN
        if results.get("overall_ai_probability", 0) >= 0.7:
            prob_color = typer.colors.RED
        elif results.get("overall_ai_probability", 0) >= 0.3:
            prob_color = typer.colors.YELLOW
        report_parts.append(
            f"AI Probability: {typer.style(overall_perc, fg=prob_color, bold=True)}"
        )
        report_parts.append(
            f"Classification: {typer.style(classification, fg=prob_color)}"
        )

        report_parts.append(
            typer.style(
                "\nModel-Specific Full Text Analysis (first ~500 words):", bold=True
            )
        )
        report_parts.append("-------------------------------------------------")
        if results.get("full_text_analysis"):
            for model, scores in results["full_text_analysis"].items():
                model_prob_color = typer.colors.GREEN
                if scores.get("ai_probability", 0) >= 0.7:
                    model_prob_color = typer.colors.RED
                elif scores.get("ai_probability", 0) >= 0.3:
                    model_prob_color = typer.colors.YELLOW
                report_parts.append(
                    f"  {model}: {typer.style(scores['ai_percentage'], fg=model_prob_color)} AI probability"
                )
        else:
            report_parts.append("  No full text analysis available or models failed.")

        report_parts.append(
            typer.style(
                "\nChunk Analysis (Top 5 most AI-like chunks, if any):", bold=True
            )
        )
        report_parts.append("----------------------------------------------------")
        chunk_analysis_data = results.get("chunk_analysis", {})
        report_parts.append(
            f"Total Chunks Analyzed: {chunk_analysis_data.get('total_chunks', 'N/A')}"
        )

        sorted_chunks = sorted(
            chunk_analysis_data.get("chunks", []),
            key=lambda x: x.get("average_ai_probability", 0.0),
            reverse=True,
        )[:5]

        if sorted_chunks:
            for chunk_item in sorted_chunks:
                avg_prob = chunk_item.get("average_ai_probability", 0.0)
                avg_perc = chunk_item.get("average_ai_percentage", "0.00%")
                chunk_color = typer.colors.GREEN
                if avg_prob >= 0.7:
                    chunk_color = typer.colors.RED
                elif avg_prob >= 0.3:
                    chunk_color = typer.colors.YELLOW

                report_parts.append(
                    f"\n  Chunk {chunk_item['chunk_id']}: {typer.style(avg_perc, fg=chunk_color)} avg AI probability"
                )
                report_parts.append(f"  Preview: {chunk_item['chunk_text_preview']}")
        elif chunk_analysis_data.get("total_chunks", 0) > 0:
            report_parts.append(
                "  All chunks had low AI probability or analysis issues."
            )
        else:
            report_parts.append("  No chunks analyzed or available.")

        return "\n".join(report_parts)


class PerplexityBasedDetector:
    def __init__(self, model_id="gpt2"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_perplexity(self, text: str) -> float:
        if not text.strip():
            return float("inf")
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024, padding=True
        )
        input_ids = inputs.input_ids

        try:
            with torch.no_grad():
                outputs = self.model(**inputs, labels=input_ids.clone())
                loss = outputs.loss
            return loss.item() * 10
        except Exception as e:
            if hasattr(self, "verbose") and self.verbose:
                typer.echo(
                    f"Error calculating pseudo-perplexity for '{text[:50]}...': {e}",
                    color=typer.colors.YELLOW,
                )
            return float("inf")

    def detect_ai_by_perplexity(self, text: str, threshold: float = 100.0) -> Dict:
        perplexity = self.calculate_perplexity(text)
        is_ai_likely = perplexity < threshold
        return {
            "text_preview": text[:100] + "...",
            "pseudo_perplexity_score": round(perplexity, 2),
            "is_ai_likely_by_perplexity": is_ai_likely,
            "threshold_used": threshold,
            "notes": "Lower score (pseudo-perplexity) might indicate AI for some generative models. This is experimental.",
        }


def _adv_detector_process_func(
    detector_instance: AITextDetector, text_content: str, extra_kwargs: Dict[str, Any]
) -> str:
    source_name = extra_kwargs.get("source", "Input Text")
    verbose_mode = extra_kwargs.get("verbose", True)

    if (
        hasattr(detector_instance, "verbose")
        and detector_instance.verbose != verbose_mode
    ):
        detector_instance.verbose = verbose_mode
        if verbose_mode:
            typer.echo(
                f"Advanced AI Detector verbose mode: {detector_instance.verbose}",
                color=typer.colors.MAGENTA,
            )

    if verbose_mode:
        typer.echo(
            f"Advanced AI Detector received text from: {source_name}",
            color=typer.colors.MAGENTA,
        )

    analysis_results = detector_instance.analyze_text_content(
        text_content, source_description=source_name
    )
    return detector_instance.generate_report(analysis_results)


def _adv_detector_extra_setup(cli_args: Dict[str, Any]) -> Dict[str, Any]:
    return {"verbose": cli_args.get("verbose", True), "model": cli_args.get("model")}


app = typer.Typer(
    name="ai-detector-adv",
    help="Advanced AI Text Detector using multiple HuggingFace models.",
    add_completion=False,
    no_args_is_help=True,
)

adv_detector_command_func = generic_main_cli(
    humanizer_class=AITextDetector,
    process_func=_adv_detector_process_func,
    extra_setup=_adv_detector_extra_setup,
)


@app.command(
    help="Detects AI-written text using multiple HuggingFace models (RoBERTa, GPT-2 detector).",
    no_args_is_help=True,
)
def main(
    text: Annotated[
        Optional[str],
        typer.Option(help="Text to analyze directly.", rich_help_panel="Input Options"),
    ] = None,
    file: Annotated[
        Optional[str],
        typer.Option(
            help="Path to a text file to analyze.", rich_help_panel="Input Options"
        ),
    ] = None,
    folder: Annotated[
        Optional[str],
        typer.Option(
            help="Path to a folder with .txt/.md files.",
            rich_help_panel="Input Options",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            help="Enable verbose output (model loading, chunk processing etc.)."
        ),
    ] = True,
    model: Annotated[
        Optional[str],
        typer.Option(
            help="Specify a primary model name (e.g., 'roberta', 'gpt2_detector'). Currently used to pass to detector; future use for selective loading.",
            rich_help_panel="Advanced Options",
        ),
    ] = None,
):
    """
    Advanced AI Text Detector.
    Analyzes text from string, file, or folder using multiple HuggingFace models.
    Provides a detailed report including overall AI probability and chunk-level analysis.
    """
    adv_detector_command_func(
        text=text,
        file=file,
        folder=folder,
        verbose=verbose,
        model=model,
    )


if __name__ == "__main__":
    app()
