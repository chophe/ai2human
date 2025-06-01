import re
import nltk
from collections import Counter
import statistics
import os
import typer
from typing import Dict, Any, Optional, Annotated

# Import generic_main_cli from the local utils file
from .humanize_cli_utils import generic_main_cli

# Download required NLTK data (run once)
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
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except nltk.downloader.DownloadError:
    nltk.download("averaged_perceptron_tagger", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class AITextDetector:
    def __init__(self, model_name: Optional[str] = None):
        self.stop_words = set(stopwords.words("english"))
        self.model_name = model_name

        # Common AI patterns and phrases
        self.ai_indicators = [
            "it's important to note",
            "it's worth noting",
            "in conclusion",
            "furthermore",
            "moreover",
            "however",
            "nevertheless",
            "on the other hand",
            "in summary",
            "to summarize",
            "it's essential",
            "it's crucial",
            "as an ai",
            "as a large language model",
            "i don't have personal",
            "i cannot",
            "i'm unable to",
            "it's important to remember",
            "in today's world",
            "in the modern era",
            "plays a crucial role",
            "significant impact",
            "various aspects",
            "comprehensive understanding",
            "it's vital",
            "key factors",
            "multiple factors",
            "delve into",
            "explore the",
            "the realm of",
        ]

    def calculate_sentence_uniformity(self, text):
        """Calculate how uniform sentence lengths are (AI tends to be more uniform)"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0

        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_length = statistics.mean(sentence_lengths)
        if len(sentence_lengths) <= 1:
            std_dev = 0.0
        else:
            try:
                std_dev = statistics.stdev(sentence_lengths)
            except statistics.StatisticsError:
                std_dev = 0.0

        cv = (std_dev / avg_length) if avg_length > 0 else 0
        uniformity_score = max(0.0, 1.0 - cv) * 100.0
        return uniformity_score

    def calculate_vocabulary_diversity(self, text):
        """Calculate vocabulary diversity (AI often has less diverse vocabulary)"""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]

        if not words:
            return 0.0

        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        ai_score = (1.0 - diversity_ratio) * 100.0
        return ai_score

    def check_ai_phrases(self, text):
        """Check for common AI-generated phrases"""
        text_lower = text.lower()
        found_phrases_count = 0

        for phrase in self.ai_indicators:
            if phrase in text_lower:
                found_phrases_count += 1

        if not self.ai_indicators:
            return 0.0

        ai_phrase_score = (found_phrases_count / len(self.ai_indicators)) * 100.0
        return ai_phrase_score

    def analyze_punctuation_patterns(self, text):
        """Analyze punctuation patterns (AI tends to use more formal punctuation)"""
        sentences = sent_tokenize(text)
        formal_endings = 0
        total_sentences = len(sentences)

        if total_sentences == 0:
            return 0.0

        for sentence in sentences:
            if sentence.strip().endswith("."):
                formal_endings += 1

        formality_score = (formal_endings / total_sentences) * 100.0
        return formality_score

    def calculate_repetition_score(self, text):
        """Check for repetitive patterns in the text"""
        words = word_tokenize(text.lower())
        words = [
            w for w in words if w.isalpha() and len(w) > 3 and w not in self.stop_words
        ]

        if len(words) < 20:
            return 0.0

        word_freq = Counter(words)
        excess_repetitions = sum(count - 1 for count in word_freq.values() if count > 1)
        repetition_score = (
            (excess_repetitions / len(words)) * 100.0 if len(words) > 0 else 0.0
        )
        return min(repetition_score, 100.0)

    def detect_ai_percentage(self, text: str) -> Dict[str, Any]:
        """Main function to detect AI-generated content percentage"""
        if not text or len(text.strip()) < 100:
            return {
                "ai_probability": 0.0,
                "details": {
                    "sentence_uniformity": 0.0,
                    "vocabulary_pattern": 0.0,
                    "ai_phrases_found": 0.0,
                    "punctuation_formality": 0.0,
                    "repetition_pattern": 0.0,
                },
                "notes": "Text too short or empty for reliable analysis (minimum 100 characters). Assigning 0% AI probability.",
            }

        uniformity_score = self.calculate_sentence_uniformity(text)
        vocabulary_score = self.calculate_vocabulary_diversity(text)
        ai_phrase_score = self.check_ai_phrases(text)
        punctuation_score = self.analyze_punctuation_patterns(text)
        repetition_score = self.calculate_repetition_score(text)

        weights = {
            "uniformity": 0.20,
            "vocabulary": 0.20,
            "ai_phrases": 0.30,
            "punctuation": 0.15,
            "repetition": 0.15,
        }

        ai_probability = (
            uniformity_score * weights["uniformity"]
            + vocabulary_score * weights["vocabulary"]
            + ai_phrase_score * weights["ai_phrases"]
            + punctuation_score * weights["punctuation"]
            + repetition_score * weights["repetition"]
        )
        ai_probability = max(0.0, min(100.0, ai_probability))

        return {
            "ai_probability": round(ai_probability, 2),
            "details": {
                "sentence_uniformity": round(uniformity_score, 2),
                "vocabulary_pattern": round(vocabulary_score, 2),
                "ai_phrases_found": round(ai_phrase_score, 2),
                "punctuation_formality": round(punctuation_score, 2),
                "repetition_pattern": round(repetition_score, 2),
            },
            "notes": "Analysis complete.",
        }


def _format_results(results: dict, source_name: str = "Text") -> str:
    output_lines = []
    output_lines.append(
        typer.style(
            f"AI Detection Analysis for: {source_name}",
            fg=typer.colors.BRIGHT_BLUE,
            bold=True,
        )
    )
    output_lines.append(typer.style(f"{'='*50}", fg=typer.colors.BLUE))

    if "error" in results:
        return typer.style(
            f"Error analyzing {source_name}: {results['error']}", fg=typer.colors.RED
        )
    if "notes" in results and "Assigning 0% AI probability" in results["notes"]:
        output_lines.append(
            typer.style(
                f"Overall AI Probability: {results['ai_probability']}%",
                fg=typer.colors.YELLOW,
            )
        )
        output_lines.append(
            typer.style(f"Note: {results['notes']}", fg=typer.colors.YELLOW)
        )
        return "\n".join(output_lines)

    output_lines.append(
        typer.style(
            f"Overall AI Probability: {results['ai_probability']}%",
            fg=typer.colors.CYAN,
            bold=True,
        )
    )
    output_lines.append("\nDetailed Scores (higher indicates more AI-like traits):")
    details = results.get("details", {})
    output_lines.append(
        f"  - Sentence Uniformity:   {details.get('sentence_uniformity', 0.0)}%"
    )
    output_lines.append(
        f"  - Vocabulary Diversity:  {details.get('vocabulary_pattern', 0.0)}% (Higher score = LESS diverse, more AI-like)"
    )
    output_lines.append(
        f"  - AI Phrase Indicators:  {details.get('ai_phrases_found', 0.0)}%"
    )
    output_lines.append(
        f"  - Punctuation Formality: {details.get('punctuation_formality', 0.0)}%"
    )
    output_lines.append(
        f"  - Repetition Patterns:   {details.get('repetition_pattern', 0.0)}%"
    )

    ai_prob = results.get("ai_probability", 0.0)
    interpretation_color = typer.colors.GREEN
    if ai_prob >= 70:
        interpretation = (
            "This text shows strong AI characteristics and is LIKELY AI-generated."
        )
        interpretation_color = typer.colors.RED
    elif ai_prob >= 50:
        interpretation = "This text shows moderate AI characteristics. It MAY be AI-assisted or edited."
        interpretation_color = typer.colors.YELLOW
    elif ai_prob >= 30:
        interpretation = "This text shows some AI-like characteristics but is LIKELY human-written with possible minor AI influence."
        interpretation_color = typer.colors.BRIGHT_GREEN
    else:
        interpretation = "This text appears to be PRIMARILY human-written."

    output_lines.append("\nInterpretation:")
    output_lines.append(typer.style(f"  {interpretation}", fg=interpretation_color))
    if "notes" in results and results["notes"] != "Analysis complete.":
        output_lines.append(
            typer.style(f"\nNote: {results['notes']}", fg=typer.colors.DIM_WHITE)
        )

    return "\n".join(output_lines)


def _process_func(
    detector_instance: AITextDetector, text: str, extra_kwargs: Dict[str, Any]
) -> str:
    source_name = extra_kwargs.get("source", "Input Text")
    if extra_kwargs.get("verbose", False):
        typer.echo(
            f"AI Detector received text from: {source_name}", color=typer.colors.MAGENTA
        )

    results = detector_instance.detect_ai_percentage(text)
    return _format_results(results, source_name)


def _detector_extra_setup(cli_args: Dict[str, Any]) -> Dict[str, Any]:
    return {"verbose": cli_args.get("verbose", True), "model": cli_args.get("model")}


app = typer.Typer(
    name="ai-detector",
    help="Rule-based AI Text Detector: Analyzes text for AI-like characteristics.",
    add_completion=False,
    no_args_is_help=True,
)

ai_detector_command_func = generic_main_cli(
    humanizer_class=AITextDetector,
    process_func=_process_func,
    extra_setup=_detector_extra_setup,
)


@app.command(
    help="Detects AI-written text from a string, file, or folder.", no_args_is_help=True
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
            help="Enable verbose output (provides more details during processing)."
        ),
    ] = True,
    model: Annotated[
        Optional[str],
        typer.Option(
            help="Specify a model name (currently not used by this detector but available for future use)."
        ),
    ] = None,
):
    """
    Rule-based AI Text Detector.
    Analyzes text input from direct string, file, or folder to estimate AI generation probability.
    """
    ai_detector_command_func(
        text=text,
        file=file,
        folder=folder,
        verbose=verbose,
        model=model,
    )


if __name__ == "__main__":
    app()
