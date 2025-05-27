import re
import nltk
from collections import Counter
import statistics
import os
import importlib.util

# Download required NLTK data (run once)
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag


class AITextDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

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
        ]

    def calculate_sentence_uniformity(self, text):
        """Calculate how uniform sentence lengths are (AI tends to be more uniform)"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0

        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_length = statistics.mean(sentence_lengths)
        std_dev = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0

        # Lower coefficient of variation suggests more uniformity (AI-like)
        cv = (std_dev / avg_length) if avg_length > 0 else 0
        uniformity_score = max(0, 1 - cv) * 100

        return uniformity_score

    def calculate_vocabulary_diversity(self, text):
        """Calculate vocabulary diversity (AI often has less diverse vocabulary)"""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]

        if len(words) == 0:
            return 0

        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)

        # Lower diversity might indicate AI
        ai_score = (1 - diversity_ratio) * 100

        return ai_score

    def check_ai_phrases(self, text):
        """Check for common AI-generated phrases"""
        text_lower = text.lower()
        found_phrases = 0

        for phrase in self.ai_indicators:
            if phrase in text_lower:
                found_phrases += 1

        # Calculate percentage based on number of indicators found
        ai_phrase_score = (found_phrases / len(self.ai_indicators)) * 100

        return ai_phrase_score

    def analyze_punctuation_patterns(self, text):
        """Analyze punctuation patterns (AI tends to use more formal punctuation)"""
        sentences = sent_tokenize(text)

        formal_endings = 0
        total_sentences = len(sentences)

        if total_sentences == 0:
            return 0

        for sentence in sentences:
            if sentence.strip().endswith("."):
                formal_endings += 1

        formality_score = (formal_endings / total_sentences) * 100

        return formality_score

    def calculate_repetition_score(self, text):
        """Check for repetitive patterns in the text"""
        words = word_tokenize(text.lower())
        words = [
            w for w in words if w.isalpha() and len(w) > 4
        ]  # Focus on longer words

        if len(words) < 10:
            return 0

        word_freq = Counter(words)

        # Calculate repetition score
        repetitions = sum(1 for count in word_freq.values() if count > 2)
        repetition_score = (
            (repetitions / len(set(words))) * 100 if len(set(words)) > 0 else 0
        )

        return repetition_score

    def detect_ai_percentage(self, text):
        """Main function to detect AI-generated content percentage"""
        if len(text.strip()) < 100:
            return {
                "error": "Text too short for reliable analysis (minimum 100 characters)."
            }

        # Calculate individual scores
        uniformity_score = self.calculate_sentence_uniformity(text)
        vocabulary_score = self.calculate_vocabulary_diversity(text)
        ai_phrase_score = self.check_ai_phrases(text)
        punctuation_score = self.analyze_punctuation_patterns(text)
        repetition_score = self.calculate_repetition_score(text)

        # Weight the different factors
        weights = {
            "uniformity": 0.20,
            "vocabulary": 0.15,
            "ai_phrases": 0.35,
            "punctuation": 0.15,
            "repetition": 0.15,
        }

        # Calculate weighted average
        ai_probability = (
            uniformity_score * weights["uniformity"]
            + vocabulary_score * weights["vocabulary"]
            + ai_phrase_score * weights["ai_phrases"]
            + punctuation_score * weights["punctuation"]
            + repetition_score * weights["repetition"]
        )

        return {
            "ai_probability": round(ai_probability, 2),
            "details": {
                "sentence_uniformity": round(uniformity_score, 2),
                "vocabulary_pattern": round(vocabulary_score, 2),
                "ai_phrases_found": round(ai_phrase_score, 2),
                "punctuation_formality": round(punctuation_score, 2),
                "repetition_pattern": round(repetition_score, 2),
            },
        }


def _format_results(results: dict, source_name: str = "Text") -> str:
    if "error" in results:
        return f"Error analyzing {source_name}: {results['error']}"

    output_lines = []
    output_lines.append(f"AI Detection Analysis for: {source_name}")
    output_lines.append(f"{'='*50}")
    output_lines.append(f"Overall AI Probability: {results['ai_probability']}%")
    output_lines.append("\nDetailed Scores:")
    output_lines.append(
        f"  - Sentence Uniformity: {results['details']['sentence_uniformity']}%"
    )
    output_lines.append(
        f"  - Vocabulary Patterns: {results['details']['vocabulary_pattern']}%"
    )
    output_lines.append(
        f"  - AI Phrase Indicators: {results['details']['ai_phrases_found']}%"
    )
    output_lines.append(
        f"  - Punctuation Formality: {results['details']['punctuation_formality']}%"
    )
    output_lines.append(
        f"  - Repetition Patterns: {results['details']['repetition_pattern']}%"
    )

    ai_prob = results["ai_probability"]
    output_lines.append("\nInterpretation:")
    if ai_prob < 30:
        output_lines.append("  This text appears to be primarily human-written.")
    elif ai_prob < 50:
        output_lines.append(
            "  This text shows some AI-like characteristics but is likely human-written."
        )
    elif ai_prob < 70:
        output_lines.append(
            "  This text shows moderate AI characteristics. It may be AI-assisted or edited."
        )
    else:
        output_lines.append(
            "  This text shows strong AI characteristics and is likely AI-generated."
        )

    return "\n".join(output_lines)


def _process_func(detector, text, extra_kwargs):
    source_name = extra_kwargs.get(
        "source", "Input Text"
    )  # Get source from kwargs if available
    results = detector.detect_ai_percentage(text)
    return _format_results(results, source_name)


if __name__ == "__main__":
    import sys

    # Construct the path to humanize_cli_utils.py relative to this script
    # This assumes humanize_cli_utils.py is in the same directory (src)
    current_dir = os.path.dirname(__file__)
    utils_path = os.path.join(current_dir, "humanize_cli_utils.py")

    spec = importlib.util.spec_from_file_location("humanize_cli_utils", utils_path)
    if spec is None:
        print(
            f"Error: Could not load humanize_cli_utils.py from {utils_path}. Ensure the file exists."
        )
        sys.exit(1)

    cli_utils = importlib.util.module_from_spec(spec)
    sys.modules["humanize_cli_utils"] = (
        cli_utils  # Add to sys.modules before exec_module
    )
    spec.loader.exec_module(cli_utils)

    cli_utils.generic_main_cli(
        description="AI Text Detector (Heuristic-based)",
        humanizer_class=AITextDetector,  # Pass the class itself
        process_func=_process_func,
        extra_args=None,  # No extra arguments for this detector
        extra_setup=lambda args: {
            "verbose": args.verbose
        },  # Pass verbose to process_func if needed for source name extraction
    )
