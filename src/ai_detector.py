import re
import nltk
from collections import Counter
import statistics

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
        repetition_score = (repetitions / len(set(words))) * 100

        return repetition_score

    def detect_ai_percentage(self, text):
        """Main function to detect AI-generated content percentage"""
        if len(text.strip()) < 100:
            return {"error": "Text too short for reliable analysis"}

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


def analyze_file(filename):
    """Analyze a text file for AI-generated content"""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        detector = AITextDetector()
        results = detector.detect_ai_percentage(text)

        if "error" in results:
            print(f"Error: {results['error']}")
            return

        print(f"\n{'='*50}")
        print(f"AI Detection Analysis for: {filename}")
        print(f"{'='*50}")
        print(f"\nOverall AI Probability: {results['ai_probability']}%")
        print(f"\nDetailed Scores:")
        print(f"  - Sentence Uniformity: {results['details']['sentence_uniformity']}%")
        print(f"  - Vocabulary Patterns: {results['details']['vocabulary_pattern']}%")
        print(f"  - AI Phrase Indicators: {results['details']['ai_phrases_found']}%")
        print(
            f"  - Punctuation Formality: {results['details']['punctuation_formality']}%"
        )
        print(f"  - Repetition Patterns: {results['details']['repetition_pattern']}%")

        # Interpretation
        ai_prob = results["ai_probability"]
        print(f"\nInterpretation:")
        if ai_prob < 30:
            print("  This text appears to be primarily human-written.")
        elif ai_prob < 50:
            print(
                "  This text shows some AI-like characteristics but is likely human-written."
            )
        elif ai_prob < 70:
            print(
                "  This text shows moderate AI characteristics. It may be AI-assisted or edited."
            )
        else:
            print(
                "  This text shows strong AI characteristics and is likely AI-generated."
            )

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")


# Example usage
if __name__ == "__main__":
    # You can change this to your file path
    filename = input("Enter the path to your text file: ").strip()
    analyze_file(filename)

    # Optional: Analyze text directly
    print("\n" + "=" * 50)
    choice = input("\nWould you like to analyze text directly? (y/n): ").strip().lower()
    if choice == "y":
        print("Enter your text (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        text = "\n".join(lines)
        if text:
            detector = AITextDetector()
            results = detector.detect_ai_percentage(text)
            if "error" not in results:
                print(
                    f"\nDirect Text Analysis - AI Probability: {results['ai_probability']}%"
                )
