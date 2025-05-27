import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import json
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


class AITextDetector:
    def __init__(self):
        """Initialize AI text detector with multiple detection methods"""
        self.models = {}
        self.tokenizers = {}

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

    def load_ai_text_classifier(self):
        """Load general AI text classifier"""
        try:
            model_name = "Hello-SimpleAI/chatgpt-detector-roberta-chinese"
            print(f"Loading {model_name}...")
            self.tokenizers["ai_classifier"] = AutoTokenizer.from_pretrained(model_name)
            self.models["ai_classifier"] = (
                AutoModelForSequenceClassification.from_pretrained(model_name)
            )
            print("AI text classifier loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading AI text classifier: {e}")
            return False

    def detect_with_model(self, text: str, model_name: str) -> float:
        """Detect AI-generated content using a specific model"""
        if model_name not in self.models:
            return -1

        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]

            # Tokenize input
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )

            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get AI probability (usually class 1 or 0 depending on model)
            ai_probability = predictions[0][1].item()  # Adjust index if needed

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

    def analyze_text_file(self, file_path: str) -> Dict:
        """Analyze a text file for AI-generated content"""
        try:
            # Read file
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                return {"error": "File is empty"}

            # Load models if not already loaded
            models_loaded = []
            if not self.models:
                if self.load_roberta_detector():
                    models_loaded.append("roberta")
                if self.load_gpt2_detector():
                    models_loaded.append("gpt2_detector")
                # You can add more models here

            # Analyze full text and chunks
            results = {
                "file_path": file_path,
                "total_characters": len(text),
                "total_words": len(text.split()),
                "models_used": models_loaded,
                "full_text_analysis": {},
                "chunk_analysis": {},
                "overall_ai_probability": 0,
            }

            # Analyze full text
            for model_name in self.models:
                ai_prob = self.detect_with_model(
                    text[:2000], model_name
                )  # Limit for some models
                if ai_prob >= 0:
                    results["full_text_analysis"][model_name] = {
                        "ai_probability": ai_prob,
                        "ai_percentage": f"{ai_prob * 100:.2f}%",
                    }

            # Analyze chunks
            chunks = self.chunk_text(text)
            chunk_results = []

            for i, chunk in enumerate(chunks):
                chunk_scores = {}
                for model_name in self.models:
                    ai_prob = self.detect_with_model(chunk, model_name)
                    if ai_prob >= 0:
                        chunk_scores[model_name] = ai_prob

                if chunk_scores:
                    avg_score = np.mean(list(chunk_scores.values()))
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

            # Calculate overall AI probability
            all_scores = []
            for analysis in results["full_text_analysis"].values():
                all_scores.append(analysis["ai_probability"])

            for chunk in chunk_results:
                all_scores.append(chunk["average_ai_probability"])

            if all_scores:
                overall_probability = np.mean(all_scores)
                results["overall_ai_probability"] = overall_probability
                results["overall_ai_percentage"] = f"{overall_probability * 100:.2f}%"

                # Classification
                if overall_probability < 0.3:
                    results["classification"] = "Likely Human-written"
                elif overall_probability < 0.7:
                    results["classification"] = "Mixed/Uncertain"
                else:
                    results["classification"] = "Likely AI-generated"

            return results

        except Exception as e:
            return {"error": f"Error analyzing file: {str(e)}"}

    def generate_report(self, results: Dict) -> str:
        """Generate a readable report from analysis results"""
        if "error" in results:
            return f"Error: {results['error']}"

        report = f"""
AI Text Detection Report
========================

File: {results['file_path']}
Total Words: {results['total_words']}
Total Characters: {results['total_characters']}

Overall Results:
---------------
AI Probability: {results.get('overall_ai_percentage', 'N/A')}
Classification: {results.get('classification', 'N/A')}

Model-Specific Results:
----------------------"""

        for model, scores in results["full_text_analysis"].items():
            report += f"\n{model}: {scores['ai_percentage']} AI probability"

        report += f"""

Chunk Analysis:
--------------
Total Chunks Analyzed: {results['chunk_analysis']['total_chunks']}
"""

        # Show top 5 most AI-like chunks
        if results["chunk_analysis"]["chunks"]:
            sorted_chunks = sorted(
                results["chunk_analysis"]["chunks"],
                key=lambda x: x["average_ai_probability"],
                reverse=True,
            )[:5]

            report += "\nTop 5 Most AI-like Chunks:\n"
            for chunk in sorted_chunks:
                report += f"\nChunk {chunk['chunk_id']}: {chunk['average_ai_probability']*100:.2f}% AI probability"
                report += f"\nPreview: {chunk['chunk_text_preview']}\n"

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


# Main usage example
def main():
    # Initialize detector
    detector = AITextDetector()

    # Specify the file to analyze
    file_path = "sample_text.txt"  # Change this to your file path

    print("Starting AI text detection...")
    print("-" * 50)

    # Analyze the file
    results = detector.analyze_text_file(file_path)

    # Generate and print report
    report = detector.generate_report(results)
    print(report)

    # Save detailed results to JSON
    with open("ai_detection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDetailed results saved to 'ai_detection_results.json'")

    # Optional: Use perplexity-based detection
    print("\n" + "-" * 50)
    print("Running perplexity-based detection...")

    try:
        perp_detector = PerplexityBasedDetector()
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()[:1000]  # Use first 1000 chars for perplexity

        perp_results = perp_detector.detect_ai_by_perplexity(text)
        print(f"\nPerplexity-based results:")
        print(f"Perplexity: {perp_results['perplexity']:.2f}")
        print(f"AI Probability: {perp_results['ai_percentage']}")
        print(f"Classification: {perp_results['classification']}")
    except Exception as e:
        print(f"Perplexity detection error: {e}")


if __name__ == "__main__":
    main()
