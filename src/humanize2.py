import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from typing import List, Dict
import argparse
import glob

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()


class TextHumanizer:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4.1",
        api_base_url: str = None,
    ):
        """
        Initialize the TextHumanizer with OpenAI API key, model, and optional API base URL.
        Args:
            api_key: OpenAI API key (optional, will use env if not provided)
            model_name: Model to use (default: gpt-3.5-turbo)
            api_base_url: Base URL for OpenAI API (optional, will use env if not provided)
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment."
            )
        if api_base_url is None:
            api_base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = api_key
        llm_kwargs = {"temperature": 0.7, "model": model_name}
        if api_base_url:
            llm_kwargs["base_url"] = api_base_url
        self.llm = ChatOpenAI(openai_api_key=api_key, **llm_kwargs)
        self.iteration_history = []

    def create_humanization_prompts(self) -> List[Dict[str, str]]:
        prompts = [
            {
                "description": "Initial humanization - Add natural flow",
                "template": """
                Please rewrite the following text to make it sound more natural and human-like. 
                Focus on:
                - Using conversational tone
                - Adding natural transitions
                - Varying sentence structure
                
                Original text:
                {text}
                
                Humanized version:
                """,
            },
            {
                "description": "Add personality and emotion",
                "template": """
                Take this text and enhance it by:
                - Adding appropriate emotional nuances
                - Including personal touches where suitable
                - Making it more engaging and relatable
                
                Current text:
                {text}
                
                Enhanced version:
                """,
            },
            {
                "description": "Improve readability and flow",
                "template": """
                Refine this text further by:
                - Ensuring smooth flow between ideas
                - Using more natural word choices
                - Breaking up complex sentences if needed
                - Adding subtle colloquialisms where appropriate
                
                Current text:
                {text}
                
                Refined version:
                """,
            },
            {
                "description": "Final polish",
                "template": """
                Give this text a final polish to ensure it sounds completely natural:
                - Check for any remaining stiff or formal language
                - Ensure consistency in tone
                - Make any final adjustments for natural human expression
                
                Current text:
                {text}
                
                Final version:
                """,
            },
        ]
        return prompts

    def humanize_text(
        self, input_text: str, iterations: int = None, verbose: bool = True
    ) -> str:
        prompts = self.create_humanization_prompts()
        if iterations is None:
            iterations = len(prompts)
        else:
            iterations = min(iterations, len(prompts))
        current_text = input_text
        self.iteration_history = [
            {"iteration": 0, "text": input_text, "description": "Original text"}
        ]
        total_tokens = 0
        total_cost = 0
        for i in range(iterations):
            prompt_info = prompts[i]
            if verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {i+1}: {prompt_info['description']}")
                print(f"{'='*50}")
            prompt_template = PromptTemplate(
                input_variables=["text"], template=prompt_info["template"]
            )
            chain = prompt_template | self.llm
            with get_openai_callback() as cb:
                response = chain.invoke({"text": current_text})
                current_text = response.content
                total_tokens += cb.total_tokens
                total_cost += cb.total_cost
            self.iteration_history.append(
                {
                    "iteration": i + 1,
                    "text": current_text,
                    "description": prompt_info["description"],
                }
            )
            if verbose:
                print(f"Tokens used: {cb.total_tokens}")
                print(f"Cost: ${cb.total_cost:.4f}")
        if verbose:
            print(f"\n{'='*50}")
            print(f"Total tokens used: {total_tokens}")
            print(f"Total cost: ${total_cost:.4f}")
            print(f"{'='*50}")
        return current_text

    def get_iteration_history(self) -> List[Dict]:
        return self.iteration_history

    def compare_versions(
        self, iteration_a: int = 0, iteration_b: int = -1
    ) -> Dict[str, str]:
        if not self.iteration_history:
            return {"error": "No iteration history available"}

        text_a = self.iteration_history[iteration_a]["text"]
        if not isinstance(text_a, str):
            text_a = str(text_a)

        text_b = self.iteration_history[iteration_b]["text"]
        if not isinstance(text_b, str):
            text_b = str(text_b)

        return {
            "version_a": {
                "iteration": iteration_a,
                "description": self.iteration_history[iteration_a]["description"],
                "text": text_a,
            },
            "version_b": {
                "iteration": iteration_b
                if iteration_b >= 0
                else len(self.iteration_history) + iteration_b,
                "description": self.iteration_history[iteration_b]["description"],
                "text": text_b,
            },
        }


def main_cli():
    parser = argparse.ArgumentParser(description="Text Humanizer CLI (version 2)")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", type=str, help="Text to humanize directly.")
    input_group.add_argument(
        "--file", type=str, help="Path to a text file to humanize."
    )
    input_group.add_argument(
        "--folder",
        type=str,
        help="Path to a folder containing .txt and .md files to humanize.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of humanization iterations (default: all available prompts).",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress and cost messages (default: True).",
    )

    args = parser.parse_args()

    try:
        humanizer = TextHumanizer()
    except ValueError as e:
        print(f"Error initializing TextHumanizer: {e}")
        return

    texts_to_process = []

    if args.text:
        texts_to_process.append({"source": "command-line text", "content": args.text})
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                texts_to_process.append({"source": args.file, "content": f.read()})
        except FileNotFoundError:
            print(f"Error: File not found at {args.file}")
            return
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            return
    elif args.folder:
        found_files = []
        for ext in ("*.txt", "*.md"):
            found_files.extend(glob.glob(os.path.join(args.folder, ext)))

        if not found_files:
            print(f"No .txt or .md files found in folder {args.folder}")
            return

        for filepath in found_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    texts_to_process.append({"source": filepath, "content": f.read()})
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                continue
    else:
        try:
            print(
                "No input provided via arguments. Please enter text to humanize (Ctrl+D or Ctrl+Z then Enter to finish):"
            )
            user_input_lines = []
            while True:
                line = input()
                user_input_lines.append(line)
        except EOFError:
            user_text = "\n".join(user_input_lines)
            if not user_text.strip():
                print("No input received. Exiting.")
                return
            texts_to_process.append(
                {"source": "interactive input", "content": user_text}
            )
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return

    if not texts_to_process:
        print("No text to process. Exiting.")
        return

    for item in texts_to_process:
        if args.verbose:
            print(f"\n--- Humanizing content from: {item['source']} ---")
            print(
                f"Original Text:\n{item['content'][:500]}{'...' if len(item['content']) > 500 else ''}"
            )

        humanized_text = humanizer.humanize_text(
            input_text=item["content"], iterations=args.iterations, verbose=args.verbose
        )

        if args.verbose:
            print(f"\n=== Final Humanized Text from: {item['source']} ===")
        print(humanized_text)
        if args.verbose and len(texts_to_process) > 1:
            print("---------------------------------------------------")


if __name__ == "__main__":
    main_cli()
