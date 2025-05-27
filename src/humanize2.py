import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from typing import List, Dict
import importlib.util

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
        llm_kwargs = {"temperature": 0.7, "model_name": model_name}
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


def _process_func(humanizer, text, extra_kwargs):
    return humanizer.humanize_text(
        input_text=text,
        iterations=extra_kwargs.get("iterations"),
        verbose=extra_kwargs.get("verbose", True),
    )


def _extra_args():
    return [
        {
            "flags": ["--iterations"],
            "type": int,
            "default": None,
            "help": "Number of humanization iterations (default: all available prompts).",
        },
    ]


if __name__ == "__main__":
    # Dynamically import the generic_main_cli from humanize_cli_utils.py
    import importlib.util
    import sys

    cli_utils_path = os.path.join(os.path.dirname(__file__), "humanize_cli_utils.py")
    spec = importlib.util.spec_from_file_location("humanize_cli_utils", cli_utils_path)
    cli_utils = importlib.util.module_from_spec(spec)
    sys.modules["humanize_cli_utils"] = cli_utils
    spec.loader.exec_module(cli_utils)

    cli_utils.generic_main_cli(
        description="Text Humanizer CLI (version 2)",
        humanizer_class=TextHumanizer,
        process_func=_process_func,
        extra_args=_extra_args(),
        extra_setup=lambda args: {
            "iterations": args.iterations,
            "verbose": args.verbose,
        },
    )
