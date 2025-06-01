import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from typing import List, Dict, Any, Optional, Annotated
import typer

# Import generic_main_cli from the local utils file
from .humanize_cli_utils import generic_main_cli

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()


class TextHumanizer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the TextHumanizer.
        API key sourced from param, then OPENAI_API_KEY env. Model from param, env, then default.
        """
        actual_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not actual_api_key:
            typer.echo(
                "OpenAI API key not provided or found in environment (OPENAI_API_KEY).",
                err=True,
                color=typer.colors.RED,
            )
            raise typer.Exit(code=1)
        os.environ["OPENAI_API_KEY"] = actual_api_key
        self.verbose = verbose

        final_model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        actual_api_base_url = (
            api_base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        )

        llm_kwargs: Dict[str, Any] = {
            "temperature": 0.7,
            "model_name": final_model_name,
        }
        if actual_api_base_url:
            llm_kwargs["base_url"] = actual_api_base_url

        if self.verbose:
            typer.echo(
                f"Initializing TextHumanizer with model: {final_model_name}",
                color=typer.colors.BLUE,
            )
            if actual_api_base_url:
                typer.echo(
                    f"Using base URL: {actual_api_base_url}", color=typer.colors.BLUE
                )

        try:
            self.llm = ChatOpenAI(openai_api_key=actual_api_key, **llm_kwargs)
        except Exception as e:
            typer.echo(f"Error initializing LLM: {e}", err=True, color=typer.colors.RED)
            raise typer.Exit(code=1)

        self.iteration_history: List[Dict[str, Any]] = []

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
        self,
        input_text: str,
        iterations: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> str:
        current_verbose = self.verbose if verbose is None else verbose

        prompts = self.create_humanization_prompts()
        num_prompts = len(prompts)
        actual_iterations = (
            num_prompts if iterations is None else min(iterations, num_prompts)
        )

        current_text = input_text
        self.iteration_history = [
            {"iteration": 0, "text": input_text, "description": "Original text"}
        ]
        total_tokens = 0
        total_cost = 0.0

        if not input_text or not input_text.strip():
            if current_verbose:
                typer.echo(
                    "Input text is empty. Returning as is.", color=typer.colors.YELLOW
                )
            return input_text

        for i in range(actual_iterations):
            prompt_info = prompts[i]
            if current_verbose:
                typer.echo(f"\n{typer.style('='*50, fg=typer.colors.BLUE)}")
                typer.echo(
                    typer.style(
                        f"Iteration {i+1}/{actual_iterations}: {prompt_info['description']}",
                        bold=True,
                        fg=typer.colors.CYAN,
                    )
                )
                typer.echo(typer.style(f"{'='*50}", fg=typer.colors.BLUE))

            prompt_template = PromptTemplate(
                input_variables=["text"], template=prompt_info["template"]
            )
            chain = prompt_template | self.llm

            try:
                with get_openai_callback() as cb:
                    response = chain.invoke({"text": current_text})
                    new_text = (
                        response.content.strip()
                        if hasattr(response, "content")
                        else str(response).strip()
                    )

                    cb_total_tokens = cb.total_tokens
                    cb_total_cost = cb.total_cost

                total_tokens += cb_total_tokens
                total_cost += cb_total_cost

                if not new_text or new_text.lower() == current_text.lower():
                    if current_verbose:
                        typer.echo(
                            "No significant change or empty response in this iteration.",
                            color=typer.colors.YELLOW,
                        )
                    self.iteration_history.append(
                        {
                            "iteration": i + 1,
                            "text": current_text,
                            "description": prompt_info["description"],
                            "status": "No change or empty response",
                        }
                    )
                    if not new_text:
                        continue
                else:
                    current_text = new_text
                    self.iteration_history.append(
                        {
                            "iteration": i + 1,
                            "text": current_text,
                            "description": prompt_info["description"],
                            "status": "Processed",
                        }
                    )

                if current_verbose:
                    typer.echo(f"Tokens used this iteration: {cb_total_tokens}")
                    typer.echo(f"Cost this iteration: ${cb_total_cost:.6f}")
            except Exception as e:
                if current_verbose:
                    typer.echo(
                        f"Error during iteration {i+1} ({prompt_info['description']}): {e}",
                        err=True,
                        color=typer.colors.RED,
                    )
                self.iteration_history.append(
                    {
                        "iteration": i + 1,
                        "text": current_text,
                        "description": prompt_info["description"],
                        "status": f"Error: {e}",
                    }
                )
                break

        if current_verbose:
            typer.echo(f"\n{typer.style('='*50, fg=typer.colors.GREEN)}")
            typer.echo(f"Total tokens used for all iterations: {total_tokens}")
            typer.echo(f"Total cost for all iterations: ${total_cost:.6f}")
            typer.echo(typer.style(f"{'='*50}", fg=typer.colors.GREEN))
        return current_text

    def get_iteration_history(self) -> List[Dict[str, Any]]:
        return self.iteration_history

    def compare_versions(
        self, iteration_a: int = 0, iteration_b: int = -1
    ) -> Dict[str, Any]:
        if not self.iteration_history:
            return {"error": "No iteration history available"}
        num_history = len(self.iteration_history)
        idx_a = iteration_a if iteration_a < num_history else num_history - 1
        idx_b = (
            iteration_b
            if iteration_b != -1 and iteration_b < num_history
            else num_history - 1
        )
        if idx_a < 0 or idx_a >= num_history or idx_b < 0 or idx_b >= num_history:
            return {
                "error": f"Invalid iteration indices: {iteration_a}, {iteration_b} for history of length {num_history}"
            }

        text_a = self.iteration_history[idx_a]["text"]
        text_b = self.iteration_history[idx_b]["text"]

        return {
            "version_a": {
                "iteration": self.iteration_history[idx_a]["iteration"],
                "description": self.iteration_history[idx_a]["description"],
                "text": str(text_a),
            },
            "version_b": {
                "iteration": self.iteration_history[idx_b]["iteration"],
                "description": self.iteration_history[idx_b]["description"],
                "text": str(text_b),
            },
        }


def _humanize2_process_func(
    humanizer_instance: TextHumanizer, text: str, extra_kwargs: Dict[str, Any]
) -> str:
    cli_verbose = extra_kwargs.get("verbose", True)
    if (
        hasattr(humanizer_instance, "verbose")
        and humanizer_instance.verbose != cli_verbose
    ):
        humanizer_instance.verbose = cli_verbose

    return humanizer_instance.humanize_text(
        input_text=text, iterations=extra_kwargs.get("iterations"), verbose=cli_verbose
    )


def _humanize2_extra_setup(cli_args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "api_key": cli_args.get("api_key"),
        "model_name": cli_args.get("model"),
        "api_base_url": cli_args.get("base_url"),
        "verbose": cli_args.get("verbose", True),
        "iterations": cli_args.get("iterations"),
    }


app = typer.Typer(
    name="humanize2",
    help="Text Humanizer CLI (v2) - Iteratively refines text to sound more natural.",
    add_completion=False,
    no_args_is_help=True,
)

humanize2_command_func = generic_main_cli(
    humanizer_class=TextHumanizer,
    process_func=_humanize2_process_func,
    extra_setup=lambda cli_args_dict: {
        "api_key": cli_args_dict.get("api_key"),
        "model_name": cli_args_dict.get("model"),
        "api_base_url": cli_args_dict.get("base_url"),
        "verbose": cli_args_dict.get("verbose", True),
    },
)


@app.command(
    help="Humanizes text from string, file, or folder using iterative LLM calls.",
    no_args_is_help=True,
)
def main(
    text: Annotated[
        Optional[str],
        typer.Option(
            help="Text to humanize directly.", rich_help_panel="Input Options"
        ),
    ] = None,
    file: Annotated[
        Optional[str],
        typer.Option(
            help="Path to a text file to humanize.", rich_help_panel="Input Options"
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
        bool, typer.Option(help="Enable verbose output (iteration details, costs).")
    ] = True,
    model: Annotated[
        Optional[str],
        typer.Option(
            help="OpenAI model (e.g., gpt-4o). Overrides OPENAI_MODEL_NAME.",
            rich_help_panel="LLM Configuration",
        ),
    ] = None,
    iterations: Annotated[
        Optional[int],
        typer.Option(
            min=1, help="Number of humanization iterations (default: all prompts)."
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            help="OpenAI API key (overrides env OPENAI_API_KEY). Env variable takes precedence if this is not set.",
            rich_help_panel="LLM Configuration",
        ),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            help="Custom OpenAI API base URL (overrides env OPENAI_BASE_URL). Env variable takes precedence if this is not set.",
            rich_help_panel="LLM Configuration",
        ),
    ] = None,
):
    humanize2_command_func(
        text=text,
        file=file,
        folder=folder,
        verbose=verbose,
        model=model,
        iterations=iterations,
        api_key=api_key,
        base_url=base_url,
    )


if __name__ == "__main__":
    app()
