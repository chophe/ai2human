import os
import glob
import typer
from typing import Callable, Any, Dict, List, Optional, Annotated


def generic_main_cli(
    humanizer_class: Callable[..., Any],
    process_func: Callable[[Any, str, Dict[str, Any]], str],
    extra_args_def: Optional[List[Dict[str, Any]]] = None,
    extra_setup: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Callable[..., None]:
    """
    Generic CLI handler factory for humanizer scripts using Typer.
    Returns a function that can be registered as a Typer command.
    """

    def command_function(
        text: Annotated[
            Optional[str],
            typer.Option(
                help="Text to process directly.", rich_help_panel="Input Options"
            ),
        ] = None,
        file: Annotated[
            Optional[str],
            typer.Option(
                help="Path to a text file to process.", rich_help_panel="Input Options"
            ),
        ] = None,
        folder: Annotated[
            Optional[str],
            typer.Option(
                help="Path to a folder with .txt/.md files to process.",
                rich_help_panel="Input Options",
            ),
        ] = None,
        verbose: Annotated[
            bool, typer.Option(help="Print progress and cost messages.")
        ] = True,
        model: Annotated[
            Optional[str],
            typer.Option(
                help="LLM model (e.g., gpt-4o). Overrides OPENAI_MODEL_NAME env var."
            ),
        ] = None,
        **kwargs: Any,
    ) -> None:
        """
        This function will be registered as a Typer command in the calling script.
        It handles the core processing logic.
        """
        cli_args = {
            "text": text,
            "file": file,
            "folder": folder,
            "verbose": verbose,
            "model": model,
            **kwargs,
        }

        input_options_count = sum(
            1 for option in [text, file, folder] if option is not None
        )
        if input_options_count > 1:
            typer.echo(
                "Error: --text, --file, and --folder are mutually exclusive.", err=True
            )
            raise typer.Exit(code=1)

        extra_kwargs_from_setup = extra_setup(cli_args) if extra_setup else {}

        init_kwargs: Dict[str, Any] = {}
        if isinstance(humanizer_class, type):
            try:
                constructor = getattr(humanizer_class, "__init__", None)
                if constructor:
                    arg_spec = getattr(constructor, "__code__", None)
                    if arg_spec:
                        constructor_params = arg_spec.co_varnames[
                            : arg_spec.co_argcount
                        ]

                        if "api_key" in constructor_params and os.getenv(
                            "OPENAI_API_KEY"
                        ):
                            init_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

                        base_url_to_use = os.getenv("OPENAI_BASE_URL") or os.getenv(
                            "OPENAI_API_BASE"
                        )
                        if base_url_to_use:
                            if "base_url" in constructor_params:
                                init_kwargs["base_url"] = base_url_to_use
                            if "api_base_url" in constructor_params:
                                init_kwargs["api_base_url"] = base_url_to_use

                        determined_model_name = cli_args.get("model")
                        if not determined_model_name:
                            determined_model_name = os.getenv("OPENAI_MODEL_NAME")

                        if determined_model_name:
                            if "model_name" in constructor_params:
                                init_kwargs["model_name"] = determined_model_name
                            elif "model" in constructor_params:
                                init_kwargs["model"] = determined_model_name
            except AttributeError:
                if cli_args["verbose"]:
                    typer.echo(
                        "Notice: Could not fully introspect humanizer_class constructor for API keys/model.",
                        color=typer.colors.YELLOW,
                    )

        if "model" not in init_kwargs and cli_args.get("model"):
            if not isinstance(humanizer_class, type) or not hasattr(
                humanizer_class.__init__, "__code__"
            ):
                init_kwargs["model"] = cli_args["model"]
            elif isinstance(humanizer_class, type) and hasattr(
                humanizer_class.__init__, "__code__"
            ):
                if humanizer_class.__init__.__code__.co_flags & 0x08:
                    init_kwargs["model"] = cli_args["model"]

        try:
            humanizer_instance = humanizer_class(**init_kwargs)
        except Exception as e:
            typer.echo(
                f"Error initializing class {getattr(humanizer_class, '__name__', str(humanizer_class))}: {e}",
                err=True,
            )
            raise typer.Exit(code=1)

        texts_to_process: List[Dict[str, str]] = []
        source_for_processing = "Input Text"

        if cli_args.get("text"):
            texts_to_process.append(
                {"source": "command-line text", "content": cli_args["text"]}
            )
            source_for_processing = "Command-line text"
        elif cli_args.get("file"):
            file_path = cli_args["file"]
            try:
                with open(file_path, "r", encoding="utf-8") as f_in:
                    texts_to_process.append(
                        {"source": file_path, "content": f_in.read()}
                    )
                source_for_processing = file_path
            except FileNotFoundError:
                typer.echo(f"Error: File not found at {file_path}", err=True)
                raise typer.Exit(code=1)
            except Exception as e:
                typer.echo(f"Error reading file {file_path}: {e}", err=True)
                raise typer.Exit(code=1)
        elif cli_args.get("folder"):
            folder_path = cli_args["folder"]
            found_files_paths = []
            for ext in ("*.txt", "*.md"):
                found_files_paths.extend(glob.glob(os.path.join(folder_path, ext)))

            if not found_files_paths:
                typer.echo(
                    f"No .txt or .md files found in folder {folder_path}", err=True
                )
                raise typer.Exit(code=1)

            for filepath_item in found_files_paths:
                try:
                    with open(filepath_item, "r", encoding="utf-8") as f_in:
                        texts_to_process.append(
                            {"source": filepath_item, "content": f_in.read()}
                        )
                except Exception as e:
                    typer.echo(f"Error reading file {filepath_item}: {e}", err=True)
                    continue
            source_for_processing = folder_path
        else:
            typer.echo(
                "No input source (--text, --file, --folder) provided. Reading from stdin.",
                color=typer.colors.BLUE,
            )
            typer.echo(
                "Enter text (Ctrl+D or Ctrl+Z then Enter on Windows to finish):",
                color=typer.colors.BLUE,
            )
            user_input_lines = []
            try:
                while True:
                    line = input()
                    user_input_lines.append(line)
            except EOFError:
                user_text = "\n".join(user_input_lines)
                if not user_text.strip():
                    typer.echo("No input received from stdin. Exiting.", err=True)
                    raise typer.Exit(code=1)
                texts_to_process.append(
                    {"source": "interactive stdin", "content": user_text}
                )
                source_for_processing = "Interactive stdin"
            except KeyboardInterrupt:
                typer.echo("\nOperation cancelled by user.", err=True)
                raise typer.Exit(code=1)

        if not texts_to_process:
            typer.echo("No text to process. Exiting.", err=True)
            raise typer.Exit(code=1)

        all_process_kwargs = {
            **cli_args,
            **extra_kwargs_from_setup,
        }

        for item_idx, item in enumerate(texts_to_process):
            current_text_content = item.get("content", "")
            current_source = item.get("source", "Unknown source")

            if cli_args.get("verbose", True):
                typer.echo(f"\n--- Processing content from: ", nl=False)
                typer.echo(f"{current_source}", color=typer.colors.CYAN)
                display_content = current_text_content[:500] + (
                    "..." if len(current_text_content) > 500 else ""
                )
                typer.echo(f"Original Text (first 500 chars):\n{display_content}")

            current_item_process_kwargs = {
                **all_process_kwargs,
                "source": current_source,
            }

            try:
                result = process_func(
                    humanizer_instance,
                    current_text_content,
                    current_item_process_kwargs,
                )
                if cli_args.get("verbose", True):
                    typer.echo(f"\n=== Result for: ", nl=False)
                    typer.echo(f"{current_source}", color=typer.colors.CYAN, bold=True)
                    typer.echo(" ===", bold=True)
                typer.echo(result)
            except Exception as e:
                typer.echo(
                    f"Error during processing of {current_source}: {e}",
                    err=True,
                    color=typer.colors.RED,
                )
                if len(texts_to_process) == 1:
                    raise typer.Exit(code=1)

            if (
                cli_args.get("verbose", True)
                and len(texts_to_process) > 1
                and item_idx < len(texts_to_process) - 1
            ):
                typer.echo(
                    "---------------------------------------------------",
                    color=typer.colors.BLUE,
                )

        if cli_args.get("verbose", True):
            typer.echo("\n✨ Processing complete. ✨", color=typer.colors.GREEN)

    return command_function
