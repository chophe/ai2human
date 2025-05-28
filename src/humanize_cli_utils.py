import os
import glob
import argparse
from typing import Callable, Any, Dict, List, Optional


def generic_main_cli(
    description: str,
    humanizer_class: Callable[..., Any],
    process_func: Callable[[Any, str, Dict[str, Any]], str],
    extra_args: Optional[List[Dict[str, Any]]] = None,
    extra_setup: Optional[Callable[[argparse.Namespace], Dict[str, Any]]] = None,
):
    """
    Generic CLI handler for humanizer scripts.

    Args:
        description: CLI description string
        humanizer_class: The class to instantiate for humanization
        process_func: Function to call for processing (instance, text, extra_kwargs) -> str
        extra_args: List of dicts for extra argparse arguments.
                    Each dict should have a 'flags' key (list of strings for arg names)
                    and other keys as valid kwargs for parser.add_argument().
        extra_setup: Function to extract extra kwargs from parsed args
    """
    parser = argparse.ArgumentParser(description=description)
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
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress and cost messages (default: True).",
    )
    # Add --model argument
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name of the LLM model to use (e.g., gpt-4o, gpt-3.5-turbo). Overrides OPENAI_MODEL_NAME env variable.",
    )
    # Add any extra arguments
    if extra_args:
        for arg_config in extra_args:
            flags = arg_config.pop("flags", None)
            if flags:
                parser.add_argument(*flags, **arg_config)
            else:
                # This case should ideally not happen if extra_args are structured correctly
                print(f"Warning: Argument config missing 'flags': {arg_config}")

    args = parser.parse_args()

    # Setup extra kwargs for processing
    extra_kwargs_from_setup = extra_setup(args) if extra_setup else {}

    # Initialize humanizer/detector class
    # Pass API key and base_url if the class expects them and they are in os.environ
    init_kwargs = {}
    # Check if humanizer_class is a type (class) before inspecting constructor
    # and not a lambda function that might not have typical class properties.
    if isinstance(humanizer_class, type):
        try:
            constructor_params = list(humanizer_class.__init__.__code__.co_varnames)
            constructor_params = constructor_params[
                : humanizer_class.__init__.__code__.co_argcount
            ]

            # API Key
            if "api_key" in constructor_params and os.getenv("OPENAI_API_KEY"):
                init_kwargs["api_key"] = os.getenv("OPENAI_API_KEY")

            # Base URL
            # Prefer OPENAI_BASE_URL for clarity, but support OPENAI_API_BASE for TextHumanizer
            base_url_to_use = os.getenv("OPENAI_BASE_URL") or os.getenv(
                "OPENAI_API_BASE"
            )
            if base_url_to_use:
                if "base_url" in constructor_params:
                    init_kwargs["base_url"] = base_url_to_use
                if (
                    "api_base_url" in constructor_params
                ):  # Specifically for TextHumanizer
                    init_kwargs["api_base_url"] = base_url_to_use

            # Model Name - prioritize CLI, then ENV
            determined_model_name = args.model  # From CLI
            if not determined_model_name:
                determined_model_name = os.getenv("OPENAI_MODEL_NAME")  # From ENV

            if determined_model_name:
                if "model_name" in constructor_params:
                    init_kwargs["model_name"] = determined_model_name
                elif "model" in constructor_params:  # some classes might use 'model'
                    init_kwargs["model"] = determined_model_name

        except (
            AttributeError
        ):  # Lambdas or other callables might not have __init__ or __code__
            pass

    try:
        humanizer_instance = humanizer_class(**init_kwargs)
    except Exception as e:
        print(
            f"Error initializing class {humanizer_class.__name__ if hasattr(humanizer_class, '__name__') else str(humanizer_class)}: {e}"
        )
        return

    texts_to_process = []
    source_for_processing = "Input Text"

    if args.text:
        texts_to_process.append({"source": "command-line text", "content": args.text})
        source_for_processing = "Command-line text"
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                texts_to_process.append({"source": args.file, "content": f.read()})
            source_for_processing = args.file
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
                continue  # Process next file
        source_for_processing = (
            args.folder
        )  # For a general source name if multiple files
    else:
        try:
            print(
                "No input provided via arguments. Please enter text to humanize/analyze (Ctrl+D or Ctrl+Z then Enter to finish):"
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
            source_for_processing = "Interactive input"
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return

    if not texts_to_process:
        # This case should ideally be caught earlier for file/folder if no files are found/readable.
        # For interactive input, it's caught if user_text is empty.
        print("No text to process. Exiting.")
        return

    # Prepare combined kwargs for process_func, including args and setup_kwargs
    # Give priority to specific kwargs from extra_setup if there are name clashes.
    # Also pass the 'source' determined above to extra_kwargs for _process_func.
    all_cli_args = {
        **vars(args),
        **extra_kwargs_from_setup,
        "source": source_for_processing,
    }

    for item in texts_to_process:
        current_source = item[
            "source"
        ]  # Use specific source for each item in batch processing
        if args.verbose:
            print(f"\n--- Processing content from: {current_source} ---")
            print(
                f"Original Text:\n{item['content'][:500]}{'...' if len(item['content']) > 500 else ''}"
            )

        # Update the 'source' in all_cli_args for the current item being processed
        # This allows _process_func to know the specific source of the current text item.
        current_item_cli_args = {**all_cli_args, "source": current_source}

        result = process_func(
            humanizer_instance, item["content"], current_item_cli_args
        )
        if args.verbose:
            print(f"\n=== Result for: {current_source} ===")
        print(result)
        if args.verbose and len(texts_to_process) > 1:
            print("---------------------------------------------------")
