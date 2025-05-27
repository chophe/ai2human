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
        extra_args: List of dicts for extra argparse arguments (see below)
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
    # Add any extra arguments
    if extra_args:
        for arg in extra_args:
            parser.add_argument(**arg)

    args = parser.parse_args()

    # Setup extra kwargs for processing
    extra_kwargs = extra_setup(args) if extra_setup else {}

    try:
        humanizer = humanizer_class()
    except Exception as e:
        print(f"Error initializing humanizer: {e}")
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
        result = process_func(
            humanizer, item["content"], {**vars(args), **extra_kwargs}
        )
        if args.verbose:
            print(f"\n=== Final Humanized Text from: {item['source']} ===")
        print(result)
        if args.verbose and len(texts_to_process) > 1:
            print("---------------------------------------------------")
