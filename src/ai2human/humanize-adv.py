import os
import json
from datetime import datetime
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional, Any, Annotated
import hashlib
import textwrap
import argparse
import glob
import importlib.util
import typer
from dotenv import load_dotenv
from .humanize_cli_utils import generic_main_cli

load_dotenv()

class AdvancedTextHumanizer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_chat_model: bool = True,
        base_url: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the Advanced TextHumanizer.
        API key will be sourced from param, then OPENAI_API_KEY env var.
        """
        actual_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not actual_api_key:
            typer.echo("OpenAI API key not provided or found in environment (OPENAI_API_KEY).", err=True, color=typer.colors.RED)
            raise typer.Exit(code=1)
        
        os.environ["OPENAI_API_KEY"] = actual_api_key
        self.verbose = verbose

        final_model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
        actual_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        llm_params: Dict[str, Any] = {
            "temperature": 0.7,
            "model_name": final_model_name,
        }
        if actual_base_url:
            llm_params["openai_api_base"] = actual_base_url

        if self.verbose: 
            typer.echo(f"Initializing AdvancedTextHumanizer with model: {final_model_name}", color=typer.colors.BLUE)
            if actual_base_url: typer.echo(f"Using base URL: {actual_base_url}", color=typer.colors.BLUE)

        try:
            if use_chat_model:
                self.llm = ChatOpenAI(openai_api_key=actual_api_key, **llm_params)
            else:
                if "openai_api_base" in llm_params and "base_url" not in llm_params:
                    llm_params["base_url"] = llm_params["openai_api_base"]
                self.llm = OpenAI(openai_api_key=actual_api_key, **llm_params)
        except Exception as e:
            typer.echo(f"Error initializing LLM: {e}", err=True, color=typer.colors.RED)
            raise typer.Exit(code=1)

        self.iteration_history: List[Dict[str, Any]] = []
        self.memory = ConversationBufferMemory()

    def create_style_specific_prompts(
        self, style: str = "conversational"
    ) -> List[Dict[str, str]]:
        style_guidelines = {
            "conversational": {
                "tone": "friendly and approachable",
                "features": ["contractions", "personal pronouns", "rhetorical questions"],
                "avoid": ["jargon", "overly formal language"],
            },
            "professional": {
                "tone": "polished, clear, and concise",
                "features": ["clear structure", "confident language", "appropriate industry terminology"],
                "avoid": ["slang", "overly casual expressions", "ambiguity"],
            },
            "casual": {
                "tone": "relaxed and informal",
                "features": ["colloquialisms", "humor (if appropriate)", "direct address"],
                "avoid": ["stiffness", "excessive formality", "technical jargon unless explained"],
            },
            "academic": {
                "tone": "scholarly, objective, and precise",
                "features": ["clear thesis", "logical argumentation", "citations (if applicable, placeholder here)", "formal vocabulary"],
                "avoid": ["casual language", "unsubstantiated claims", "emotional appeals"],
            },
            "technical": {
                "tone": "precise, informative, and unambiguous",
                "features": ["accurate terminology", "clear explanations of complex topics", "logical structure", "step-by-step instructions where applicable"],
                "avoid": ["vagueness", "colloquialisms", "unnecessary jargon if audience is mixed"],
            },
        }
        guidelines = style_guidelines.get(style, style_guidelines["conversational"])
        prompts = [
            {
                "description": f"Apply {style} style focusing on tone and core features",
                "template": f"""
                Revise the following text to adopt a {guidelines['tone']} tone. 
                Key characteristics to incorporate: {', '.join(guidelines['features'])}. 
                What to avoid: {', '.join(guidelines['avoid'])}. 
                Preserve the original meaning and factual information unless the style explicitly requires a shift (e.g. simplifying for a casual audience).
                Original Text:
                {{text}}
                Revised Text:
                """,
            },
            {
                "description": "Enhance flow, clarity, and engagement",
                "template": """
                Refine the text below for better flow, clarity, and engagement, maintaining the established {style} style. 
                Consider varying sentence structure, improving transitions, and ensuring the language is engaging for the target reader.
                If the style is '{style}', ensure it is consistently applied.
                Current Text:
                {{text}}
                Refined Text:
                """,
            },
            {
                "description": "Final polish for human-like quality and {style} consistency",
                "template": """
                Perform a final polish on this text. Ensure it sounds natural, human-like, and perfectly embodies the {style} style. 
                Check for any remaining robotic phrasing, awkward sentences, or inconsistencies in tone. 
                The text should read as if a skilled human writer crafted it in the {style} style.
                Current Text:
                {{text}}
                Polished Text:
                """,
            },
        ]
        return prompts

    def analyze_text_formality(self, text: str) -> Dict[str, Any]:
        analysis_prompt_template = """
        Analyze the formality level of this text. Provide a JSON response including:
        1. "formality_score" (integer 1-10, 1=very informal, 10=very formal).
        2. "detected_style" (e.g., conversational, professional, casual, academic, technical, neutral).
        3. "key_indicators" (list of phrases or features from the text supporting your analysis).
        Text:
        {text}
        JSON Response:
        """
        analysis_prompt = PromptTemplate(input_variables=["text"], template=analysis_prompt_template)
        chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
        try:
            response_str = chain.run(text=text)
            match = re.search(r"\{.*\}|\\[.*\\]\", response_str, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                if self.verbose: typer.echo("Warning: Could not extract JSON from formality analysis response.", color=typer.colors.YELLOW)
                return {"formality_score": 5, "detected_style": "neutral", "key_indicators": ["Failed to parse LLM JSON response"]}
        except json.JSONDecodeError as e:
            if self.verbose: typer.echo(f"JSONDecodeError in formality analysis: {e}. Response: {response_str[:200]}", color=typer.colors.YELLOW)
            return {"formality_score": 5, "detected_style": "neutral", "key_indicators": [f"JSON parsing error: {e}"]}
        except Exception as e:
            if self.verbose: typer.echo(f"Error in formality analysis: {e}", color=typer.colors.RED)
            return {"formality_score": 5, "detected_style": "neutral", "key_indicators": [f"Analysis error: {e}"]}

    def humanize_with_context(
        self,
        input_text: str,
        context: Optional[str] = None,
        target_style: str = "conversational",
        max_iterations: int = 3,
        humanizer_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not input_text or not input_text.strip():
            return {"error": "Input text is empty.", "final_text": ""}

        original_analysis = self.analyze_text_formality(input_text)
        text_hash = hashlib.md5(input_text.encode()).hexdigest()[:8]
        results: Dict[str, Any] = {
            "original_text": input_text,
            "original_analysis": original_analysis,
            "iterations_data": [],
            "final_text": input_text,
            "metadata": {
                "text_hash": text_hash,
                "timestamp": datetime.now().isoformat(),
                "target_style": target_style,
                "context_provided": bool(context),
                "max_iterations_requested": max_iterations,
                "model_used": self.llm.model_name if hasattr(self.llm, 'model_name') else "unknown",
            },
            "cost_info": None,
        }

        current_text = input_text
        prompts_to_run = self.create_style_specific_prompts(target_style)
        actual_iterations = 0

        with get_openai_callback() as cb:
            for i in range(min(max_iterations, len(prompts_to_run))):
                actual_iterations += 1
                prompt_info = prompts_to_run[i]
                current_prompt_template_str = prompt_info["template"]
                
                if isinstance(self.llm, ChatOpenAI):
                    chat_template = PromptTemplate.from_template(current_prompt_template_str)
                    chain = LLMChain(llm=self.llm, prompt=chat_template)
                else:
                    prompt_template_obj = PromptTemplate(input_variables=["text"], template=current_prompt_template_str)
                    chain = LLMChain(llm=self.llm, prompt=prompt_template_obj)

                if self.verbose: 
                    typer.echo(f"\nIteration {i+1}/{min(max_iterations, len(prompts_to_run))}: {prompt_info['description']}", color=typer.colors.CYAN)
                
                try:
                    run_input = {"text": current_text}
                    if "{context}" in current_prompt_template_str and context:
                        run_input["context"] = context
                        if not isinstance(self.llm, ChatOpenAI):
                            prompt_template_obj = PromptTemplate(input_variables=["text", "context"], template=current_prompt_template_str)
                            chain = LLMChain(llm=self.llm, prompt=prompt_template_obj)
                    
                    new_text = chain.run(**run_input).strip()

                    if not new_text or new_text.lower() == current_text.lower():
                        if self.verbose: typer.echo("No significant change or empty response in this iteration.", color=typer.colors.YELLOW)
                        results["iterations_data"].append({
                            "iteration": i + 1,
                            "prompt_description": prompt_info["description"],
                            "text_before": current_text,
                            "text_after": new_text,
                            "status": "No change or empty response"
                        })
                        if not new_text: continue 
                   
                    current_text = new_text
                    results["iterations_data"].append({
                        "iteration": i + 1,
                        "prompt_description": prompt_info["description"],
                        "text_before": results["iterations_data"][-1]["text_after"] if i > 0 and results["iterations_data"] else input_text,
                        "text_after": current_text,
                        "status": "Processed"
                    })
                    if self.verbose: typer.echo(textwrap.shorten(f"Output: {current_text}", width=150, placeholder="..."), color=typer.colors.GREEN)
                except Exception as e:
                    typer.echo(f"Error during humanization iteration {i+1}: {e}", err=True, color=typer.colors.RED)
                    results["iterations_data"].append({
                        "iteration": i + 1,
                        "prompt_description": prompt_info["description"],
                        "error": str(e),
                        "status": "Error"
                    })
                    break 
            results["final_text"] = current_text
            results["cost_info"] = cb.__dict__
            results["metadata"]["actual_iterations_completed"] = actual_iterations
        
        if self.verbose and results["cost_info"]:
            cost = results["cost_info"]["total_cost"]
            typer.echo(f"Total tokens: {cost.get('total_tokens',0)}, Total cost (USD): ${cost:.6f}", color=typer.colors.BLUE)

        return results

    def _verify_facts_preserved(self, original: str, humanized: str) -> str:
        verification_prompt_str = """
        Original text: "{original_text}"
        Revised text: "{revised_text}"
        Are all factual statements from the original text accurately preserved in the revised text? 
        If not, identify discrepancies. Respond with 'Facts Preserved' or list discrepancies.
        Response:
        """
        prompt = PromptTemplate(input_variables=["original_text", "revised_text"], template=verification_prompt_str)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        try:
            response = chain.run(original_text=original, revised_text=humanized)
            return response
        except Exception as e:
            if self.verbose: typer.echo(f"Fact verification call failed: {e}", color=typer.colors.YELLOW)
            return "Fact verification failed to run."

    def batch_humanize(
        self,
        texts_with_sources: List[Dict[str, str]],
        target_style: str = "conversational",
        output_folder: Optional[str] = None,
        max_iterations: int = 3
    ) -> List[Dict[str, Any]]:
        all_results = []
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            if self.verbose: typer.echo(f"Saving batch results to: {output_folder}", color=typer.colors.BLUE)

        for item in texts_with_sources:
            text_content = item.get("content")
            source_name = item.get("source", "unknown_source")
            if not text_content:
                if self.verbose: typer.echo(f"Skipping empty content from {source_name}", color=typer.colors.YELLOW)
                all_results.append({"source": source_name, "error": "Empty content", "final_text": ""})
                continue

            if self.verbose: typer.echo(f"\nProcessing batch item: {source_name}", color=typer.colors.MAGENTA)
            
            result_data = self.humanize_with_context(
                input_text=text_content,
                target_style=target_style,
                max_iterations=max_iterations,
                context=f"This text is from {source_name}"
            )
            result_data["metadata"]["source_file_or_id"] = source_name
            all_results.append(result_data)

            if output_folder and "final_text" in result_data:
                base_filename = os.path.basename(source_name)
                name, ext = os.path.splitext(base_filename)
                safe_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", name) + f"_humanized{ext or '.txt'}"
                output_path = os.path.join(output_folder, safe_filename)
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result_data["final_text"])
                    if self.verbose: typer.echo(f"Saved humanized text for {source_name} to {output_path}", color=typer.colors.GREEN)
                except Exception as e:
                    typer.echo(f"Error saving file {output_path}: {e}", err=True, color=typer.colors.RED)
            elif output_folder and "error" in result_data:
                 if self.verbose: typer.echo(f"Not saving file for {source_name} due to error: {result_data['error']}", color=typer.colors.YELLOW)
        return all_results

    def interactive_humanize(self, initial_text: str) -> str:
        current_text = initial_text
        if (
            not self.iteration_history
            or self.iteration_history[-1]["text"] != current_text
        ):
            self.iteration_history.append(
                {
                    "text": current_text,
                    "instruction": "Initial text for interactive session",
                }
            )

        while True:
            print("=" * 50)
            print("Current text:")
            print(current_text[:500] + ("..." if len(current_text) > 500 else ""))
            print("=" * 50)

            print("Choose an option:")
            print("1. Make more conversational")
            print("2. Add emotional depth")
            print("3. Simplify language")
            print("4. Add examples/analogies")
            print("5. Custom instruction")
            print("6. Undo last change")
            print("7. Finish")

            choice = input("Enter your choice (1-7): ")

            if choice == "7":
                break
            elif choice == "6":
                if (
                    len(self.iteration_history) > 1
                    and self.iteration_history[-1]["instruction"]
                    != "Initial text for interactive session"
                ):
                    self.iteration_history.pop()
                    current_text = self.iteration_history[-1]["text"]
                    print("Undone last change.")
                elif (
                    len(self.iteration_history) == 1
                    and self.iteration_history[0]["instruction"]
                    == "Initial text for interactive session"
                ):
                    print("At initial text, cannot undo further.")
                else:
                    print("No changes to undo or already at initial state.")
                continue

            instruction_map = {
                "1": "Make this text more conversational and friendly",
                "2": "Add appropriate emotional depth and human feelings",
                "3": "Simplify the language and make it more accessible",
                "4": "Add relevant examples or analogies to illustrate points",
            }

            instruction_text = instruction_map.get(choice)

            if choice == "5":
                instruction_text = input("Enter your custom instruction: ")

            if instruction_text:
                prompt = PromptTemplate(
                    input_variables=["text", "instruction"],
                    template="""
                    {instruction}
                    
                    Text:
                    {text}
                    
                    Result:
                    """,
                )

                chain = LLMChain(llm=self.llm, prompt=prompt)
                try:
                    processed_text = chain.run(
                        text=current_text, instruction=instruction_text
                    )
                    current_text = processed_text
                    self.iteration_history.append(
                        {"text": current_text, "instruction": instruction_text}
                    )
                except Exception as e:
                    print(f"Error during processing: {e}")

            else:
                if choice not in [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                ]:
                    print("Invalid choice, please try again.")

        return current_text

    def create_custom_pipeline(
        self, pipeline_steps: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        custom_prompts = []

        for step in pipeline_steps:
            prompt = {
                "description": step.get("description", "Custom step"),
                "template": f"""
                {step.get('instructions', 'Process the text according to the following instructions:')}
                
                Text to process:
                {{text}}
                
                Processed version:
                """,
            }
            custom_prompts.append(prompt)

        return custom_prompts


def compare_texts_side_by_side(original: str, humanized: str, width: int = 40):
    original_lines = original.splitlines()
    humanized_lines = humanized.splitlines()
    max_len = max(len(original_lines), len(humanized_lines))
    output = []
    header1 = "Original Text".center(width)
    header2 = "Humanized Text".center(width)
    output.append(typer.style(f"{header1} | {header2}", bold=True, fg=typer.colors.BLUE))
    output.append(typer.style("-" * (width * 2 + 3), fg=typer.colors.BLUE))

    for i in range(max_len):
        orig_line = textwrap.wrap(original_lines[i] if i < len(original_lines) else "", width)
        hum_line = textwrap.wrap(humanized_lines[i] if i < len(humanized_lines) else "", width)
        max_sub_lines = max(len(orig_line), len(hum_line))
        for j in range(max_sub_lines):
            ol = orig_line[j] if j < len(orig_line) else ""
            hl = hum_line[j] if j < len(hum_line) else ""
            output.append(f"{ol:<{width}} | {hl:<{width}}")
        if i < max_len -1 : output.append("-" * (width*2+3))

    return "\n".join(output)


def _adv_humanizer_process_func(humanizer_instance: AdvancedTextHumanizer, text_content: str, extra_kwargs: Dict[str, Any]) -> str:
    target_style = extra_kwargs.get("style", "conversational")
    context = extra_kwargs.get("context")
    iterations = extra_kwargs.get("iterations", 3)
    output_folder = extra_kwargs.get("output_folder")
    compare_output = extra_kwargs.get("compare", False)
    verbose_cli = extra_kwargs.get("verbose", False)

    if hasattr(humanizer_instance, 'verbose') and humanizer_instance.verbose != verbose_cli:
        humanizer_instance.verbose = verbose_cli
    
    if output_folder:
        source_name = extra_kwargs.get("source")
        if not source_name: source_name = "cli_folder_item"
        
        humanizer_instance.verbose = verbose_cli
        
        results_list = humanizer_instance.batch_humanize(
            texts_with_sources=[{"source": source_name, "content": text_content}],
            target_style=target_style,
            output_folder=output_folder,
            max_iterations=iterations
        )
        if results_list and results_list[0]:
            single_result = results_list[0]
            if "error" in single_result:
                return typer.style(f"Error processing {source_name}: {single_result['error']}", fg=typer.colors.RED)
            final_text = single_result.get("final_text", "")
            cost_info = single_result.get('cost_info')
            cost_str = f"Total Cost (USD): ${cost_info['total_cost']:.6f}" if cost_info else "Cost info n/a"
            return f"Processed: {source_name} -> {os.path.join(output_folder, os.path.basename(source_name) + '_humanized.txt')}. {cost_str}"
        else:
            return typer.style(f"Error: Batch processing for {source_name} returned no results.", fg=typer.colors.RED)
    else:
        results = humanizer_instance.humanize_with_context(
            input_text=text_content,
            context=context,
            target_style=target_style,
            max_iterations=iterations
        )
        if "error" in results:
            return typer.style(f"Error: {results['error']}", fg=typer.colors.RED)
        
        final_text = results.get("final_text", "")
        original_text = results.get("original_text", text_content)
        cost_info = results.get('cost_info')
        cost_str = f"Total Cost (USD): ${cost_info['total_cost']:.6f}" if cost_info else "Cost info n/a"
        output_str = ""
        if compare_output:
            output_str = compare_texts_side_by_side(original_text, final_text)
        else:
            output_str = final_text
        return f"{output_str}\n\n{cost_str}"


def _adv_humanizer_extra_setup(cli_args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "style": cli_args.get("style", "conversational"),
        "context": cli_args.get("context"),
        "iterations": cli_args.get("iterations", 3),
        "output_folder": cli_args.get("output_folder"),
        "compare": cli_args.get("compare", False),
        "verbose": cli_args.get("verbose", False),
        "model": cli_args.get("model"),
        "api_key": cli_args.get("api_key"),
        "base_url": cli_args.get("base_url"),
        "humanizer_api_key": cli_args.get("api_key"),
        "humanizer_model_name": cli_args.get("model"),
        "humanizer_base_url": cli_args.get("base_url"),
        "humanizer_verbose": cli_args.get("verbose"),
    }


app = typer.Typer(
    name="humanize-adv",
    help="Advanced Text Humanizer CLI with multiple styles, context, and iteration control.",
    add_completion=False,
    no_args_is_help=True,
)

adv_humanizer_command_func = generic_main_cli(
    humanizer_class=AdvancedTextHumanizer,
    process_func=_adv_humanizer_process_func,
    extra_setup=_adv_humanizer_extra_setup,
)

@app.command(
    help="Humanizes text using advanced options like style, context, and iterations.",
    no_args_is_help=True
)
def main(
    text: Annotated[Optional[str], typer.Option(help="Text to humanize directly.", rich_help_panel="Input Options")] = None,
    file: Annotated[Optional[str], typer.Option(help="Path to a text file to humanize.", rich_help_panel="Input Options")] = None,
    folder: Annotated[Optional[str], typer.Option(help="Path to a folder with .txt/.md files to humanize (implies batch processing to output-folder if set).", rich_help_panel="Input Options")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output (humanization steps, costs, etc.).")] = False,
    model: Annotated[Optional[str], typer.Option(help="OpenAI model name (e.g., gpt-4o, gpt-3.5-turbo). Overrides OPENAI_MODEL_NAME env var.", rich_help_panel="Model Configuration")] = None,
    
    style: Annotated[str, typer.Option(help="Target humanization style.", case_sensitive=False, rich_help_panel="Humanization Options")] = "conversational",
    context_text: Annotated[Optional[str], typer.Option("--context", help="Additional context for humanization.", rich_help_panel="Humanization Options")] = None,
    iterations: Annotated[int, typer.Option(min=1, max=5, help="Number of humanization iterations (1-3 recommended).", rich_help_panel="Humanization Options")] = 3,
    output_folder: Annotated[Optional[str], typer.Option(help="Folder to save humanized text files (batch mode for folder input, or single file output).", rich_help_panel="Output Options")] = None,
    compare: Annotated[bool, typer.Option(help="Show side-by-side comparison of original and humanized text.", rich_help_panel="Output Options")] = False,
    api_key: Annotated[Optional[str], typer.Option(help="OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.", rich_help_panel="Model Configuration")] = None,
    base_url: Annotated[Optional[str], typer.Option(help="Optional custom base URL for OpenAI API. Overrides OPENAI_BASE_URL env var.", rich_help_panel="Model Configuration")] = None,
):
    """
    Advanced Text Humanizer: Applies various styles, uses context, and iterates for refined output.
    Accepts direct text, a single file, or a folder of .txt/.md files.
    When --folder is used, it processes all found files. If --output-folder is also specified, 
    humanized versions are saved there. Otherwise, results for folder processing are printed to console.
    """
    if folder and not output_folder:
        typer.echo(typer.style("Error: --output-folder is required when using --folder for batch processing.", fg=typer.colors.RED, bold=True))
        raise typer.Exit(code=1)

    adv_humanizer_command_func(
        text=text,
        file=file,
        folder=folder,
        verbose=verbose,
        model=model,
        style=style,
        context=context_text,
        iterations=iterations,
        output_folder=output_folder,
        compare=compare,
        api_key=api_key,
        base_url=base_url,
        humanizer_api_key=api_key,
        humanizer_model_name=model,
        humanizer_base_url=base_url,
        humanizer_verbose=verbose
    )

if __name__ == "__main__":
    app()
