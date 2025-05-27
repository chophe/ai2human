import os
import json
from datetime import datetime
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional
import hashlib
import textwrap


class AdvancedTextHumanizer:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4.1",
        use_chat_model: bool = True,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Advanced TextHumanizer with more features.

        Args:
            api_key: OpenAI API key
            model_name: Model to use
            use_chat_model: Whether to use chat model (recommended)
            base_url: Optional base URL for the OpenAI API
        """
        os.environ["OPENAI_API_KEY"] = api_key

        llm_params = {
            "temperature": 0.7,
            "model_name": model_name,
        }
        if base_url:
            llm_params["base_url"] = base_url

        if use_chat_model:
            self.llm = ChatOpenAI(**llm_params)
        else:
            self.llm = OpenAI(**llm_params)

        self.iteration_history = []
        self.memory = ConversationBufferMemory()

    def create_style_specific_prompts(
        self, style: str = "conversational"
    ) -> List[Dict[str, str]]:
        """
        Create style-specific humanization prompts.

        Args:
            style: The style to apply (conversational, professional, casual, academic)

        Returns:
            List of style-specific prompts
        """
        style_guidelines = {
            "conversational": {
                "tone": "friendly and approachable",
                "features": [
                    "contractions",
                    "personal pronouns",
                    "rhetorical questions",
                ],
                "avoid": ["jargon", "overly formal language"],
            },
            "professional": {
                "tone": "polished but warm",
                "features": [
                    "clear structure",
                    "confident language",
                    "appropriate formality",
                ],
                "avoid": ["slang", "overly casual expressions"],
            },
            "casual": {
                "tone": "relaxed and informal",
                "features": [
                    "colloquialisms",
                    "humor where appropriate",
                    "personal anecdotes",
                ],
                "avoid": ["stiffness", "technical jargon"],
            },
            "academic": {
                "tone": "scholarly but accessible",
                "features": [
                    "clear explanations",
                    "logical flow",
                    "evidence-based statements",
                ],
                "avoid": ["unnecessary complexity", "pretentious language"],
            },
        }

        guidelines = style_guidelines.get(style, style_guidelines["conversational"])

        prompts = [
            {
                "description": f"Apply {style} style",
                "template": f"""
                Rewrite this text in a {guidelines['tone']} tone.
                Include: {', '.join(guidelines['features'])}
                Avoid: {', '.join(guidelines['avoid'])}
                
                Text to rewrite:
                {{text}}
                
                Rewritten version:
                """,
            },
            {
                "description": "Add human touches",
                "template": """
                Enhance this text with human elements:
                - Add subtle imperfections that humans naturally make
                - Include thought processes (e.g., "I think", "It seems like")
                - Use varied paragraph lengths
                - Add appropriate pauses or emphasis
                
                Current text:
                {text}
                
                Enhanced version:
                """,
            },
            {
                "description": "Context-aware refinement",
                "template": """
                Refine this text considering the context and audience:
                - Ensure the tone matches throughout
                - Add relevant examples or analogies
                - Make abstract concepts more concrete
                - Use active voice where appropriate
                
                Current text:
                {text}
                
                Context: {context}
                
                Refined version:
                """,
            },
        ]

        return prompts

    def analyze_text_formality(self, text: str) -> Dict[str, any]:
        """
        Analyze the formality level of the input text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with formality analysis
        """
        analysis_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Analyze the formality level of this text and return a JSON response with:
            1. formality_score (1-10, where 1 is very informal and 10 is very formal)
            2. detected_style (conversational/professional/casual/academic)
            3. key_indicators (list of features that indicate the formality level)
            
            Text to analyze:
            {text}
            
            JSON Response:
            """,
        )

        chain = LLMChain(llm=self.llm, prompt=analysis_prompt)

        try:
            response = chain.run(text=text)
            return json.loads(response)
        except Exception:  # Be more specific if possible, e.g., json.JSONDecodeError
            return {
                "formality_score": 5,
                "detected_style": "neutral",
                "key_indicators": ["Unable to parse analysis"],
            }

    def humanize_with_context(
        self,
        input_text: str,
        context: Optional[str] = None,
        target_style: str = "conversational",
        preserve_facts: bool = True,
        max_iterations: int = 3,
    ) -> Dict[str, any]:
        """
        Humanize text with context awareness and fact preservation.

        Args:
            input_text: Text to humanize
            context: Additional context about the text
            target_style: Target writing style
            preserve_facts: Whether to preserve factual information
            max_iterations: Maximum number of iterations

        Returns:
            Dictionary with humanized text and metadata
        """
        original_analysis = self.analyze_text_formality(input_text)
        text_hash = hashlib.md5(input_text.encode()).hexdigest()[:8]

        results = {
            "original_text": input_text,
            "original_analysis": original_analysis,
            "iterations": [],
            "final_text": "",
            "metadata": {
                "text_hash": text_hash,
                "timestamp": datetime.now().isoformat(),
                "target_style": target_style,
                "context": context,
            },
        }

        current_text = input_text
        prompts_to_run = self.create_style_specific_prompts(target_style)

        for i in range(min(max_iterations, len(prompts_to_run))):
            prompt_info = prompts_to_run[i]
            current_prompt_template_str = prompt_info["template"]

            template_vars_for_run = {"text": current_text}
            prompt_input_vars_list = ["text"]

            if "{context}" in current_prompt_template_str:
                prompt_input_vars_list.append("context")
                template_vars_for_run["context"] = context if context else ""

            prompt_template = PromptTemplate(
                input_variables=prompt_input_vars_list,
                template=current_prompt_template_str,
            )

            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            current_text = chain.run(template_vars_for_run)

            if preserve_facts:
                current_text = self._verify_facts_preserved(input_text, current_text)

            results["iterations"].append(
                {
                    "iteration": i + 1,
                    "description": prompt_info["description"],
                    "text": current_text,
                }
            )

        results["final_text"] = current_text
        results["final_analysis"] = self.analyze_text_formality(current_text)

        return results

    def _verify_facts_preserved(self, original: str, humanized: str) -> str:
        """
        Verify that key facts are preserved in the humanized version.

        Args:
            original: Original text
            humanized: Humanized text

        Returns:
            Corrected text if needed
        """
        verification_prompt = PromptTemplate(
            input_variables=["original", "humanized"],
            template="""
            Compare these two texts and ensure all factual information from the original is preserved in the humanized version.
            If any facts are missing or altered, correct the humanized version.
            
            Original text:
            {original}
            
            Humanized text:
            {humanized}
            
            Corrected version (or return the humanized text if all facts are preserved):
            """,
        )

        chain = LLMChain(llm=self.llm, prompt=verification_prompt)
        return chain.run(original=original, humanized=humanized)

    def batch_humanize(
        self, texts: List[str], style: str = "conversational", save_results: bool = True
    ) -> List[Dict[str, any]]:
        """
        Humanize multiple texts in batch.

        Args:
            texts: List of texts to humanize
            style: Target style for all texts
            save_results: Whether to save results to file

        Returns:
            List of humanization results
        """
        results_list = []
        total_cost = 0.0
        total_tokens_used = 0

        print(f"Processing {len(texts)} texts...")

        for idx, text_item in enumerate(texts):
            print(f"Processing text {idx + 1}/{len(texts)}")

            with get_openai_callback() as cb:
                result_item = self.humanize_with_context(
                    input_text=text_item, target_style=style
                )
                total_tokens_used += cb.total_tokens
                total_cost += cb.total_cost

            result_item["metadata"]["batch_index"] = idx
            result_item["metadata"]["tokens_used_for_item"] = cb.total_tokens
            result_item["metadata"]["cost_for_item"] = cb.total_cost

            results_list.append(result_item)

        batch_summary = {
            "total_texts": len(texts),
            "total_tokens": total_tokens_used,
            "total_cost": total_cost,
            "average_tokens_per_text": total_tokens_used / len(texts) if texts else 0,
            "timestamp": datetime.now().isoformat(),
        }

        if save_results:
            filename = (
                f"humanization_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    {"summary": batch_summary, "results": results_list},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"Results saved to {filename}")

        print(f"Batch processing complete!")
        print(f"Total cost: ${total_cost:.4f}")

        return results_list

    def create_custom_pipeline(
        self, pipeline_steps: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Create a custom humanization pipeline with user-defined steps.

        Args:
            pipeline_steps: List of dictionaries with 'description' and 'instructions'

        Returns:
            List of formatted prompts
        """
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

    def interactive_humanize(self, initial_text: str) -> str:
        """
        Interactive humanization where user can choose options at each step.

        Args:
            initial_text: Text to humanize

        Returns:
            Final humanized text
        """
        current_text = initial_text
        # self.iteration_history is class-level, clear or manage if needed per call
        # For this example, it will append to the existing history.
        # If fresh history is needed per call: self.iteration_history = [{"text": current_text, "instruction": "Initial text"}]

        # Add initial state to history for undo purposes if not handled elsewhere
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
                # Ensure there's something to undo beyond the initial state
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
                else:  # Covers empty or only initial state
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
                # It's good practice to handle potential errors from chain.run
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
                ]:  # Invalid choice not covered by undo/finish
                    print("Invalid choice, please try again.")

        return current_text


# Utility functions
def compare_texts_side_by_side(original: str, humanized: str, width: int = 40):
    """
    Display original and humanized texts side by side for comparison.

    Args:
        original: Original text
        humanized: Humanized text
        width: Width of each column
    """
    original_lines = textwrap.wrap(original, width=width)
    humanized_lines = textwrap.wrap(humanized, width=width)

    max_lines = max(len(original_lines), len(humanized_lines))

    header = f"{'ORIGINAL':<{width}} | {'HUMANIZED':<{width}}"
    separator = "=" * (width * 2 + 3)  # Adjusted for " | "

    print("" + separator)
    print(header)
    print(separator)

    for i in range(max_lines):
        orig_line = original_lines[i] if i < len(original_lines) else ""
        human_line = humanized_lines[i] if i < len(humanized_lines) else ""
        print(f"{orig_line:<{width}} | {human_line:<{width}}")

    print(separator)


# Example usage with advanced features
def advanced_example():
    # Initialize the advanced humanizer
    # IMPORTANT: Replace "your-openai-api-key-here" with your actual OpenAI API key
    # or ensure it's set as an environment variable.
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    if api_key == "your-openai-api-key-here":
        print("Warning: Using placeholder API key. Please set your OpenAI API key.")

    # Example: Replace "your-base-url-here" with your actual base URL if needed
    base_url = os.getenv("OPENAI_BASE_URL", None)  # "your-base-url-here"

    humanizer = AdvancedTextHumanizer(api_key=api_key, base_url=base_url)

    # Example texts with different styles
    examples = {
        "technical": """
Dear Dr. Hamid Nick and the Selection Committee,

I am thrilled to apply for the PhD position at DTU Offshore, focusing on the bio-geo-chemistry of Underground Hydrogen Storage (UHS), Carbon Capture and Storage (CCS), and their environmental impacts (Job ID: 5163). With a Master’s degree in Renewable Energy Engineering and over 15 years of experience in computational modeling and sustainable energy solutions, I am eager to contribute my technical expertise and passion for environmental innovation to your esteemed research team.

My academic journey has been driven by a deep curiosity about solving complex engineering challenges. During my Master’s at the Materials and Energy Research Center, I conducted a thesis titled “Numerical Study of the Accuracy of the Three-Bowl Anemometer under Diagonal Flows.” Using ANSYS Fluent, I developed and validated airflow models, optimizing mesh configurations to reduce measurement errors by up to 5%. This experience not only honed my skills in Computational Fluid Dynamics (CFD) but also sparked my interest in interdisciplinary research, particularly in areas like energy storage and environmental sustainability that align with the goals of this PhD program.

Professionally, I have spent 15 years as a Renewable Energy Engineer at Nik Andish Kaveh Company, where I designed solar photovoltaic systems and leveraged tools like MATLAB and PVsyst to enhance efficiency. A highlight of my career was reducing installation costs by 20% while maintaining a 95% client satisfaction rate—a testament to my ability to balance technical rigor with practical impact. Additionally, as the founder of EverClean, a startup developing biodegradable products, I led laboratory research to create compostable materials, deepening my understanding of sustainable innovation and cross-disciplinary collaboration.

The opportunity to join DTU Offshore excites me because it bridges my expertise in computational modeling with the pressing need to address bio-geo-chemical challenges in UHS and CCS. I am particularly drawn to the project’s focus on microbial interactions and CO2 leakage assessment, as these areas combine my technical skills with my commitment to mitigating climate change. With proficiency in ANSYS, MATLAB, and Python, and a proven ability to work in diverse teams, I am confident in my capacity to contribute meaningfully to your research.

Thank you for considering my application. I would be delighted to discuss how my background and enthusiasm can support DTU’s mission to advance the energy transition. I look forward to the possibility of joining your vibrant research community.

Sincerely,  
Dena Milani  
[Email: dena.milani@gmail.com | Phone: +989128137344]
        """,
        "corporate": """
        Our Q3 performance metrics indicate a 23% YoY growth in revenue streams, 
        primarily driven by strategic market penetration initiatives and optimized 
        operational efficiencies. The EBITDA margin expanded by 340 basis points, 
        reflecting our commitment to sustainable profitability.
        """,
        "academic": """
        The empirical analysis conducted on a sample size of n=1,247 participants 
        revealed statistically significant correlations (p<0.001) between social 
        media usage patterns and self-reported anxiety levels. The regression model 
        explained 47% of the variance in the dependent variable.
        """,
    }

    for text_type, text_content in examples.items():
        print(f"{'='*60}")
        print(f"Processing {text_type.upper()} text")
        print(f"{'='*60}")

        style_map = {
            "technical": "conversational",
            "corporate": "professional",
            "academic": "academic",
        }

        result = humanizer.humanize_with_context(
            input_text=text_content,
            target_style=style_map.get(text_type, "conversational"),
            context=f"This is a {text_type} document that needs to be more accessible.",
            preserve_facts=True,
        )

        print(
            f"Original Formality Score: {result['original_analysis'].get('formality_score', 'N/A')}/10"
        )
        print(
            f"Final Formality Score: {result['final_analysis'].get('formality_score', 'N/A')}/10"
        )

        compare_texts_side_by_side(text_content, result["final_text"])

    print(f"{'='*60}")
    print("BATCH PROCESSING EXAMPLE")
    print(f"{'='*60}")

    batch_texts = [
        """Dear Dr. Hamid Nick and the Selection Committee,

I am thrilled to apply for the PhD position at DTU Offshore, focusing on the bio-geo-chemistry of Underground Hydrogen Storage (UHS), Carbon Capture and Storage (CCS), and their environmental impacts (Job ID: 5163). With a Master’s degree in Renewable Energy Engineering and over 15 years of experience in computational modeling and sustainable energy solutions, I am eager to contribute my technical expertise and passion for environmental innovation to your esteemed research team.

My academic journey has been driven by a deep curiosity about solving complex engineering challenges. During my Master’s at the Materials and Energy Research Center, I conducted a thesis titled “Numerical Study of the Accuracy of the Three-Bowl Anemometer under Diagonal Flows.” Using ANSYS Fluent, I developed and validated airflow models, optimizing mesh configurations to reduce measurement errors by up to 5%. This experience not only honed my skills in Computational Fluid Dynamics (CFD) but also sparked my interest in interdisciplinary research, particularly in areas like energy storage and environmental sustainability that align with the goals of this PhD program.

Professionally, I have spent 15 years as a Renewable Energy Engineer at Nik Andish Kaveh Company, where I designed solar photovoltaic systems and leveraged tools like MATLAB and PVsyst to enhance efficiency. A highlight of my career was reducing installation costs by 20% while maintaining a 95% client satisfaction rate—a testament to my ability to balance technical rigor with practical impact. Additionally, as the founder of EverClean, a startup developing biodegradable products, I led laboratory research to create compostable materials, deepening my understanding of sustainable innovation and cross-disciplinary collaboration.

The opportunity to join DTU Offshore excites me because it bridges my expertise in computational modeling with the pressing need to address bio-geo-chemical challenges in UHS and CCS. I am particularly drawn to the project’s focus on microbial interactions and CO2 leakage assessment, as these areas combine my technical skills with my commitment to mitigating climate change. With proficiency in ANSYS, MATLAB, and Python, and a proven ability to work in diverse teams, I am confident in my capacity to contribute meaningfully to your research.

Thank you for considering my application. I would be delighted to discuss how my background and enthusiasm can support DTU’s mission to advance the energy transition. I look forward to the possibility of joining your vibrant research community.

Sincerely,  
Dena Milani  
[Email: dena.milani@gmail.com | Phone: +989128137344]""",
        "Implementation of new protocols will commence immediately.",
        "Statistical analysis reveals noteworthy trends in consumer behavior.",
    ]

    humanizer.batch_humanize(
        batch_texts, style="casual", save_results=True
    )  # Results are printed within batch_humanize

    print(f"{'='*60}")
    print("INTERACTIVE MODE EXAMPLE (Commented out by default)")
    print(f"{'='*60}")

    interactive_text_example = """
    The corporation's strategic initiatives have yielded substantial returns 
    on investment, demonstrating the efficacy of our methodological approach 
    to market expansion.
    """
    print("To try interactive mode, uncomment the lines in advanced_example()")
    # try:
    #     final_interactive_text = humanizer.interactive_humanize(interactive_text_example)
    #     print("Final interactive result:", final_interactive_text)
    # except Exception as e:
    #     print(f"Could not run interactive example: {e}")


# Custom pipeline example
def custom_pipeline_example():
    api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    if api_key == "your-openai-api-key-here":
        print("Warning: Using placeholder API key for custom_pipeline_example.")

    # Example: Replace "your-base-url-here" with your actual base URL if needed
    base_url = os.getenv("OPENAI_BASE_URL", None)  # "your-base-url-here"
    humanizer = AdvancedTextHumanizer(api_key=api_key, base_url=base_url)

    custom_steps = [
        {
            "description": "Add storytelling elements",
            "instructions": "Rewrite this text as if you're telling a story to a friend. Add narrative elements and make it engaging.",
        },
        {
            "description": "Insert personal touches",
            "instructions": "Add personal pronouns, rhetorical questions, and make the reader feel involved.",
        },
        {
            "description": "Polish with humor",
            "instructions": "Where appropriate, add light humor or witty observations while maintaining the core message.",
        },
    ]

    custom_prompts = humanizer.create_custom_pipeline(custom_steps)

    text_to_process = """
   Dear Dr. Hamid Nick and the Selection Committee,

I am thrilled to apply for the PhD position at DTU Offshore, focusing on the bio-geo-chemistry of Underground Hydrogen Storage (UHS), Carbon Capture and Storage (CCS), and their environmental impacts (Job ID: 5163). With a Master's degree in Renewable Energy Engineering and over 15 years of experience in computational modeling and sustainable energy solutions, I am eager to contribute my technical expertise and passion for environmental innovation to your esteemed research team.

My academic journey has been driven by a deep curiosity about solving complex engineering challenges. During my Master's at the Materials and Energy Research Center, I conducted a thesis titled "Numerical Study of the Accuracy of the Three-Bowl Anemometer under Diagonal Flows." Using ANSYS Fluent, I developed and validated airflow models, optimizing mesh configurations to reduce measurement errors by up to 5%. This experience not only honed my skills in Computational Fluid Dynamics (CFD) but also sparked my interest in interdisciplinary research, particularly in areas like energy storage and environmental sustainability that align with the goals of this PhD program.

Professionally, I have spent 15 years as a Renewable Energy Engineer at Nik Andish Kaveh Company, where I designed solar photovoltaic systems and leveraged tools like MATLAB and PVsyst to enhance efficiency. A highlight of my career was reducing installation costs by 20% while maintaining a 95% client satisfaction rate—a testament to my ability to balance technical rigor with practical impact. Additionally, as the founder of EverClean, a startup developing biodegradable products, I led laboratory research to create compostable materials, deepening my understanding of sustainable innovation and cross-disciplinary collaboration.

The opportunity to join DTU Offshore excites me because it bridges my expertise in computational modeling with the pressing need to address bio-geo-chemical challenges in UHS and CCS. I am particularly drawn to the project's focus on microbial interactions and CO2 leakage assessment, as these areas combine my technical skills with my commitment to mitigating climate change. With proficiency in ANSYS, MATLAB, and Python, and a proven ability to work in diverse teams, I am confident in my capacity to contribute meaningfully to your research.

Thank you for considering my application. I would be delighted to discuss how my background and enthusiasm can support DTU's mission to advance the energy transition. I look forward to the possibility of joining your vibrant research community.

Sincerely,  
Dena Milani  
[Email: dena.milani@gmail.com | Phone: +989128137344]
       """

    print(f"{'='*60}")
    print("CUSTOM PIPELINE EXAMPLE")
    print(f"Original text for custom pipeline: {text_to_process}")
    print(f"{'='*60}")

    current_text_pipeline = text_to_process
    for prompt_config in custom_prompts:
        template = PromptTemplate(
            input_variables=["text"], template=prompt_config["template"]
        )
        chain = LLMChain(llm=humanizer.llm, prompt=template)
        try:
            current_text_pipeline = chain.run(text=current_text_pipeline)
            print(f"After {prompt_config['description']}:")
            print(current_text_pipeline)
        except Exception as e:
            print(
                f"Error during custom pipeline step '{prompt_config['description']}': {e}"
            )
            break  # Stop pipeline on error


if __name__ == "__main__":
    print("Running Advanced Humanizer Examples...")
    advanced_example()

    # print("Running Custom Pipeline Example...")
    # custom_pipeline_example()
    print("Examples complete. Uncomment specific examples in main block to run them.")
