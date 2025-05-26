import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in your .env file or environment."
    )

LLM_MODEL = "gpt-4o"  # Or "gpt-3.5-turbo", "gpt-4-turbo" etc.
TEMPERATURE = (
    0.7  # Controls randomness: 0.0 for more deterministic, 1.0 for more creative
)

# Langchain Setup
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    base_url=OPENAI_BASE_URL,
)

humanize_prompt_template_str = """
You are a skilled text editor specializing in making content sound more human, natural, and engaging.
Your goal is to revise the provided text to achieve this.

Consider the following aspects for humanization:
- **Tone:** Make it more conversational, friendly, and approachable.
- **Clarity:** Simplify complex sentences or jargon if appropriate, without losing essential meaning.
- **Flow:** Ensure smooth transitions and logical progression of ideas.
- **Word Choice:** Use more common and relatable vocabulary. Avoid overly formal or robotic language.
- **Sentence Structure:** Vary sentence length and structure to make it more dynamic.
- **Contractions:** Feel free to use contractions (e.g., "it's" instead of "it is", "you're" instead of "you are").
- **Empathy/Relatability (Subtle):** If appropriate for the context, subtly weave in elements that make the text more relatable.

**IMPORTANT:**
- Do NOT add new factual information or change the core meaning of the text.
- Focus solely on rephrasing and restyling the existing content.
- If the text is already very human-like, you can make minor refinements or indicate that minimal changes are needed.

Here is the text to humanize:
---
{text_to_humanize}
---

Please provide only the humanized version of the text.
"""

humanize_prompt = ChatPromptTemplate.from_template(humanize_prompt_template_str)
output_parser = StrOutputParser()
humanize_chain = humanize_prompt | llm | output_parser


def iterative_humanize_text(
    initial_text: str, num_iterations: int = 3
) -> tuple[str, list[dict]]:
    """
    Iteratively humanizes text using an LLM chain.

    Args:
        initial_text: The text to start humanizing.
        num_iterations: The number of times to run the humanization prompt.

    Returns:
        A tuple containing:
            - The final humanized text.
            - A list of dictionaries, where each dictionary records the text at each iteration.
    """
    current_text = initial_text
    history = [{"iteration": 0, "text": current_text, "type": "Original"}]

    print(f"Original Text:\n{current_text}\n{'-'*30}")

    for i in range(1, num_iterations + 1):
        print(f"Processing Iteration {i}...")
        try:
            humanized_output = humanize_chain.invoke({"text_to_humanize": current_text})
            if humanized_output.strip().lower() == current_text.strip().lower():
                print(f"Iteration {i}: No significant change detected. Stopping early.")
                break
            current_text = humanized_output
            history.append({"iteration": i, "text": current_text, "type": "Humanized"})
            print(f"Humanized Text (Iteration {i}):\n{current_text}\n{'-'*30}")
        except Exception as e:
            print(f"Error during iteration {i}: {e}")
            break
    return current_text, history


if __name__ == "__main__":
    sample_text_1 = """
    The system's operational parameters have been optimized for enhanced performance efficiency.
    User interaction protocols require adherence to predefined procedural guidelines to ensure data integrity.
    It is imperative that all personnel complete the mandatory training module prior to system access.
    """

    sample_text_2 = """
    Conclusion: The conducted experiment yielded results indicative of a positive correlation
    between the independent variable and the dependent variable. Statistical significance was observed.
    Further investigation is warranted to explore underlying mechanisms.
    """

    print("--- Humanizing Sample Text 1 ---")
    final_text_1, history_1 = iterative_humanize_text(sample_text_1, num_iterations=2)
    print("\n=== Final Humanized Text 1 ===")
    print(final_text_1)

    print("\n\n--- Humanizing Sample Text 2 ---")
    final_text_2, history_2 = iterative_humanize_text(sample_text_2, num_iterations=3)
    print("\n=== Final Humanized Text 2 ===")
    print(final_text_2)
