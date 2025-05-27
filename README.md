# AI Text Humanizer

**Description:**

This project provides a suite of Python tools designed to transform text, making it sound more natural, engaging, and human-like. Utilizing the power of Large Language Models (LLMs) through the Langchain library and OpenAI's API, these tools can take formal, robotic, or AI-generated text and refine it to better suit various communication styles.

**Key Features:**

- **Iterative Humanization:** Applies multiple refinement passes to progressively enhance the text's human-like qualities.
- **Style-Specific Adaptation:** Tailors the output to different tones and styles, including conversational, professional, casual, and academic.
- **Contextual Awareness:** Can consider additional context to ensure the humanized text is relevant and appropriate.
- **Fact Preservation:** Includes mechanisms to verify that key factual information from the original text is retained in the humanized version.
- **Formality Analysis:** Assesses the formality level of text to guide the humanization process.
- **Batch Processing:** Efficiently humanizes multiple text documents at once, with cost and token tracking.
- **Interactive Mode:** Allows for step-by-step humanization with user guidance and the ability to choose refinement options or provide custom instructions.
- **Customizable Pipelines:** Supports the creation of unique humanization workflows with user-defined processing steps.
- **OpenAI Model Integration:** Leverages models like GPT-4 for high-quality text generation and refinement.

**Core Functionality:**

The project offers several approaches to text humanization:

1.  A basic iterative humanizer (`src/ai2human/humanize.py`) that applies a general humanization prompt multiple times.
2.  A more structured class-based humanizer (`src/ai2human/humanize2.py`) that uses a sequence of specific prompts for iterative improvement and includes cost tracking.
3.  An advanced humanizer (`src/ai2human/humanize-adv.py`) with sophisticated features like style targeting, formality analysis, fact verification, batch operations, and an interactive refinement loop.

**Potential Use Cases:**

- Improving the readability and engagement of AI-generated content.
- Making chatbot responses more natural and less robotic.
- Softening formal documents or emails for a wider audience.
- Enhancing marketing copy or creative writing.
- Assisting in the editing process to achieve a specific tone or style.

**Technology Stack:**

- Python
- Langchain
- OpenAI API (gpt-3.5-turbo, gpt-4, gpt-4o, etc.)
- Dotenv for environment management

This project is ideal for developers, writers, and content creators looking to bridge the gap between machine-generated text and human expression.

## Getting Started / Usage Examples

### Basic Humanization (`humanize.py`)

1.  **Install dependencies:**
    Ensure you have `langchain`, `langchain-openai`, and `python-dotenv` installed. If using `rye` (as per your custom instructions):
    ```bash
    rye add langchain langchain-openai python-dotenv
    rye sync
    ```
    Or with `uv`/`pip`:
    ```bash
    uv pip install langchain langchain-openai python-dotenv
    # or
    # pip install langchain langchain-openai python-dotenv
    ```
2.  **Set your OpenAI API key:**
    Create a `.env` file in the project root with your OpenAI API key:
    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    # Optional: If using a custom OpenAI base URL
    # OPENAI_BASE_URL="your_custom_base_url_here"
    ```
3.  **Run the script:**

    ```bash
    python src/ai2human/humanize.py
    ```

    This will run the example texts defined in the script.

4.  **Use in your code:**

    ```python
    from ai2human.humanize import iterative_humanize_text

    sample_text = "The system's operational parameters have been optimized."
    final_text, history = iterative_humanize_text(sample_text, num_iterations=2)
    print("Final Humanized Text:")
    print(final_text)
    ```

### Class-based Humanizer (`humanize2.py`)

1.  **Setup:** (Dependencies and `.env` file as above)
2.  **Run the script:**

    ```bash
    python src/ai2human/humanize2.py
    ```

    This will process the example text in the `main()` function and save a `humanization_history.txt`.

3.  **Use in your code:**

    ```python
    from ai2human.humanize2 import TextHumanizer

    humanizer = TextHumanizer() # API key from .env
    formal_text = "Dear Dr. Hamid Nick and the Selection Committee, I am thrilled to apply..." # Your long formal text
    humanized_text = humanizer.humanize_text(formal_text, iterations=3, verbose=True)
    print("\\nFINAL RESULT:")
    print(humanized_text)

    # Get history
    history = humanizer.get_iteration_history()
    for item in history:
        print(f"Iteration {item['iteration']}: {item['description']}")
        print(item['text'])
    ```

### Advanced Humanizer (`humanize-adv.py`)

1.  **Setup:** (Dependencies and `.env` file as above)
2.  **Run the example:**

    ```bash
    python src/ai2human/humanize-adv.py
    ```

    This script demonstrates various advanced features like style targeting, batch processing, and provides a structure for interactive humanization.

3.  **Use in your code (Example - Style specific):**

    ```python
    import os
    from ai2human.humanize_adv import AdvancedTextHumanizer

    api_key = os.getenv("OPENAI_API_KEY")
    # base_url = os.getenv("OPENAI_BASE_URL") # Optional

    humanizer = AdvancedTextHumanizer(api_key=api_key) #, base_url=base_url)

    technical_text = "The device exhibits suboptimal performance under high-stress conditions."
    result = humanizer.humanize_with_context(
        input_text=technical_text,
        target_style="conversational",
        context="This is a technical report snippet for a general audience."
    )
    print("Original:", result['original_text'])
    print("Humanized:", result['final_text'])
    print("Original Analysis:", result['original_analysis'])
    print("Final Analysis:", result['final_analysis'])
    ```

This project is ideal for developers, writers, and content creators looking to bridge the gap between machine-generated text and human expression.
