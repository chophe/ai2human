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

## Running Scripts with Rye

This project is Rye-enabled for easy script execution. After installing dependencies and setting up your `.env`, you can run all main tools using short aliases:

- `rye run humanize --help`
- `rye run humanize2 --help`
- `rye run humanize_adv --help`
- `rye run ai_detector --help`
- `rye run ai_detector_adv --help`

These aliases are defined in the `[tool.rye.scripts]` section of `pyproject.toml` and map to the main scripts in `src/ai2human/`.

You can pass any CLI arguments as usual, for example:

```sh
rye run humanize2 --iterations 2 --input "Your text here"
rye run humanize_adv --style conversational --input "Some text"
```

## Getting Started / Usage Examples

### Basic Humanization (`humanize.py`)

**With Rye:**

```sh
rye run humanize --help
rye run humanize --iterations 2 --input "Your text here"
```

**Direct Python (also works):**

```sh
python src/ai2human/humanize.py --help
```

### Class-based Humanizer (`humanize2.py`)

**With Rye:**

```sh
rye run humanize2 --help
rye run humanize2 --iterations 2 --input "Your text here"
```

**Direct Python (also works):**

```sh
python src/ai2human/humanize2.py --help
```

### Advanced Humanizer (`humanize-adv.py`)

**With Rye:**

```sh
rye run humanize_adv --help
rye run humanize_adv --style conversational --input "Some text"
```

**Direct Python (also works):**

```sh
python src/ai2human/humanize-adv.py --help
```

This project is ideal for developers, writers, and content creators looking to bridge the gap between machine-generated text and human expression.
