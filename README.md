# ai2human

Describe your project here.

## Humanize Text with Langchain

1. Install dependencies (if not already):
   ```bash
   pip install langchain langchain-openai python-dotenv
   ```
2. Set your OpenAI API key in a `.env` file:
   ```env
   OPENAI_API_KEY="your_openai_api_key_here"
   ```
3. Run the humanization script:
   ```bash
   python src/ai2human/humanize.py
   ```

You can also import and use the `iterative_humanize_text` function in your own code:

```python
from ai2human.humanize import iterative_humanize_text

final_text, history = iterative_humanize_text("Your text here", num_iterations=2)
print(final_text)
```
