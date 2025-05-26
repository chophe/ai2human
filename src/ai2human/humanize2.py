import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from typing import List, Dict

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()


class TextHumanizer:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4.1",
        api_base_url: str = None,
    ):
        """
        Initialize the TextHumanizer with OpenAI API key, model, and optional API base URL.
        Args:
            api_key: OpenAI API key (optional, will use env if not provided)
            model_name: Model to use (default: gpt-3.5-turbo)
            api_base_url: Base URL for OpenAI API (optional, will use env if not provided)
        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment."
            )
        if api_base_url is None:
            api_base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = api_key
        llm_kwargs = {"temperature": 0.7, "model": model_name}
        if api_base_url:
            llm_kwargs["base_url"] = api_base_url
        self.llm = ChatOpenAI(openai_api_key=api_key, **llm_kwargs)
        self.iteration_history = []

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
        self, input_text: str, iterations: int = None, verbose: bool = True
    ) -> str:
        prompts = self.create_humanization_prompts()
        if iterations is None:
            iterations = len(prompts)
        else:
            iterations = min(iterations, len(prompts))
        current_text = input_text
        self.iteration_history = [
            {"iteration": 0, "text": input_text, "description": "Original text"}
        ]
        total_tokens = 0
        total_cost = 0
        for i in range(iterations):
            prompt_info = prompts[i]
            if verbose:
                print(f"\n{'='*50}")
                print(f"Iteration {i+1}: {prompt_info['description']}")
                print(f"{'='*50}")
            prompt_template = PromptTemplate(
                input_variables=["text"], template=prompt_info["template"]
            )
            chain = prompt_template | self.llm
            with get_openai_callback() as cb:
                response = chain.invoke({"text": current_text})
                current_text = response.content
                total_tokens += cb.total_tokens
                total_cost += cb.total_cost
            self.iteration_history.append(
                {
                    "iteration": i + 1,
                    "text": current_text,
                    "description": prompt_info["description"],
                }
            )
            if verbose:
                print(f"Tokens used: {cb.total_tokens}")
                print(f"Cost: ${cb.total_cost:.4f}")
        if verbose:
            print(f"\n{'='*50}")
            print(f"Total tokens used: {total_tokens}")
            print(f"Total cost: ${total_cost:.4f}")
            print(f"{'='*50}")
        return current_text

    def get_iteration_history(self) -> List[Dict]:
        return self.iteration_history

    def compare_versions(
        self, iteration_a: int = 0, iteration_b: int = -1
    ) -> Dict[str, str]:
        if not self.iteration_history:
            return {"error": "No iteration history available"}

        text_a = self.iteration_history[iteration_a]["text"]
        if not isinstance(text_a, str):
            text_a = str(text_a)

        text_b = self.iteration_history[iteration_b]["text"]
        if not isinstance(text_b, str):
            text_b = str(text_b)

        return {
            "version_a": {
                "iteration": iteration_a,
                "description": self.iteration_history[iteration_a]["description"],
                "text": text_a,
            },
            "version_b": {
                "iteration": iteration_b
                if iteration_b >= 0
                else len(self.iteration_history) + iteration_b,
                "description": self.iteration_history[iteration_b]["description"],
                "text": text_b,
            },
        }


def main():
    # Initialize the humanizer (API key loaded from env or .env)
    humanizer = TextHumanizer()
    formal_text = """
    Dear Dr. Hamid Nick and the Selection Committee,

I am thrilled to apply for the PhD position at DTU Offshore, focusing on the bio-geo-chemistry of Underground Hydrogen Storage (UHS), Carbon Capture and Storage (CCS), and their environmental impacts (Job ID: 5163). With a Master’s degree in Renewable Energy Engineering and over 15 years of experience in computational modeling and sustainable energy solutions, I am eager to contribute my technical expertise and passion for environmental innovation to your esteemed research team.

My academic journey has been driven by a deep curiosity about solving complex engineering challenges. During my Master’s at the Materials and Energy Research Center, I conducted a thesis titled “Numerical Study of the Accuracy of the Three-Bowl Anemometer under Diagonal Flows.” Using ANSYS Fluent, I developed and validated airflow models, optimizing mesh configurations to reduce measurement errors by up to 5%. This experience not only honed my skills in Computational Fluid Dynamics (CFD) but also sparked my interest in interdisciplinary research, particularly in areas like energy storage and environmental sustainability that align with the goals of this PhD program.

Professionally, I have spent 15 years as a Renewable Energy Engineer at Nik Andish Kaveh Company, where I designed solar photovoltaic systems and leveraged tools like MATLAB and PVsyst to enhance efficiency. A highlight of my career was reducing installation costs by 20% while maintaining a 95% client satisfaction rate—a testament to my ability to balance technical rigor with practical impact. Additionally, as the founder of EverClean, a startup developing biodegradable products, I led laboratory research to create compostable materials, deepening my understanding of sustainable innovation and cross-disciplinary collaboration.

The opportunity to join DTU Offshore excites me because it bridges my expertise in computational modeling with the pressing need to address bio-geo-chemical challenges in UHS and CCS. I am particularly drawn to the project’s focus on microbial interactions and CO2 leakage assessment, as these areas combine my technical skills with my commitment to mitigating climate change. With proficiency in ANSYS, MATLAB, and Python, and a proven ability to work in diverse teams, I am confident in my capacity to contribute meaningfully to your research.

Thank you for considering my application. I would be delighted to discuss how my background and enthusiasm can support DTU’s mission to advance the energy transition. I look forward to the possibility of joining your vibrant research community.

Sincerely,  
Dena Milani  
[Email: dena.milani@gmail.com | Phone: +989128137344]    """
    humanized_text = humanizer.humanize_text(formal_text, verbose=True)
    print("\n\nFINAL RESULT:")
    print("=" * 50)
    if hasattr(humanized_text, "content"):
        print(humanized_text.content)
    else:
        print(humanized_text)
    print("\n\nCOMPARISON:")
    print("=" * 50)
    comparison = humanizer.compare_versions()
    print(f"Original:\n{comparison['version_a']['text']}")
    print(f"\nFinal:\n{comparison['version_b']['text']}")
    history = humanizer.get_iteration_history()
    with open("humanization_history.txt", "w", encoding="utf-8") as f:
        for item in history:
            f.write(f"\nIteration {item['iteration']}: {item['description']}\n")
            f.write("-" * 50 + "\n")
            text_to_write = item["text"]
            if hasattr(text_to_write, "content"):
                text_to_write = text_to_write.content
            elif not isinstance(text_to_write, str):
                text_to_write = str(text_to_write)
            f.write(text_to_write + "\n")


if __name__ == "__main__":
    main()
