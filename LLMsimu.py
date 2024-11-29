import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import json
from typing import List, Dict
import sys
torch.manual_seed(42)
random.seed(42)


class LLMManager:
    """Manages interactions with both API-based and open-source LLMs."""

    def __init__(self, model_name: str, api_key: str = None, open_source: bool = False):
        self.model_name = model_name
        self.api_key = api_key
        self.open_source = open_source

        if self.open_source:
            # Load open-source model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

    def generate_response(self, prompt: str) -> str:
        try:
            if self.open_source:
   
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512 
                ).to("cuda")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,  
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
   
                openai.api_key = self.api_key
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are participating in a group discussion."},
                        {"role": "user", "content": prompt},
                    ]
                )["choices"][0]["message"]["content"]
        except torch.cuda.OutOfMemoryError:
            print("CUDA Out of Memory. Consider reducing max_new_tokens or using CPU offloading.")
            torch.cuda.empty_cache()
            return "Error: Unable to generate response due to memory constraints."
        return response.strip()





class DiscussionSimulator:


    def __init__(self, llms: List[LLMManager]):
        self.llms = llms

    def simulate_discussion(
        self, task_description: str, roles: List[str] = None, max_rounds: int = 5
    ) -> Dict:
        """
        Simulates a group discussion.
        Args:
            task_description: Description of the discussion topic or task.
            roles: Roles assigned to each LLM (e.g., leader, intern). If None, all LLMs are neutral.
            max_rounds: Maximum number of discussion rounds.
        Returns:
            A dictionary containing the discussion transcript and final output.
        """
        transcript = []
        context = task_description

        if roles:
            for llm, role in zip(self.llms, roles):
                role_prompt = f"You are assigned the role of {role}. {task_description}"
                context = f"{role_prompt}\n{context}"

        for round_num in range(max_rounds):
            round_transcript = {"round": round_num + 1, "responses": []}
            print(f"\n--- Round {round_num + 1} ---")

            for llm_idx, llm in enumerate(self.llms):
                prompt = f"Current discussion context: {context}\nWhat is your input for this round?"
                response = llm.generate_response(prompt)
                context += f"\nLLM-{llm_idx + 1}: {response}"
                round_transcript["responses"].append({"LLM": f"LLM-{llm_idx + 1}", "response": response})

            transcript.append(round_transcript)

        return {"transcript": transcript, "final_output": context}



def run_experiment_1(simulator):
    """Run Neutral Group Discussion Experiment."""
    print("\nRunning Experiment 1: Neutral Group Discussion")
    task = "Categorize the following text data into themes: 'The product is amazing, but the service is slow.'"
    results = simulator.simulate_discussion(task_description=task, max_rounds=3)
    print(json.dumps(results, indent=2))
    with open("discussion_transcript_experiment_1.json", "w") as f:
        json.dump(results, f, indent=2)


def run_experiment_2(simulator):
    """Run Hierarchical Roles Experiment."""
    print("\nRunning Experiment 2: Hierarchical Roles Discussion")
    roles = ["leader", "intern", "neutral"]
    task_with_roles = "Discuss and decide the top 3 most important themes for customer feedback analysis."
    results = simulator.simulate_discussion(task_description=task_with_roles, roles=roles, max_rounds=3)
    print(json.dumps(results, indent=2))
    with open("discussion_transcript_experiment_2.json", "w") as f:
        json.dump(results, f, indent=2)



def thematic_coding(llms: List[LLMManager], transcript: str) -> Dict[str, List[str]]:

    codes = {}
    for idx, llm in enumerate(llms):
        prompt = f"Analyze the following transcript and provide thematic codes for it:\n\n{transcript}"
        response = llm.generate_response(prompt)
        codes[f"LLM-{idx + 1}"] = response.split(", ") 
    return codes


def simulate_coding_discussion(llms: List[LLMManager], codes: Dict[str, List[str]]) -> Dict:

    context = "Here are the codes proposed by each LLM:\n"
    for llm, llm_codes in codes.items():
        context += f"{llm}: {', '.join(llm_codes)}\n"

    context += "\nDiscuss and agree on a final set of thematic codes for the transcript."

    discussion_transcript = []
    for round_num in range(3):  
        round_responses = []
        for idx, llm in enumerate(llms):
            prompt = f"Current discussion context:\n{context}\n\nWhat is your contribution to the discussion?"
            response = llm.generate_response(prompt)
            round_responses.append(f"LLM-{idx + 1}: {response}")
            context += f"\nLLM-{idx + 1}: {response}"
        discussion_transcript.append({"round": round_num + 1, "responses": round_responses})

    final_codes = discussion_transcript[-1]["responses"][-1].split(": ", 1)[-1]
    return {"final_codes": final_codes.split(", "), "discussion_transcript": discussion_transcript}


def run_thematic_analysis(simulator, transcript_path: str):
    """Run thematic analysis: coding and discussion."""
    with open(transcript_path, "r") as f:
        transcript = f.read()
    print("\nStep 1: Thematic Coding by Each LLM")
    codes = thematic_coding(simulator.llms, transcript)
    print(json.dumps(codes, indent=2))

    print("\nStep 2: Simulating Discussion Among LLMs")
    discussion_results = simulate_coding_discussion(simulator.llms, codes)
    print("\nFinal Agreed Codes:")
    print(discussion_results["final_codes"])

    results = {
        "transcript": transcript,
        "initial_codes": codes,
        "discussion_results": discussion_results,
    }
    with open("thematic_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)


def load_api_keys(file_path: str) -> dict:
    api_keys = {}
    with open(file_path, "r") as f:
        for line in f:
            if ":" in line:
                model_name, api_key = line.strip().split(":", 1)
                api_keys[model_name] = api_key
    return api_keys


def main():
    api_keys = load_api_keys("keys.txt")

    llms = [
        LLMManager(model_name="gpt-4", api_key=api_keys.get("gpt-4")),  # API-based
        LLMManager(model_name="gpt-3.5-turbo", api_key=api_keys.get("gpt-3.5-turbo")),  # API-based
        LLMManager(model_name="meta-llama/Llama-2-70b-chat-hf", open_source=True),  # Llama 2
    ]

    simulator = DiscussionSimulator(llms)

    task = sys.argv[1] if len(sys.argv) > 1 else "analysis"

    if task == "1":
        run_experiment_1(simulator)
    elif task == "2":
        run_experiment_2(simulator)
    elif task == "analysis":
        run_thematic_analysis(simulator, "text.txt")
    else:
        print("Invalid task. Choose '1', '2', or 'analysis'.")


if __name__ == "__main__":
    main()




