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
                print(response)
            else:
   
                openai.api_key = self.api_key
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are participating in a group discussion."},
                        {"role": "user", "content": prompt},
                    ]
                )["choices"][0]["message"]["content"]
                print(response)
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
                #Here I am making only summary of context
                prompt2 = f"Summarize main takeaways in one sentence: {response}"
                summary = llm.generate_response(prompt2)
                context += f"\nLLM-{llm_idx + 1}: {summary}"
                round_transcript["responses"].append({"LLM": f"LLM-{llm_idx + 1}", "response": response, "summary": summary})

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

def truncate_context(context, max_tokens=8000):
    """
    Truncates the context to fit within the token limit.
    Args:
        context (str): The full context text.
        max_tokens (int): The maximum number of tokens allowed (default: 8000).
    Returns:
        str: The truncated context, containing the last `max_tokens` tokens.
    """
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Compatible tokenizer
    tokens = tokenizer.encode(context)

    if len(tokens) > max_tokens:
        truncated_tokens = tokens[-max_tokens:]  # Keep only the most recent tokens
        return tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    return context


def thematic_coding(llms: List[LLMManager], transcript: str) -> Dict[str, List[str]]:

    codes = {}
    for idx, llm in enumerate(llms):
        prompt = f"Analyze the following transcript of interview on the topic of Interaction with Voice Assistants. Perform a thematic analysis like you would do as a qualitative researcher in social science, and provide the list of codes:\n\n{transcript}"
        prompt = truncate_context(prompt)
        response = llm.generate_response(prompt)
        codes[f"LLM-{idx + 1}"] = response 
    return codes

def truncate_context(context, max_tokens=8000):
    """
    Truncates context to fit within the token limit.
    Args:
        context (str): The entire context text.
        max_tokens (int): Maximum allowable tokens (default: 8000).
    Returns:
        str: Truncated context.
    """
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Compatible tokenizer
    tokens = tokenizer.encode(context)

    if len(tokens) > max_tokens:
        truncated_tokens = tokens[-max_tokens:]  # Keep the most recent tokens
        return tokenizer.decode(truncated_tokens)
    return context


def simulate_coding_discussion(llms: List[LLMManager], initial_codes: Dict[str, List[str]]) -> Dict:
    """
    Simulates a discussion among LLMs with a sliding context window and memory of each LLM's contributions.
    Args:
        llms: List of LLM managers.
        initial_codes: Initial codes generated by the LLMs.
    Returns:
        Dictionary with discussion transcript and final thematic codes.
    """
    context = "Here are the codes proposed by each LLM:\n"
    memory = {}  # Memory to track each LLM's contributions

    for llm, llm_codes in initial_codes.items():
        context += f"{llm}: {llm_codes}\n"
        memory[llm] = {"initial_codes": llm_codes, "discussions": []}

    context += (
        "\n Compare your codes to the codes of other LLMs and discuss them. "
        "Think about what codes you can improve, eliminate, or add."
    )

    discussion_transcript = []

    for round_num in range(3):  # Discussion rounds
        round_responses = []
        print(f"\n--- Round {round_num + 1} ---")

        for idx, llm in enumerate(llms):
            llm_id = f"LLM-{idx + 1}"
            llm_context = context + f"\nPreviously, you proposed these codes:\n{memory[llm_id]['initial_codes']}"
            llm_context += "\n\nHere's a summary of your previous contributions:\n"
            for summary in memory[llm_id]["discussions"]:
                llm_context += f"- {summary}\n"

            try:
                # Main discussion response
                prompt = f"Current discussion context:\n{llm_context}\n\nWhat is your input for this round?"
                response = llm.generate_response(prompt)

                # Summarize the response
                prompt2 = f"Summarize your main takeaways in one sentence: {response}"
                summary = llm.generate_response(prompt2)

                # Update memory for the current LLM
                memory[llm_id]["discussions"].append(summary)

                # Add to context
                context += f"\n{llm_id}: {summary}"
                context = truncate_context(context, max_tokens=8000)

                # Store in the transcript
                round_responses.append({"LLM": llm_id, "response": response, "summary": summary})
            except Exception as e:
                print(f"Error in {llm_id}: {e}")
                round_responses.append({"LLM": llm_id, "response": "Error generating response.", "summary": "Error."})

        discussion_transcript.append({"round": round_num + 1, "responses": round_responses})

    # Final consolidation
    final_responses = []
    for idx, llm in enumerate(llms):
        llm_id = f"LLM-{idx + 1}"
        
        # Prepare tailored context for each LLM
        final_context = "Here are the initial codes from all LLMs:\n"
        for other_llm, codes in initial_codes.items():
            final_context += f"{other_llm}: {codes}\n"
        
        final_context += "\nHere is a summary of the discussion so far:\n"
        for discussion_summary in memory[llm_id]["discussions"]:
            final_context += f"- {discussion_summary}\n"
        
        final_prompt = (
            f"{final_context}\n\n"
            "Based on the initial codes and the discussion so far, consolidate and finalize the thematic codes "
            "into a concise list. Each code should be no more than 2-5 words. Include original quotes where relevant."
        )
        
        # Generate final response
        try:
            response = llm.generate_response(final_prompt)
            final_responses.append({"LLM": llm_id, "response": response})
        except Exception as e:
            print(f"Error in final response for {llm_id}: {e}")
            final_responses.append({"LLM": llm_id, "response": "Error generating final response."})


    return {
        "initial_codes": initial_codes,
        "discussion_transcript": discussion_transcript,
        "final_codes": [res["response"] for res in final_responses],
        "memory": memory,  # Return the memory for future use or debugging
    }




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
        LLMManager(model_name="gpt-4o", api_key=api_keys.get("gpt-4")), 
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




