
import json
import openai
from tqdm import tqdm
import re
import argparse

# -------------------------------
# argpars
# -------------------------------
parser = argparse.ArgumentParser(description="Run debate-based QA with GPT")
parser.add_argument("--local_agent_path", type=str, required=True, help="Path to local agent output JSON file (e.g., detail.json)")
parser.add_argument("--global_agent_path", type=str, required=True, help="Path to global agent output JSON file (e.g., global.json)")
parser.add_argument("--qa_file", type=str, required=True, help="Path to QA JSON file")
parser.add_argument("--output_file", type=str, required=True, help="Path to save debate results")
parser.add_argument("--error_log_file", type=str, required=True, help="Path to save error logs")
parser.add_argument("--api_key", type=str, required=True, help="Your OpenAI API key")
parser.add_argument("--max_turns", type=int, default=3, help="Maximum number of debate turns (default: 3)")
args = parser.parse_args()

# -------------------------------
# OpenAI API 
# -------------------------------
openai.api_key = args.api_key

GPT_PRICING = {
    "gpt-3.5-turbo-1106": {
        "input_per_1k_tokens": 0.0010,  # $0.001 per 1K input tokens
        "output_per_1k_tokens": 0.0020  # $0.002 per 1K output tokens
    },
    "gpt-4-1106-preview": {
        "input_per_1k_tokens": 0.01,   # $0.01 per 1K input tokens
        "output_per_1k_tokens": 0.03   # $0.03 per 1K output tokens
    }

}



# Track total token usage
total_input_tokens = 0
total_output_tokens = 0



def ask_gpt(prompt, q_uid, step, model="gpt-3.5-turbo-1106"):
    try:
        if step == "Agent1 selection" or "Agent2 selection":
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and precise assistant for select option."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
        elif step == "Agent1_debate selection"or "Agent2_debate selection":
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and precise assistant for checking reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful and precise assistant for checking the correctness of the answers from two agents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

        raw_content = response["choices"][0]["message"]["content"].strip()
        #usage = response["usage"]
        #total_input_tokens += usage["prompt_tokens"]  # Now correctly references global variable
        #total_output_tokens += usage["completion_tokens"]


        json_match = re.search(r'```json\s*([\s\S]+?)\s*```', raw_content)
        if json_match:
            raw_content = json_match.group(1).strip()  
        else:
            json_match = re.search(r'{[\s\S]+}', raw_content)
            if json_match:
                raw_content = json_match.group(0).strip()
            else:
                raise ValueError("No JSON found in the response.")

        result = json.loads(raw_content)
        
        return result
    except Exception as e:
        if 'response' in locals():
            print(f"Response: {response}")
        print(f"Error occurred while parsing response for {q_uid} at step {step}: {e}")

        errors[q_uid] = {"error_step": step, "error": str(e)}
        return None
        
def load_json(filepath):
    """Load a JSON file and return its content."""
    with open(filepath, "r") as f:
        return json.load(f)


# Load input files


with open(args.local_agent_path, "r") as f:
    local_agent_path = json.load(f)["data"]

with open(args.global_agent_path, "r") as f:
    global_agent_path = json.load(f)


with open(args.qa_file, "r") as f:
    questions = json.load(f)


def debate(q_uid, max_turns=3):
    if q_uid not in questions or q_uid not in summaries:
        return None
    caption = detail_agent_path[q_uid]["captions"]
    qa = questions[q_uid]
    summary = nocap_agent_path[q_uid]["summary"]
    question_text = qa["question"]
    options = {key: value for key, value in qa.items() if key.startswith("option")}

    # Extract choices and reasoning
    agent1_choice = nocap_agent_path[q_uid]["pred"]
    agent1_reasoning = nocap_agent_path[q_uid]["response"]
    agent2_choice = detail_agent_path[q_uid]["pred"]
    agent2_reasoning = detail_agent_path[q_uid]["response"]


    #print(f"Agent 1 chose {agent1_choice}: {agent1_reasoning}")
    #print(f"Agent 2 chose {agent2_choice}: {agent2_reasoning}")
    debate_log = {
        "q_uid": q_uid,
        "question": question_text,
        "options": options,
        "summary": summary,
        "turns": []
    }

    turn = 0
    #while turn < max_turns and agent1_choice != agent2_choice:

    if agent1_choice == agent2_choice:
        turn_log = {
            "turn": turn + 1,
            "agent1_choice": agent1_choice,
            "agent1_reasoning": agent1_reasoning,
            "agent2_choice": agent2_choice,
            "agent2_reasoning": agent2_reasoning
        }

        debate_log["final_answer"] = agent1_choice
        debate_log["agreement"] = True
    else:
        turn = 0
        while turn < max_turns and agent1_choice != agent2_choice:
            turn_log = {
                "turn": turn + 1,
                "agent1_choice": agent1_choice,
                "agent1_reasoning": agent1_reasoning,
                "agent2_choice": agent2_choice,
                "agent2_reasoning": agent2_reasoning
            }

            prompt_debate1 = f"""
            You are an AI for **logical evaluation in video-based QA debates**. Your role is to compare two arguments:
            1. **Your past reasoning** (Agent1)
            2. **Agent2’s counter-argument**

            Your task is to determine which argument is **logically superior** using **explicit video evidence, logical soundness, and relevance to the question**.

            ---
            ### **Debate Process:**
            1. **Analyze Agent2’s Counter-Argument:**
            - Does it use **explicit video evidence**?
            - Does it contain logical flaws or misinterpretations?
            - Does it effectively challenge your past reasoning?

            2. **Re-evaluate Your Past Reasoning (Agent1):**
            - Does it align with the **video summary and captions**?
            - Are there any **gaps in evidence or logical errors**?

            3. **Compare & Decide:**
            - Which argument is **better supported** by video-based evidence?
            - If both are flawed, **choose a more reasonable alternative**.

            ---
            ### **Input:**
            - **Question:** {question_text}
            - **Video Summary:** {summary}
            - **Video Captions:** {caption}
            - **Options:** {options}

            #### **Your Past Choice (Agent1):**
            - **Option:** {agent1_choice} - "{options[f'option {agent1_choice}']}"
            - **Reasoning:** {agent1_reasoning}

            #### **Agent2’s Counter-Argument:**
            - **Option:** {agent2_choice} - "{options[f'option {agent2_choice}']}"
            - **Reasoning:** {agent2_reasoning}

            ---
            ### **Decision Criteria:**
            - **Fact-Based Justification:** Must cite explicit video evidence.
            - **Logical Coherence:** Argument must be logically sound.
            - **Relevance:** Argument must directly answer the question.

            ---
            ### **Final Decision (JSON only):**
            {{
                "option": OPTION_NUMBER,
                "reason": {{
                    "fact_evidence": "Explicit video evidence supporting the choice.",
                    "logical_consistency": "Why this argument follows a strong reasoning structure.",
                    "relevance": "How well this choice answers the question.",
                    "counter_evidence": "If rejecting an argument, explain why it is flawed."
                }}
            }}

            ---
            ### **Rules:**
            - **Output JSON only. No extra text.**
            - **The option must be an integer (e.g., 0, 1, 2), not a string.**
            - **Explicitly cite video-based evidence.**
            """





            agent1_new_response = ask_gpt(prompt_debate1, q_uid, "Agent1_debate selection")
            if agent1_new_response is None:
                return None  # Continue processing other questions

            print(agent1_new_response)

            agent1_new_choice = agent1_new_response["option"]
            agent1_new_reasoning = agent1_new_response["reason"]

            remaining_options = {k: v for k, v in options.items() if k != f"option {agent1_choice}"}
            prompt_debate2 = f"""
            You are an AI for **logical evaluation in video-based QA debates**. Your role is to compare two arguments:
            1. **Your past reasoning** (Agent2)
            2. **Agent1’s counter-argument**

            Your task is to determine which argument is **logically superior** based on explicit **video evidence, logical soundness, and relevance to the question**.

            ---
            ### **Debate Process:**
            1. **Analyze Agent1’s Counter-Argument:**
            - Does it use **explicit evidence** from the video summary/captions?
            - Does it misinterpret or overgeneralize key details?
            - Does it **effectively challenge** your past reasoning?

            2. **Re-evaluate Your Past Reasoning:**
            - Is it factually correct based on the video summary/captions?
            - Does it lack key evidence or contain logical flaws?

            3. **Compare & Decide:**
            - Which argument is **better supported** by video-based evidence?
            - If both are flawed, **choose a more reasonable alternative**.

            ---
            ### **Input:**
            - **Question:** {question_text}
            - **Video Summary:** {summary}
            - **Video Captions:** {caption}
            - **Options:** {options}

            #### **Your Past Choice (Agent2):**
            - **Option:** {agent2_choice} - "{options[f'option {agent2_choice}']}"
            - **Reasoning:** {agent2_reasoning}

            #### **Agent1’s Counter-Argument:**
            - **Option:** {agent1_choice} - "{options[f'option {agent1_choice}']}"
            - **Reasoning:** {agent1_reasoning}

            ---
            ### **Decision Criteria:**
            - **Fact-Based Justification:** Cite explicit video evidence.
            - **Logical Soundness:** Argument must follow a strong reasoning structure.
            - **Relevance:** Argument must directly answer the question.

            ---
            ### **Final Decision (JSON only):**
            {{
                "option": OPTION_NUMBER,
                "reason": {{
                    "fact_evidence": "Explicit video evidence supporting the choice.",
                    "logical_consistency": "Why this argument follows a strong reasoning structure.",
                    "relevance": "How well this choice answers the question.",
                    "counter_evidence": "If rejecting an argument, explain why it is flawed."
                }}
            }}

            ---
            ### **Rules:**
            - **Output JSON only. No extra text.**
            - **The option must be an integer (e.g., 0, 1, 2), not a string.**
            - **Explicitly cite video-based evidence.**
            """





            agent2_new_response = ask_gpt(prompt_debate2, q_uid, "Agent2_debate selection")
            if agent2_new_response is None:
                return None  # Continue processing other questions
            print(agent2_new_response)
            agent2_new_choice = agent2_new_response["option"]
            agent2_new_reasoning = agent2_new_response["reason"]

            turn_log["agent1_new_choice"] = agent1_new_choice
            turn_log["agent1_new_reasoning"] = agent1_new_reasoning
            turn_log["agent2_new_choice"] = agent2_new_choice
            turn_log["agent2_new_reasoning"] = agent2_new_reasoning
            debate_log["turns"].append(turn_log)


            agent1_choice, agent1_reasoning = agent1_new_choice, agent1_new_reasoning
            agent2_choice, agent2_reasoning = agent2_new_choice, agent2_new_reasoning

            turn += 1
        if agent1_choice == agent2_choice:
            debate_log["final_answer"] = agent1_choice
            debate_log["agreement"] = True
        else:

            prompt_final = f"""        
            ----
            You are an impartial and logical evaluator. Based on the given **video summary**, **video caption**, **question**, and available **options**, two agents (Agent 1 and Agent 2) have analyzed the situation and proposed their respective choices with justifications.

            Your task is to **carefully analyze their reasoning**, compare their arguments, and make the **final decision** based on the most well-supported choice. **If the Video Summary lacks necessary details, use the Video Caption to extract relevant evidence and clarify the context.**  

            ---
            ### **Input:**
            - **Question:** {question_text}  
            - **Video Summary:** {summary}  
            - **Video Caption:** {caption}  
            - **Options:** {options}  

            ---
            ### **Agent's Argument:**
            {debate_log["turns"]}
            ---
            ### **Your Task:**
            1. **Extract and Clarify Video Information:**  
            - If the **Video Summary is vague**, use the **Video Caption** to derive **specific evidence** supporting or contradicting each agent's claim.  

            2. **Analyze Each Agent's Reasoning in Depth:**  
            - Evaluate **Agent 1’s choice, reasoning, and advocacy** in detail.  
            - Assess **Agent 2’s criticism** against Agent 1's argument.  
            - Evaluate **Agent 2’s choice, reasoning, and advocacy** in detail.  
            - Assess **Agent 1’s criticism** against Agent 2's argument.  

            3. **Check Consistency with the Video Summary and Caption:**  
            - Compare both agents' arguments with the **video summary and caption**.  
            - Identify which choice is better supported by concrete evidence.  

            4. **Make the Final Decision:**  
            - Select the most **logically justified option** based on **clear and well-supported reasoning**.  
            - Provide a **detailed explanation** explicitly citing evidence from the **video summary, caption, and agent arguments**.  
            - Clearly explain **why the other choice is rejected**.  

            ---
            ### Output format (STRICTLY JSON):
            {{
                "option": OPTION_NUMBER,  # Option should be strictly a number (e.g., 0, 1, 2)
                "reason": "your_reasoning (must explicitly cite supporting evidence from the video summary, caption, and both agent arguments)"
            }}

            Your response MUST be in this JSON format only, with no additional explanations.
            """

            agent3_response = ask_gpt(prompt_final, q_uid, "Agent3_final_decision")
            if agent3_response is None:
                return None  # Continue processing other questions

            print(agent3_response)
            agent3_choice = agent3_response["option"]
            agent3_reasoning = agent3_response["reason"]

            # 최종 답변을 Agent3의 선택으로 결정
            debate_log["final_answer"] = agent3_choice
            debate_log["final_reasoning"] = agent3_reasoning
            debate_log["agreement"] = False  # Agent3가 개입했다는 것은 초기 두 에이전트가 합의하지 않았음을 의미
    return debate_log


all_results = {}
errors = {}
for q_uid in tqdm(questions.keys(), desc="Processing questions", unit="question"):
    try:
        result = debate(q_uid, max_turns=1)
        if result:
            all_results[q_uid] = result
    except Exception as e:
        errors[q_uid] = errors.get(q_uid, [])
        errors[q_uid].append({"error_step": "Main loop", "error": str(e)})
        print(f"⚠️ Error processing q_uid {q_uid}: {e}")

try:
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Debate results saved to {output_file}")
except Exception as e:
    print(f"Error saving debate results: {e}")

if errors:
    try:
        with open(error_log_file, "w") as f:
            json.dump(errors, f, indent=4)
        print(f"Errors saved to {error_log_file}")
    except Exception as e:
        print(f"Error saving error log: {e}")
