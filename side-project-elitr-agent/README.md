# Outcome of ALPS 2025 Mini Project - Create an Agent Benchmark Dataset from a Q&A Benchmark Dataset (Meetings Domain)

**Supervisor:** [Laurent.Besacier@naverlabs.com](mailto:Laurent.Besacier@naverlabs.com)  
**ALPS 2025 students:**  
- [philippe.martin@irisa.fr](mailto:philippe.martin@irisa.fr)  
- [salim.aissi@isir.upmc.fr](mailto:salim.aissi@isir.upmc.fr)  
- [florian.le-bronnec@dauphine.psl.eu](mailto:florian.le-bronnec@dauphine.psl.eu)  
- [s.duong@criteo.com](mailto:s.duong@criteo.com)  

📥 **Download:** [elitr-bench-agent-dev.zip](./elitr-bench-agent-dev.zip) — the outcome of the miniproject, containing the _elitr-bench-agent_ dataset for all dev meetings.
🎓 **Final student presentation** of the mini-project, presented on the last day of **ALPS 2025** — [view here](./ELITR_Agent_Bench__An_Agent_Benchmark_for_Meeting_Transcripts.pdf).



---

## Short Description

From a Q&A dataset that evaluates LLMs, augment it (using strong models like GPT-4o, OpenAI o-1, DeepSeek, etc.) to make it an **Agent benchmark dataset** — one that includes **actions** (also called *function calls*) in addition to natural language generation.

---

## More Details

We aim to move from an **assistant** to an **agent** benchmark — i.e., from an AI that only answers questions in natural language, to one that can **take actions** (e.g., sending emails, creating calendar events, updating spreadsheets, posting on social media).

We will build upon an existing meeting Q&A dataset:  
🔗 [ELITR-Bench](https://github.com/utter-project/ELITR-Bench)

The goal is to enrich the Q&A task by embedding **action tokens** into generated answers, allowing us to evaluate an LLM’s ability to infer and propose relevant actions based on meeting transcripts.

These actions could be carried out either by human meeting participants or by the AI agent. For human-assigned tasks, the AI would propose to assist the person in completing them.

This project will focus on **extracting a structured list of actions** from the meeting data. Interpretation or execution (with or without human confirmation) is out of scope for now.

---

## Example: Action-Enriched Dataset Entry

```json
"meeting_id": "meeting_en_dev_001",
"actions": [
  {
    "action_type": "send_email",
    "description": "Send an email to [PERSON8] (currently on vacation) to discuss their availability and seek advice on downloading scripts.",
    "responsible_person": "[PERSON3]",
    "priority": "medium"
  },
  {
    "action_type": "schedule_meeting",
    "description": "Schedule a follow-up meeting with [PERSON1] to confirm their role in preparing the corpus.",
    "responsible_person": "[PERSON6]",
    "priority": "high"
  },
  {
    "action_type": "data_analysis",
    "description": "Prepare a summary of the [ORGANIZATION1] corpus and evaluate its suitability as a separate track in the shared task.",
    "responsible_person": "[PERSON5]",
    "priority": "high"
  }
]
```

➡️ This could result in the agent generating a message like:

> Hi [PERSON3],  
> I noticed you're assigned to the following task:  
> *"Send an email to [PERSON8] (currently on vacation) to discuss their availability and seek advice on downloading scripts."*  
> If you'd like, I can take care of this action on your behalf. Let me know if you're okay with that, and I’ll proceed accordingly.  
>  
> Best,  
> **TheAgent**

---

## To Do in the mini-project


1. **Read** the ELITR-Bench dataset paper:  
   📄 [https://aclanthology.org/2025.coling-main.28/](https://aclanthology.org/2025.coling-main.28/)

2. **Explore** the ELITR-Bench repository:  
   🔗 [https://github.com/utter-project/ELITR-Bench](https://github.com/utter-project/ELITR-Bench)  
   ➕ Use the `data.zip` file and English meeting transcripts from:  
   🔗 [https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4692](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4692)

3. **Generate actions** using an LLM:
   - Start with the OpenAI o-1 model (or try others).
   - Use the prompt provided below.
   - Consider breaking the task into steps (CoT):
     1. Generate a list of relevant actions.
     2. Associate a dozen of actions per meeting using transcripts and Q&A.
     3. Prompt multiple models to compare results.
   - Once you find a good `{prompt, LLM}` pair, apply it across all `{meeting, Q&A}` sets to create the **Elitr-Bench-Agent** dataset.

4. **Manually review and clean** the generated actions to ensure alignment with the meeting content.

5. **Evaluate** how well different LLMs can generate actions for each `{meeting, Q&A}` — using your newly created benchmark dataset.

---

### Prompt Used to Generate the Example

```
Here are two files:  
- meeting_en_dev_001.txt: a transcript of a past meeting  
- meeting_en_dev_001.json: contains questions about the meeting and their human-annotated answers  

These are currently used to evaluate meeting assistants powered by LLMs.

NOW I WANT TO move from assistants to agents — AI that not only answers in natural language but also performs actions (like sending emails, creating calendar events, updating spreadsheets, posting on social media).  

Your task is to analyze both files and propose a list of **actions** that should be taken based on the meeting transcript, in a **re-usable format** (your choice), so I can later use them along with the two files to evaluate agent capabilities.
```

# Method

**Main idea:** Split the task into simpler subtasks for better performance and control.

---

## Prompt Pipeline

Prompts are described above. They might need refinement—especially adding an explicit list of possible actions/tools should help improve action selection.

### 1. Exhaustive Extraction

Implementation in **LangChain** with structured outputs:

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
   openai_api_key="API_KEY",
   openai_api_base="URL",
   model_name="MODEL",
   max_tokens=1024,
   temperature=0.6,
   top_p=0.6,
)

class Action(BaseModel):
   action_id: int
   supporting_excerpt: str
   assigned_persons: list[str]
   description: str

class ActionExtractionOutput(BaseModel):
   actions: list[Action]

structured_llm = llm.with_structured_output(ActionExtractionOutput, include_raw=True)
```

**Key advantage:**  
*Supporting excerpts improve faithfulness and help with further refinement.*

---

### 2. TODO / NOT TODO Filtering

The model sometimes outputs tasks that have **already been done** instead of actual TODOs.

These are relatively easy to filter out using a **second prompt**, where we use LLM to label actions:

```python
class IsTODOAction(BaseModel):
   action: Action
   is_todo: bool
```

Then we keep only those with `is_todo=True`.

---

### 3. Actionable Actions Selection

From the TODOs, we filter again to keep only **actionable** items using a third prompt:

```python
class IsActionableAction(BaseModel):
   action: Action
   is_todo: bool
```

This yields the final list of actionable TODOs.

---

## Evaluation Pipeline

We track several **quality signals** throughout the extraction process.

### Automated Metrics

- Number of extracted tasks
- Number of assigned persons
- Does the supporting excerpt actually appear in the transcript?

---

### LLM Metrics

We split the evaluation into simpler subtasks:

- **Chunk-based assessment**: Try to match each task to a chunk of the transcript.
- **Smaller context windows**: Enable finer evaluation and help detect hallucinations.
- **Limitations**: Might miss long-range dependencies—though we haven’t found strong evidence for these.

In practice, we use **overlapping chunks** and compute the percentage of tasks matched to a chunk.

Sample code (may need adjustment for your environment):

```python
import asyncio
import nest_asyncio
from pydantic import BaseModel

nest_asyncio.apply()  # For Jupyter notebooks

class Action(BaseModel):
   action_id: int
   assigned_persons: list[str]
   action_description: str
   supporting_excerpt: str

class ActionExtractionOutput(BaseModel):
   actions: list[Action]

class SupportedAction(BaseModel):
   action: Action
   is_supported: bool

class SupportedChunkFormatter(BaseModel):
   answers: list[SupportedAction]
```

```python
from langchain_core.prompts.chat import ChatPromptTemplate

prompt_str = """Below is an excerpt from a meeting transcript:
{chunk}

Based on the transcript, here is a list of proposed follow-up actions along with supporting text:
{actions_list}

For each action, assign a label:
- YES if the action is explicitly mentioned in the transcript, and is correct, i.e., the action is relevant to the meeting and the assigned person is correct.
- NO if the action is not mentioned in the transcript, or is incorrect, i.e., the action is not relevant to the meeting or the assigned person is incorrect.
Answer following the described format."""

prompt_template = ChatPromptTemplate(
   [
       ("system", "You are a helpful assistant that helps with meeting transcripts."),
       ("human", prompt_str),
   ]
)

structured_llm = llm.with_structured_output(SupportedChunkFormatter, include_raw=True)
chain = prompt_template | structured_llm

async def process_chunk(i):
   return await chain.ainvoke(
       {
           "chunk": chunked_transcript[i],
           "actions_list": actions,
       }
   )

async def process_all_chunks():
   actions_list = [process_chunk(i) for i in range(1)]
   results = await asyncio.gather(*actions_list)
   return results

results = await process_all_chunks()
```

---

## Intermodel Agreement

We discussed how using **several model annotations** could improve reliability. However, aligning and comparing similar tasks across outputs is non-trivial and requires matching logic.

---

## Results / Further Work

- The pipeline provides **decent results**, but **some irrelevant actions still sneak in**.
- A more detailed **error analysis** is needed to refine prompts and filtering logic.
- A **clear list of valid actions** will improve model behavior and selection quality.
- Ultimately, **some human annotation** is needed to assess recall and tune the pipeline.


