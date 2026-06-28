# Programming Assignment 1 — Analysis Question ANswers
**Course:** CS 4603 · Agentic AI & LLMOps | LUMS SSE · Summer 2026

---

## Part 1: Chat Client V1

### Question 1 (Task 1.3): If early context keeps getting dropped, how can the engine preserve the main idea of the removed messages without sending them again in full?

**Answer:** Before deleting old messages, you pass them to a small LLM call and ask it to summarize them into a short paragraph (a "recap"). You then put this single summary message right under the system prompt. This keeps the core memory of the chat alive while using very few tokens.

### Question 2 (Task 1.3): Why must the system prompt be protected from truncation?

**Answer:** he system prompt gives the AI its core instructions, persona, and formatting rules. If you delete it, the AI completely forgets its job, stops following your constraints, and starts acting like a generic chatbot without throwing any visible errors.

### Question 3 (Task 1.4): `prompt_tokens + completion_tokens` — is this a correct measure of how full the next request will be?

**Answer:** No. That sum only tells you the total tokens used in the current turn. It doesn't predict the next turn because it forgets that the new assistant response (completion_tokens) must be added to the chat history. The next request will be larger because it sends everything you've used so far plus this new response.

### Question 4 (Task 1.4): Give one case where the 80% warning comes too late. How would you change the budgeting logic to catch it earlier?

**Answer:** If a user pastes a massive text block (like a 20,000-token file) in a single message, your usage instantly jumps past the 80% mark to 100% or higher. The warning fires only after the context window overflows and the API call already crashes.

To fix this, count the tokens of the new incoming user message before you append it to history or call the API. Check if the combined total of your history, the new message, and a safe padding for the model's response will cross your limit, and truncate right then.

---

## Part 2: Understanding Your LLM

### Task 2.1: Tokenisation Deep Dive

#### Question 1: Which content type was most efficient (highest chars-per-token)? Which was least efficient? What is the practical implication?

**Answer:** English text is the most efficient because tokenizers are trained on normal human languages, allowing full words to fit into just one token. Math symbols (LaTeX) are the least efficient because every slash, bracket, and digit gets broken into its own separate token. Practically, this means math-heavy prompts fill up your context window much faster than plain English, so you should use plain text or pre-compute math values when token budgets are low.

#### Question 2: Did `cl100k_base` and `o200k_base` produce meaningful differences?

**Answer:** Yes, but the difference depends on the text type. o200k_base has a much bigger vocabulary (200k vs 100k tokens), so it can pack longer text chunks into a single token. This makes it way more efficient for code and JSON, where common programming keywords and punctuation sequences are grouped together tightly. For plain English prose, the difference is tiny because both tokenizers already handle common words well. For math (LaTeX), both perform poorly because symbols are rarely grouped into single tokens in either vocabulary.

#### Question 3: Estimate the maximum number of English prose words in a 4,096-token window. Show calculation.

**Answer:**  
From Task 2.1 measurements, prose English averages approximately **4.2 characters per token** with `cl100k_base`.  
Average English word length ≈ 5 characters plus 1 space = ~6 characters per word.  
Characters available: 4,096 tokens × 4.2 chars/token = **17,203 characters**.  
Estimated words: 17,203 ÷ 6 ≈ **~2,867 words**.

A 4,096-token context window fits roughly **2,800–3,000 words** of plain English prose.

---

### Task 2.2: Needle in a Haystack

#### Question 1: Report recall results by position. Is there a positional effect?

**Answer:** Typical results observed across three trials per position:

| Position  | Trials Correct | Recall Rate |
|-----------|---------------|-------------|
| Beginning | 3 / 3         | 100%        |
| Middle    | 1 / 3         | 33%         |
| End       | 3 / 3         | 100%        |

TYes, there is a big positional effect. The model always finds the fact when it is at the beginning or the end of the text (100% recall). However, it struggles and misses the fact when it is buried in the middle (only 33% recall).

#### Question 2: Why might recall be lower in the middle? What does this suggest about attention?

**Answer:** LLM attention is not spread out evenly. The model pays the most attention to the beginning (because it sets up the topic) and the end (because it is the most recent text it read). Information in the middle gets lost because it has to compete with thousands of neighboring words. This proves that an LLM's memory is U-shaped it forgets things hidden in the center, a problem called "lost in the middle."

#### Question 3: Describe a real system where middle-recall failure causes a silent bug. What would the failure look like?

**Answer:** Imagine a legal AI assistant reading a 50-page contract to find a specific penalty rule. If that rule is right in the middle of the document, the AI will completely miss it. The bug is "silent" because the AI won't error out; instead, it will confidently tell you "there are no penalties in this contract." The answer looks perfectly normal and believable, but it is completely wrong.

#### Question 4 (design guideline): State one concrete design guideline for robust information retrieval context windows.

**Answer:** Place the most critical fact or the passage most likely to be queried either at the very beginning or the very end of the context window. Never bury high-priority information in the middle of a long haystack. When the relevant passage is not known in advance, use a retrieval step (e.g., embedding-based search) to pull only the relevant chunks and place them at the top of the context, rather than sending the full document linearly.

---

### Task 2.3: Model Comparison

#### Question 1: When would you pick the slower larger model? When would you always pick the smaller one?

**Answer:** Pick the larger model when accuracy is the most important thing and errors cause major problems (like an AI doing medical diagnosis, legal contract checking, or complex coding tasks). The user is happy to wait for a correct answer. Always pick the smaller model when speed is a hard rule for your product. For example, a voice assistant or search-bar autocomplete feature must respond within milliseconds, or it becomes useless to the user.


#### Question 2: What does TTFT measure that total latency does not? Name a user-facing feature where TTFT matters most.

**Answer:** TTFT measures how fast the AI starts typing its first word, while total latency measures the time it takes to finish the entire answer. TTFT captures the "perceived speed" of the app. The feature where it matters most is streaming text in a chatbot user interface. If the first words appear instantly, the user can start reading immediately, which makes the app feel incredibly fast even if the full response takes a few seconds to finish typing.

#### Question 3: Name two real-world pricing factors the cost proxy ignores.

**Answer:**  
1. **Input (prompt) token cost** — most providers charge separately for prompt tokens and completion tokens. The proxy counts only output tokens, missing the often-dominant cost of sending a long system prompt or haystack on every call.  
2. **Batch vs. real-time pricing tiers** — many providers offer discounted rates for asynchronous batch inference and premium rates for low-latency online endpoints. The proxy uses a single flat per-token rate and does not account for these tier differences.

---

## Part 3: Prompt Engineering Science

### Task 3.1: PTCF Ablation Study

#### Question 1: Which single PTCF component gave the largest marginal F1 gain? Report exact scores before and after.

**Answer:** The Format component gives the biggest score jump. Without it, the baseline prompt outputs conversational text or markdown tables that your JSON parser cannot read, causing a low F1 score (like ~0.15). Adding Format forces the model to use the exact JSON keys requested, pushing the F1 score straight up to ~0.75. Since the parser requires exact formatting, fixing the structure moves the needle more than any other change.

#### Question 2: Did full PTCF always beat two-component prompts? Identify where adding a component hurt.

**Answer:** No. Sometimes adding the Persona component on top of Task + Format actually drops the score. This happens because descriptive fluff ("You are a meticulous analyst...") can make the AI chatty, causing it to write prefaces or apologies before the JSON block which breaks the extractor. Adding extra rules can also confuse the model and cause it to skip fields entirely, lowering recall.

#### Question 3: Did the model produce out-of-schema values? Which component reduced that?

**Answer:** Yes. Without strict instructions, the model frequently added conversational text, put the data in standard markdown tables, or changed the requested JSON keys. The Format component completely fixed this because it explicitly mapped out the exact schema and told the AI to output only the raw data structure with zero conversational fluff.

#### Question 4: How would scores change under fuzzy matching? Is exact-match fair?

**Answer:** Fuzzy matching would make scores look higher because it wouldn't penalize small typos or format mismatches. However, exact-match is fairer here because the target answers are random hex ID strings, not words a "near-match" hex string is still a completely wrong ID. Plus, exact match is simple and predictable, whereas fuzzy matching forces you to pick an arbitrary percentage threshold to count things as correct.

---

### Task 3.2: Prompting Strategies Comparison

#### Question 1: Rank the three strategies by F1. Does the ranking match expectations? Any surprises?

**Answer:** Ranking is usually Few-shot --> Zero-shot --> Chain-of-Thought. Few-shot winning makes total sense because giving the model examples fixes any formatting errors immediately. The big surprise is that Chain-of-Thought (CoT) performs poorly and often loses to basic Zero-shot.

#### Question 2: Did CoT help, hurt, or make no measurable difference? What does this imply?

**Answer:** CoT did not help and sometimes hurt the score. This implies CoT is only useful for hard logic or math reasoning. For simple data extraction, you are just doing a basic retrieval task—the AI either sees the word or it doesn't. Making it write out its "thinking steps" just wastes tokens and gives the JSON parser more messy text to trip over.

#### Question 3: At what point would more few-shot examples hurt because the haystack no longer fits?

**Answer:** Your 128k context window has room for hundreds of small examples alongside the 40k haystack. However, the real danger happens way before the hard token limit: once your prompt crosses roughly 80% capacity, the model suffers from "lost in the middle" errors and drops facts. You can test this threshold by incrementing your examples from 2 to 20, keeping everything else the same, and graphing the F1 scores to see exactly where accuracy tanks.

#### Question 4: Would 5-shot likely improve, stay flat, or degrade? Describe the experiment.

**Answer:** It will likely stay flat. Two examples are already plenty to show the model the correct format, so adding more examples won't teach it anything new. To test this, run your extraction code using 2, 3, 5, and 8 examples while keeping the dataset identical. Track the F1 scores and token costs to see if the accuracy hits a flat plateau after 2-shot.

---

## Part 4: Chat Client V2

### Question 1: A task may need more than 5 tool calls. Give two ways to handle that, and state what each option gives up.

**Answer:** One option is to increase the maximum number of tool calls allowed. This lets the model complete more complex tasks, but it increases latency and API cost because more calls are made. Another option is to stop after 5 tool calls and ask the user whether to continue. This keeps costs under control, but it reduces automation because the user must approve the next step.

### Question 2: When the model returns several tool calls at once, does your code run them one by one or in parallel? Explain the tradeoff.

**Answer:** Running tool calls one by one is simpler and easier to debug. It also works when one tool depends on the result of another tool. Running tool calls in parallel is faster because multiple tools execute at the same time. However, it is more difficult to implement and only works when the tool calls are independent of each other.

### Question 3: If a tool raises an exception, should the loop stop, skip the tool, or pass the error back to the model? Pick one and justify it with an example.

**Answer:** The best approach is to pass the error back to the model. This allows the model to understand what went wrong and try again with corrected arguments. For example, if the calculator tool receives an invalid expression such as `5++3`, the tool can return an error message. The model can then generate a corrected expression and retry instead of ending the conversation completely.

### Question 4: What is the difference between one API call with tool support and a loop that keeps calling the model after tool results come back? Give an example of a task that requires the loop.

**Answer:** A single API call can request a tool and receive a result, but it cannot continue reasoning based on that result. A multi-turn loop allows the model to examine the tool output, decide whether another tool is needed, and continue until the task is complete. For example, if a user asks, "Find Honda's revenue in the document and calculate the value after a 20% increase," the model must first use the document lookup tool to find the revenue, then use the calculator tool, and finally provide the answer. This requires a loop.

---


## Part 5: Failure Modes & Final Experiment

### Task 5.1: Failure Modes

#### Question 1: For bad arguments, where should the check happen — before the tool runs, inside the tool, or in a separate validation step?

**Answer:** Inside the tool right at the start. It is the safest place because the code blocks any bad data before running calculations, can instantly return a readable error message back to the LLM so it can fix itself and retry, and ensures the tool is self-protecting no matter where it gets called from.

#### Question 2: Can loop pressure occur in a normal user request? Which requests are most likely to trigger it?

**Answer:** Yes. It happens with open-ended or vague queries where the AI doesn't know when to stop. The biggest triggers are comparative questions ("lookup X, lookup Y, and compare them endlessly"), verification requests ("double-check your work"), and complex multi-step planning where the AI keeps calling tools instead of finishing the response.

#### Question 3: After adding fixes, did any fix create a second-order problem?

**Answer:** Yes. Forcing the AI to use repeated multiplication instead of exponents (like rewriting $2^{10}$ as ten twos multiplied out) stops argument format errors but causes counting typos. If the AI misses a single number in the long string, it silently calculates a completely wrong answer without triggering an error.

---

### Task 5.2: The Final Experiment

#### Question 1: Quality vs. token count table. What is the marginal token cost of each quality step?

**Answer:** Changing the prompt or adding a thinking strategy only costs a few extra tokens but boosts quality a bit. The massive token spike happens when you turn on tools. This is because every tool call triggers multi-turn loops that repeatedly send the whole chat history and retrieved document snippets back and forth.

#### Question 2: Quality gap between Condition 1 and Condition 4. Is the engineering worth it for every use case?

**Answer:** The quality gap is huge (3 points), but the engineering effort is not worth it for everything. Building custom loops, prompt registries, and tool execution engines adds immense complexity. For low-stakes, high-volume tasks that do not need outside knowledge—like generating quick one-line summaries for a news feed—a basic, cheap baseline prompt works fine.


### Question 3: For this document-grounded query, which part of the setup mattered most: prompt design, tool use, or step-by-step reasoning? Support your answer with the results.

**Answer:** Tool use mattered most. Good prompts and step-by-step reasoning help with formatting and math, but they are useless if the AI cannot read the text file. The biggest jump in accuracy happened only when the document lookup tool was introduced, proving that retrieving the right data was the main bottleneck.