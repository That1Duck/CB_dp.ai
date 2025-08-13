# ChatBot

## Description
Implementation of an assistant in the form of a chatbot for deeplearning.ai. Main function is to help select a course or blog based on user request.

The implementation is based on a combination of RAG + Agetn.
The main reason for choosing this implementation option lies in the specific logic of the system's structure.

The choice of RAG + one agent-router in our variant is better because it adds three critically important properties to the classic RAG:
1. A conscious query route (course/blog/oos) with honest refusal and a confidence threshold — simple RAG almost always “pulls” the answer even outside the domain;
2. Controlled selection of the best candidate from top-k (through a strict format and post-ranking), rather than blindly returning the top-1 retriever;
3. Multimodality (image→text).

Unlike multi-agent approaches, the chosen approach avoids unnecessary coordination complexity, races, and delays: a single lightweight agent handles routing, while all the “heavy lifting” remains within the verifiable RAG circuit. The result is a simplified design with predictable linearity of choice.

| Criterion                    | RAG + Agent                                    | Plaint RAG                          | Multi Agents                                                      |
|------------------------------|------------------------------------------------|-------------------------------------|-------------------------------------------------------------------|
| Quality of response          | Routing + re-evaluation of top search results  | takes top-1 without re-evaluation   | High accuracy                                                     |
| Resistance to OOS            | Possibility of refusal + retriever threshold   | No refusal, hallucinations possible | Possible implementation, but requires coordination of agents.     |
| Delay/Cost                   | Linear call branch                             | Minimum delay and cost              | Agent coordination = frequent LLM calls + response time           |
| Complexity of implementation | Low / Average                                  | Low                                 | Average / high                                                    |
| Observability/assessment     | Medium                                         | Easy                                | Difficult                                                         |
| Multimodality                | Requires separate implementation               | Requires separate implementation    | Requires separate implementation, can be done by additional agent |
| Hybrid search support        | Unlimited choice of methods                    | Limited choice of methods           | Unlimited choice of methods                                       |
| Escort/delivery              | Relatively simple due to linearity             | Simple                              | Requires additional effort during debugging                       |


Basic logic of operation:

1. **Processing the input query**
   - Input: text or image
   - If imeage -> (OCR+BLIP) -> text
2. **Classifying the query**
   - Defines:
     - `intent`: `course` / `blog` / `oos`
     - `domain`: `deeplearning_ai` или `oos`
   - If `oos` ->  return the refusal with an explanation 
3. **Selecting a retriever**
   - Selects the vector base based on `intetn` 
   - Looking for top-k candidates
   - if there are no candidates with sufficient relevance -> refuse
4. **Searching for a suitable answer**
   - LLM analyzes top-k candidates
   - Selects the best
   - Forms a response


## First launch
Preparing the project for launch:
- Creating a virtual environment
  ```commandline
  python -m venv venv
  venv\Scripts\activate
  source venv/bin/activate # if macOS/Linux
  ```
- Installing the necessary libraries
  ```commandline
  pip install -r requirements.txt
  ```
- Creating .env
  ```
  OPENAI_API_KEY = [your key]
  LANGFUSE_PUBLIC_KEY = [your key]
  LANGFUSE_SECRET_KEY = [your key]
  LANGFUSE_HOST = https://cloud.langfuse.com
  ```

Parsing data and creating a vector database:

```commandline
python -m src.data_ingestion.collecting_blogs
python -m src.data_ingestion.collecting_courses
python -m src.data_ingestion.embedder
```

Launch:
- Launch the main application

  ```
  python main.py
  ```
- Launch foe evaluation

  ``` 
  python -m eval.run_eval_langfuse
  ```
## Basic structure
1. **data**
   - `raw` - storage location of original data after parsing
2. **vectorstore**
   - `blogs` - vectorized blogs data in the database
   - `courses` - vectorized courses data in the database
3. **src**
   - `agents` - defining intent and domain
   - `chains` - logic for selecting the best candidate
   - `retiriver` - Searching vector databases
   - `image_processing` - OCR + BLIP conversion of images to text
   - `interface` - launch of the pipeline
   - `data_ingestion` - data collection and preparation
4. **eval**
   - `dataset` - dataset for evaluation 
   - `run_eval_langfuse` - launch evaluation with LangFuse

## Test results 
To evaluate the chatbot's performance, a dataset was created with entries in the following format:
```JSON
  {
    "id": "e1",
    "query": "I want to learn Python for AI apps",
    "expected_intent": "course",
    "expected_url": "https://www.deeplearning.ai/short-courses/ai-python-for-beginners/"
  }
```

Metrics used to evaluate performance:
- intent_accuracy - does the `intent` of the request match the `expected_intent`
- oos_refusal - are the rejections of requests marked as `oos` correct
- hit@3 - does `expected_url` fall within the top 3 candidates
- url_match - does the received `url` match the `expected_url`

Test results and examples of how it works can be viewed [here](excperimets).

## Possible bottlenecks and opportunities for functional expansion
I would like to note that the following weaknesses were observed in the test results:
- converting information from an image 
- defining intent and domain
- searching in the database