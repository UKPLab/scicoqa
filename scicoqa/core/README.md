# SciCoQA Core Library

The core libary's purpose is to provide a systematic, reproducible and extensible way to perform inference with LLMs.

Therefore, each inference experiment consists of the following components:

- A model, including configuration defined in [config/models.yaml](../../config/models.yaml), which instantiates an [LLM](./llm.py) object
- A prompt, defined in [config/prompts.yaml](../../config/prompts.yaml), which instantiates a [Prompt](./prompt.py) object
- A data iterator, defined in [data_iterator.py](./data_iterator.py), which instantiates a [BaseIterator](./data_iterator.py) object
- An arguments class, defined in [args.py](./args.py), which instantiates a [BaseArgs](./args.py) object to handle common arguments for all experiments
- An experiment, defined in [experiment.py](./experiment.py) which orchestrates the other components

```mermaid
graph TD
    config_prompt[prompts.yaml] --> cls_prompt[Prompt]
    raw_data[Data] --> cls_iterator[Data Iterator]
    cls_prompt --> cls_iterator
    config_model[models.yaml] --> cls_llm[LLM]
    cls_iterator --> cls_experiment[Experiment]
    cls_llm --> cls_experiment
    cls_args[Arguments] --> cls_experiment
    cls_experiment --> prompt_json["prompt.json<br/>Prompt template"]
    cls_experiment --> llm_json["llm.json<br/>LLM configuration"]
    cls_experiment --> generations_jsonl["generations.jsonl<br/>LLM call outputs"]
    cls_experiment --> metadata_jsonl["metadata.jsonl<br/>LLM call metadata"]
    cls_experiment --> args_json["args.json<br/>Experiment arguments"]
```
