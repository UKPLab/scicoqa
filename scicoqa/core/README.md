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
    config_prompt[prompt.yaml] --> cls_prompt[Prompt]
    raw_data[Data] --> cls_iterator[Data Iterator]
    cls_prompt --> cls_iterator
    config_model[model.yaml] --> cls_llm[LLM]
    cls_iterator --> cls_experiment[Experiment]
    cls_llm --> cls_experiment
    cls_args[Arguments] --> cls_experiment
    cls_experiment --> prompt.json[prompt.json 
    hold the prompt tempalte
    ]
    cls_experiment --> llm.json[llm.json
        LLM configuration
    ]
    cls_experiment --> generations.jsonl[generations.jsonl
        output of the LLM calls
    ]
    cls_experiment --> metadata.jsonl[metadata.jsonl
        Metadata about the LLM calls
    ]
    cls_experiment --> args.json[args.json
        Arguments used for the experiment
    ]
```
