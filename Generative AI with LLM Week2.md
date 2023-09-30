# Generative AI with LLM Week2

- Limitation of In-context learning(ICL)
  - may not work well for smaller size LLM
  - examples take up space in context window, reducing the remaining room we have to include other information
- Solution: fine-tuning, a *supervise learning* process with labeled examples, updates weights of LLM.
  - labeled examples formed as `prompt-completion pairs` includes a task-specific *instruction* to LLM.
  - Instruction fine-tuning: *full fine-tuning* that updates all parameters
    - requires memory, compute budget to store all updated gradients and optimizers
    - we should use memory optimization and parallel computing strategies
    - Fine-Tuning Procedures: results in a new version of base model(the pre-trained LLM), called **instruct LLM**
      - use *prompt template libraries* to change existing unstructured dataset --> **instruction prompt dataset**
        - prompt instruction libraries: {{review_body}}
      - divide standard dataset into training, validation, test splits
      - pass them into Pre-trained LLM, generating completions
        - the output of LLM is a *probability distribution* across tokens
      - compare LLM completions with response specified in training data, then use *cross-entropy* as loss function
      - use the calculated loss to <u>update model weights</u> in back-propagation, improving model performance progressively.
      - use `validation_accuracy` from *validation dataset* to measure LLM performance.
      - after finished fine-tuning, use *test dataset* to evaluate final performance, gaining `test_accuracy`
- Fine-tuning on a <u>single</u> task:
  - good results can achieved only few examples(500~1000 is enough)
  - potential downside: catastrophic forgetting
    - reason: full fine-tuning increase model performance on a specific task, by updating original weights of LLM. However, it <u>degrade performance of other tasks</u>. Forgets previously learned information, as it learns new information
    - Definition: a common problem in DL
    - solutions:
      - check whether the catastrophic forgetting actually impact our use case.
      - if we want model to keep multitask generalized capabilities, we can perform <u>fine-tuning on multi-tasks at the same time</u>.
      - OR consider PEFT: train only a <u>small number of task-specific adapter layers and parameters</u>. Therefore, most of pre-trained LLM weights are unchanged.
- Multitask instruction fine-tuning:
  - datasets contain *examples* instructing model to complete various kinds of tasks.
  - drawback: it requires lots of data
  - One family models as example which is trained using multitask instruction fine-tuning:
    - **FLAN**
      - stands for `fine-tuned language net`
      - FLAN-T5
        - paper: <https://arxiv.org/pdf/2210.11416.pdf> with CoT reasoning tasks
        - summarization task in FLAN-T5 uses *SAMSum* datasets to summarize dialogue with prompt templates `{dialogue}` and `{summary}`, or fine-tuned with domain-specific dataset call *dialogsum*, or fine-tuned with our own data
        - Application: in chatbot, first summarize conversations to <u>identify actions customers required</u>.
- Model Evaluation
  - challenges for LLM: using traditional evaluation metric(accuracy, AUC, confusion matrix) is not fit, due to <u>non-deterministic and language based tasks</u>.
  - metrics:
    - ROUGE: Recall-Oriented Understudy for Gisting Evaluation
      - primarily used in text summarization, comparing it to other (ideal) summaries created by humans
      - 4 measures: N, L, W, S
      - ROUGE-N: package<https://pypi.org/project/rouge-score/>
        - ROUGE-N Recall=unigram matches/unigrams in reference
          - n-gram: a group of n words
          - N: # of word gram as a group
        - ROUGE-N Precision=unigram matches/unigrams in output
        - ROUGE-N F1=2*[precision*recall/(precision+recall)]
      - ROUGE-L(LCS): looks for *longest common sequence of words*
        - Advantage: not require consecutive matches, without predefined n-gram length
        - ROUGE-L Recall=LCS(Generated, Reference)/unigrams in reference
        - ROUGE-L Precision=LCS(Generated, Reference)/unigrams in output
        - ROUGE-L F1=2*[precision*recall/(precision+recall)]
        - Challenges:
          - When there exists **duplicates** in generated output, `clipping function` is needed
            - clipping function: limit the number of unigram matches to maximum count.
          - when generated output words are all present in reference, but in a different order.
      - ROUGE-S: skip-gram concurrence, allows for arbitrary gap允许中间跳过某些词
    - BLEU(Bilingual Evaluation Under Study) Score:
      - used for text translation, comparing to human generated <u>translation</u>
      - calculated with *average precision* over multiple n-gram *sizes*
- Benchmark: overall evaluation of model performance
  - vital: select right evaluation dataset to isolate specific model skills
  - eg:
    - GLUE(general understanding evaluation): sentiment analysis, Q&A
    - SuperGLUE: multi-sentence reasoning, reading comprehension<https://super.gluebenchmark.com/tasks>
    - beyond basic language understanding: HELM(multiMetric), Big-bench, MMLU
- PEFT Overview:
  - frozen most layers of LLM
  - add small number of trainable layers or parameters, fine-tuning only new components
  - can be performed in a single GPU
  - less prone to catastrophic forgetting
  - ❗️Types:
    - Selective: subset of original LLM parameters
    - Re-Parameterization: using a low-rank representation for original weights in attention layers
    - Additive: add trainable layers or parameters to model, inside the encoder or decoder after the feed-forward layers
      - Adapter
      - Soft Prompts: manipulate the input to achieve better performance, by adding trainable parameters to prompt embeddings, or by keeping the input fixed and retraining the embedding weights
        - **Prompt Tuning** technique
- LORA
  - freeze most of LLM original weights
  - inject 2 *rank decomposition matrices* with the same width and length, and constant rank $r$($r×W$; $r×L$)
    - selection of r: increase value of r has no effect on performance.$\in [1,8]$ is sufficient
  - Train the weights of smaller matrices
  - update model steps:
    - multiply the 2 decomposed matrices
    - add to original weights that have same dimensions
  - little impact on inference latency, by updating weights before inference
  - significantly reduce the number of trainable parameters
  - we can train different rank decomposition matrices for *different tasks*
  - can be evaluated by ROUGE: Full fine-tune >≈ LoRA fine-tune >> Baseline FLAN-T5
- QLoRA: combine quantization and LoRA together
- Soft-prompt
  - prompt tuning != prompt engineering
    - drawback of prompt engineering:
      - it requires lots of manual effort to write and try different prompts
      - limited by the length of context window
    - prompt tuning: add additional unseenable tokens(在现实语言中找不到对应的词语) to prompts, leaving it to supervised learning to determine their optimal values by gradient descent
      - soft-prompt:
        - the set of additional trainable tokens
        - prepended to input embedding vectors
        - has the *same length* as token vectors
        - amount: $\in [20,100]$ tokens
        - virtual tokens can take on any value within the *continuous* multidimensional embedding space
          - with KNN, the words closest to additional soft prompt tokens have similar meanings
        - we can train a subset for one task, and a different subset for another, swapping tasks at inference time flexibly
        - Evaluation by SuperGLUE: multi-task fine tuning >≈ full fine tuning > prompt tuning >> prompt engineering
          - as <u>model size increasing</u>, the performance score of multitask,full time, and prompt tuning get the same value
      - hard-prompt: revise *discrete* inputs directly, without optimizing through gradient descent ability.
      - the weights of model frozen, soft prompt trained over time