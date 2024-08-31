## CMx AI

### Model Details

* **Model Description:** CMx AI is a large language model (LLM) designed for highly secure enterprise-level legal, procurement and business applications. It excels in understanding, analyzing, and generating large set of folders and documents such as legal contracts and handles bulk document generation, compliance analysis, and response documents as per specification analyzed by AI using it's properietary auto filling (Fill in AI) and AI based autogen capabiliies .
* **Developed by:** Sysintellects LLC
* **Model type:** Large Language Model (LLM)
* **Language(s):** English
* **License:** Proprietary
* **Finetuned from model:** Llama 14b (quantized)
* **Base model size:** 7b to 14b parameters
* **Finetuned model size:** 1b, 14b

**Model Sources:**

* **Repository:** Private repository on Gitlab
* **Paper:** [We will release a blog on the model we release] 

## Uses

**Direct Use:**

* **Contextual AI:** CMx AI can analyze and summarize a wide range of legal contracts, including employment contracts, non-disclosure agreements, sales agreements, service contracts, and more specialized types. It can also help identify potential risks and generate legal templates based on user inputs. 
* **Auto Gen AI:**  CMx AI can generate proposals, contracts, reports, and letters based on user inputs and instructions.  
* **Fill In AI:** CMx AI can fill out online forms, generate financial statements, or complete tax forms based on provided data and instructions.
* **Compliance AI:** CMx AI can help users understand and comply with relevant regulations by providing summaries of legal requirements, identifying potential compliance risks, and generating reports.
* **Bid Booster AI:** CMx AI analyzes tender documents and identifies relevant opportunities for users based on company information and specific criteria and auto generates bid responses document packages.

**Downstream Use:**

* Large Document Set and Contract Documents Summarization 
* Metadata Extraction for Key Information
* Natural Language Queries using a Chat Interface on private respository documents

**Out-of-Scope Use:**

* Excluding all of Direct and Downstream Usage

## Bias, Risks, and Limitations

* **Potential Biases:**  CMx AI will likely reflect biases present in the existing dataset of contracts within the system.  We're aware of this potential bias and are currently investigating methods to mitigate it. 
* **Mitigation Strategies:** Currently, we don't have specific data de-biasing techniques in place. However, we are actively researching and exploring options to reduce bias in the model's outputs. 
* **Risks:** As with any LLM, there are risks associated with the use of CMx AI, including the potential for generating misleading or inaccurate information. These risks can be further exacerbated by the model's reliance on existing data within the system. 

## Recommendations

* **Users should be aware that CMx AI may generate biased or inaccurate outputs. They should critically evaluate the model's suggestions and consult with legal professionals for critical decisions.** 

## Training Details

**Training Data:**

* **Data Source:** We used a combination of publicly available legal datasets, including The Pile of Law (which includes legal analyses, court opinions, government publications, contracts, statutes, regulations, and casebooks) and LEGAL-BERT models (trained on diverse legal text from legislation, court cases, and contracts), along with our internal dataset of contracts.  
* **Data Preprocessing:**  We use a combination of natural language processing techniques and domain-specific knowledge to preprocess the data. This includes tokenization, cleaning, normalization, and data augmentation.

**Training Procedure:**

* **Training Algorithm:** We used the Llama architecture, which is a type of Transformer-based language model.
* **Training Hyperparameters:** We used a batch size of 64, a learning rate of 1e-5, and trained for 500 epochs. Note that these hyperparameters are subject to change based on our experiments and ongoing optimization efforts.

## Evaluation

* **Testing Data:** We used the FairLex and LexGLUE benchmarks for evaluation.
    * **FairLex:** This benchmark evaluates legal text classification and judgment prediction across five languages: English, German, French, Italian, and Chinese.
    * **LexGLUE:** This benchmark focuses on English legal NLP tasks, including classification and multiple-choice question answering.
* **Metrics:** We used standard evaluation metrics commonly used for LLMs, especially those relevant to Llama, such as accuracy, precision, recall, F1-score, perplexity, and BLEU score. 
* **Results:**  We're currently in the process of conducting comprehensive evaluations and will provide a detailed summary of results in the future.

## Technical Specifications

* **Model Architecture:** CMx AI is based on the LLaMA 3 architecture, which is a Transformer-based language model.  The LLaMA 3 model we fine-tune has 32 layers, 32 attention heads, and a vocabulary size of 128,256 tokens.  The 8B version uses Grouped-Query Attention (GQA) for enhanced scalability and longer context handling, allowing for processing sequences up to 8,192 tokens.
* **Objective Function:**  The objective function used during the training of CMx AI is autoregressive language modeling. This aims to minimize the cross-entropy loss between the predicted and actual output tokens given the input context.

**Compute Infrastructure:**

* **Hardware:** GeForce RTX 4090
* **Software:** Python, Pytorch, Unsloth

## Glossary

* **LLaMA 3:**  A large language model developed by Meta, which is known for its efficiency and high performance on various tasks. You can find more information about LLaMA 3 on the official website: [https://llama.meta.com/llama3/](https://llama.meta.com/llama3/).  
* **The Pile of Law:** This dataset, consisting of over 256GB of English legal and administrative text, is a valuable resource for training LLMs focused on legal tasks.  It was used in the training of the LegalBERT-Large model. 
* **LEGAL-BERT:** This set of models was pretrained on a 12GB dataset of diverse legal text from fields like legislation, court cases, and contracts. The data was gathered from publicly available resources.
* **FairLex:** A multilingual benchmark for legal text classification and legal judgment prediction tasks across 5 languages (English, German, French, Italian, Chinese).
* **LexGLUE:** A benchmark for English legal NLP tasks including classification and multiple-choice QA. 
* **Grouped-Query Attention (GQA):** A technique used in LLaMA 3 to enhance scalability and handle longer contexts, allowing for processing sequences up to 8,192 tokens.
* **Autoregressive Language Modeling:** A common approach used to train language models, where the model predicts the next token in a sequence based on the preceding tokens. 

## More Information

* **LLaMA 3:** [https://llama.meta.com/llama3/](https://llama.meta.com/llama3/) 
* **Unsloth:** [https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
* **Scaling Laws of RoPE-based Extrapolation:** [https://arxiv.org/abs/2310.05209](https://arxiv.org/abs/2310.05209) 

## Model Card Authors

* Ashok Poudel
* Nazar Abdul

## Model Card Contact

* ashok.poudel@sysintellects.com 
* nazar.abdul@sysintellects.com 
