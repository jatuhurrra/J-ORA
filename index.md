# J-ORA: A Robot Perception Framework for Japanese Object Identification, Reference Resolution, and Next Action Prediction

**And a Multimodal Dataset for Grounding Language and Vision in Japanese Robotics**

*Jesse Atuhurra, Hidetaka Kamigaito, Tatsuya Hiraoka*  
*NAIST, RIKEN Guardian Robot Project*
## 🎉 Accepted to IROS 2025!

[**Paper (IROS 2025)**](https://arxiv.org/abs/0000.00000) | [**Code**](https://github.com/jatuhurrra/J-ORA) | [**Dataset on HuggingFace**](https://huggingface.co/datasets/jatuhurrra/J-ORA)

<!--
<div style="width: 100%; background-color: #ffccf2; color: #800080; text-align: center; padding: 1em; font-size: 1.4em; font-weight: bold; border-radius: 8px; margin: 2em 0;">
  🎉 Accepted to IROS 2025!
</div>
-->

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Overview section (( ORIGINAL MAGENTA HERE cc00aa ))  ++++++++ ++++++++ ++++++++ --> 

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🌟 Overview
</div>

Robot perception is susceptible to object occlusions, constant object movements, and ambiguities within the scene that make *perceiving* those objects more challenging.
Yet, such *perception* is crucial to make the robot more useful in accomplishing tasks, especially when the robot *interacts* with humans.
We leverage *vision language models (VLMs)* to keep track of such object changes and also introduce an *object attribute annotations framework* to describe objects.

<div style="background-color:#e0ffff; border-left: 5px solid #00cccc; padding: 1em; margin-bottom: 1em;">
🌟 Highlight: Our framework is effective at improving VLM performance across tasks and non-English languages, e.g., Japanese.
</div>

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Problem section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🚀 The Problem
</div>

Modern robots must understand complex commands involving objects, references, and actions. 
Yet most datasets in robot vision-language learning are limited to English and rely on short prompts or synthetic setups. 
**J-ORA** addresses this gap by introducing a comprehensive **multimodal dataset grounded in Japanese**, containing real-world images and human-annotated linguistic references, actions, and object descriptions.
<div style="background-color:#ffe0f7; border-left: 5px solid #cc00aa; padding: 1em; margin-bottom: 1em;">
🌟 Highlight: Our framework is effective at improving VLM performance across tasks and non-English languages, e.g., Japanese.
</div>

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Motivation section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🚀 Motivation and J-ORA
</div>

In Japanese human-robot interaction (HRI), understanding ambiguous object references and predicting suitable actions are essential. 
Japanese presents unique linguistic challenges like elliptical expressions and non-linear syntax. 
Yet few datasets exist for grounded multimodal reasoning in Japanese, particularly in the context of robotic perception and manipulation.

J-ORA is designed to support **fine-grained multimodal understanding** and **language grounding** for tasks critical to domestic service robots, including object recognition, action prediction, and spatial reference disambiguation.

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Datasets section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  📦 Dataset Summary
</div>

**J-ORA** contains **1,100 real-world tabletop images** annotated with rich multimodal information:

- **Objects**: 6,500+ object instances from 80 household object categories.
- **Languages**: All instructions are provided in **Japanese**, with gold-standard annotations.
- **Tasks**: Each instance supports three core tasks:
  - **Object Identification (OI)**: "Which object is being referred to?"
  - **Reference Disambiguation (RD)**: "Which visual region best matches a given referring expression?"
  - **Action Prediction (AP)**: "What action is implied or requested?"

Each image is paired with:
- A Japanese natural language instruction.
- Polygon-level object segmentations.
- Bounding boxes and category labels.
- Reference relationships and action labels.

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Tasks Definitions section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🧠 Tasks in Detail
</div>

### 🟡 Object Identification (OI)
> Given a Japanese utterance and an image, identify which object is being referenced.

### 🔵 Reference Disambiguation (RD)
> Determine the bounding box or segmentation that best matches the given Japanese expression.

### 🔴 Action Prediction (AP)
> Predict the high-level action (e.g., pick, move, discard) implied by the instruction.

Each task is framed as a **multimodal question answering** problem and can be evaluated with standard accuracy, IoU, or retrieval metrics.

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Evaluations section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🧪 Evaluation and Baselines
</div>

We benchmark 6 leading **Vision-Language Models (VLMs)** on J-ORA, including:

- GPT-4o
- Gemini 1.5 Flash
- LLaVa Mistral
- Qwen-VL
- Japanese multimodal fine-tuned variants

We compare **zero-shot and fine-tuned** settings under multiple prompting styles.

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Key Findings section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  Key Findings
</div>

- **Limited Zero-Shot Transfer**: Even top models underperform in Japanese disambiguation and action understanding.
- **Multilingual Gaps Persist**: Despite multilingual training, open-source VLMs showed steep performance drops compared to GPT-4o.
- **Action Prediction is Harder**: Across models, AP lags behind OI and RD due to implicit reasoning required.
- **Fine-Tuning Helps**: Language-specific tuning significantly improves model performance, but gains vary by task.

## 📊 Quantitative Summary

| Feature | Value |
|--------|-------|
| Images | 1,100 |
| Object Categories | 80 |
| Instructions | 3,300+ |
| Avg Objects per Image | 5.9 |
| Languages | Japanese |
| Tasks | OI, RD, AP |

<!-- ++++++++++++++++ ++++++++ ++++++++ Below are the Use Cases/Resources/Citation/Contact sections  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  Use Cases/Resources
</div>

## 🔍 Use Cases

J-ORA supports research in:

- Japanese robot instruction following
- Multilingual grounding and disambiguation
- Vision-language model benchmarking
- Embodied AI with language grounding
- Fine-tuning of Japanese VLMs for manipulation tasks

## 🛠 Resources

- [**Code**](https://github.com/jatuhurrra/J-ORA): Task pipelines, training scripts, and evaluation metrics.
- [**Dataset**](https://huggingface.co/datasets/jatuhurrra/J-ORA): Full annotations and image data.
- [**Pretrained Models**](https://huggingface.co/jatuhurrra/J-ORA-models): Multilingual VLMs fine-tuned on J-ORA.

## 📄 Citation

If you use J-ORA in your research, please cite:

```
Coming soon...
```

## 📬 Contact

For questions and collaboration inquiries:

- **Jesse Atuhurra** — `atuhurra.jesse.ag2@naist.ac.jp`  
- **Tatsuya Hiraoka** — `tatsuya.hiraoka@mbzuai.ac.ae`

## 📜 License

The dataset and code are released under:

**CC BY-NC-SA 4.0**  
Attribution-NonCommercial-ShareAlike International License
