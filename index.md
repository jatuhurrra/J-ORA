# J-ORA: Japanese Object Reference and Action Dataset

**A Multimodal Dataset for Grounding Language and Vision in Japanese Robotics**

*Jesse Atuhurra, Iqra Ali, Tomoya Iwakura, Hidetaka Kamigaito, Tatsuya Hiraoka*  
*NAIST, MBZUAI, Fujitsu Research*

[**Paper (IROS 2025)**](https://arxiv.org/abs/0000.00000) | [**Code**](https://github.com/jatuhurrra/J-ORA) | [**Dataset on HuggingFace**](https://huggingface.co/datasets/jatuhurrra/J-ORA)

---

<div style="background-color:#f2f8ff; border-left: 5px solid #007acc; padding: 1em; margin-bottom: 1em;">

## 🌟 Overview

Welcome to our benchmark website! This section summarizes the motivation, design, and findings of our work.

</div>


## 🌟 Overview

Modern robots must understand complex commands involving objects, references, and actions. Yet most datasets in robot vision-language learning are limited to English and rely on short prompts or synthetic setups. **J-ORA** addresses this gap by introducing a comprehensive **multimodal dataset grounded in Japanese**, containing real-world images and human-annotated linguistic references, actions, and object descriptions.

---

## 🚀 Motivation

In Japanese human-robot interaction (HRI), understanding ambiguous object references and predicting suitable actions are essential. Japanese presents unique linguistic challenges like elliptical expressions and non-linear syntax. Yet few datasets exist for grounded multimodal reasoning in Japanese, particularly in the context of robotic perception and manipulation.

J-ORA is designed to support **fine-grained multimodal understanding** and **language grounding** for tasks critical to domestic service robots, including object recognition, action prediction, and spatial reference disambiguation.

---

## 📦 Dataset Summary

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

---

## 🧪 Evaluation and Baselines

We benchmark 6 leading **Vision-Language Models (VLMs)** on J-ORA, including:

- GPT-4o
- Gemini 1.5 Flash
- LLaVa Mistral
- Qwen-VL
- Japanese multimodal fine-tuned variants

We compare **zero-shot and fine-tuned** settings under multiple prompting styles.

### Key Findings

- **Limited Zero-Shot Transfer**: Even top models underperform in Japanese disambiguation and action understanding.
- **Multilingual Gaps Persist**: Despite multilingual training, open-source VLMs showed steep performance drops compared to GPT-4o.
- **Action Prediction is Harder**: Across models, AP lags behind OI and RD due to implicit reasoning required.
- **Fine-Tuning Helps**: Language-specific tuning significantly improves model performance, but gains vary by task.

---

## 🧠 Tasks in Detail

### 🟡 Object Identification (OI)
> Given a Japanese utterance and an image, identify which object is being referenced.

### 🔵 Reference Disambiguation (RD)
> Determine the bounding box or segmentation that best matches the given Japanese expression.

### 🔴 Action Prediction (AP)
> Predict the high-level action (e.g., pick, move, discard) implied by the instruction.

Each task is framed as a **multimodal question answering** problem and can be evaluated with standard accuracy, IoU, or retrieval metrics.

---

## 🔍 Use Cases

J-ORA supports research in:

- Japanese robot instruction following
- Multilingual grounding and disambiguation
- Vision-language model benchmarking
- Embodied AI with language grounding
- Fine-tuning of Japanese VLMs for manipulation tasks

---

## 🛠 Resources

- [**Code**](https://github.com/jatuhurrra/J-ORA): Task pipelines, training scripts, and evaluation metrics.
- [**Dataset**](https://huggingface.co/datasets/jatuhurrra/J-ORA): Full annotations and image data.
- [**Pretrained Models**](https://huggingface.co/jatuhurrra/J-ORA-models): Multilingual VLMs fine-tuned on J-ORA.

---

## 📊 Quantitative Summary

| Feature | Value |
|--------|-------|
| Images | 1,100 |
| Object Categories | 80 |
| Instructions | 3,300+ |
| Avg Objects per Image | 5.9 |
| Languages | Japanese |
| Tasks | OI, RD, AP |

---

## 📄 Citation

If you use J-ORA in your research, please cite:

```
coming soon
```

## 📬 Contact

For questions and collaboration inquiries:

- **Jesse Atuhurra** — `atuhurra.jesse.ag2@naist.ac.jp`  
- **Tatsuya Hiraoka** — `tatsuya.hiraoka@mbzuai.ac.ae`

---

## 📜 License

The dataset and code are released under:

**CC BY-NC-SA 4.0**  
Attribution-NonCommercial-ShareAlike International License
