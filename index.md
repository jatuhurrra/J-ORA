# J-ORA: A Robot Perception Framework for Japanese Object Identification, Reference Resolution, and Next Action Prediction

Including a **Multimodal Japanese Dataset for Grounding Language and Vision in Robotics**

*[<u>Jesse Atuhurra</u>](https://www.2021atajesse.com/)<sup>1,2</sup>, [<u>Hidetaka Kamigaito</u>](https://sites.google.com/site/hidetakakamigaito)<sup>1</sup>, [<u>Taro Watanabe</u>](https://sites.google.com/site/tarowtnb/)<sup>1</sup>, [<u>Koichiro Yoshino</u>](https://pomdp.net/)<sup>1,2</sup>*  
<sup>1</sup> NAIST  
<sup>2</sup> RIKEN Guardian Robot Project

## 🎉 Accepted to IROS 2025 🤖!

[**Paper (IROS 2025)**](https://arxiv.org/abs/0000.00000) | [**Code**](https://github.com/jatuhurrra/OpenPerception) | [**Dataset on HuggingFace**](https://huggingface.co/datasets/atamiles/J-ORA)

<!--
<div style="width: 100%; background-color: #ffccf2; color: #800080; text-align: center; padding: 1em; font-size: 1.4em; font-weight: bold; border-radius: 8px; margin: 2em 0;">
  🎉 Accepted to IROS 2025!
</div>
-->

<video style="width: 100%; height: auto;" controls>
  <source src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/IROSvideo.mp4?raw=true" type="video/mp4">
  Your browser does not support the video tag.
</video>

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Overview section (( ORIGINAL MAGENTA HERE cc00aa ))  ++++++++ ++++++++ ++++++++ --> 

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🌟 Overview
</div>

Robot perception is susceptible to object occlusions, constant object movements, and ambiguities within the scene that make *perceiving* those objects more challenging.
Yet, such *perception* is crucial to make the robot more useful in accomplishing tasks, especially when the robot *interacts* with humans.
We introduce an **object-attribute annotations framework** to describe objects, from the robot's **ego-centric view**, and keep track of changes in object attributes.
Then, we leverage *vision language models (VLMs)* to interpret objects and their attributes. After that, the VLMs facilitate the robot to accomplish tasks such as **object identification, reference resolution, and next-action prediction**. 

<div style="background-color:#ffe0f7; border-left: 5px solid #00cccc; padding: 1em; margin-bottom: 1em;">
🌟 Highlight: Our framework is effective in representing dynamic *object changes* in non-English languages, e.g., Japanese.
</div>

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Problem section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🧩 The Problem
</div>

Robot perception in the real world faces challenges of **(1) dynamically changing scenes in which the people and objects undergo constant changes, and (2) the co-existence of similar objects in a scene**. 
Moreover, modern robots must understand complex commands involving objects, references, and actions. 
Lastly, robots need to comprehend and make sense of commands and dialogues in numerous languages. 
Yet, most datasets in robot vision-language learning are limited to English and rely on short prompts or synthetic setups. 
**J-ORA** addresses this gap by introducing a comprehensive **multimodal dataset grounded in Japanese**, containing real-world images and human-annotated linguistic references, actions, and object descriptions.

<div style="background-color:#ffe0f7; border-left: 5px solid #cc00aa; padding: 1em; margin-bottom: 1em;">
🧩 Highlight: We aim to train robots to excellently perceive and understand the scene despite the dynamic changes in the scene.
</div>

<img src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/scenes.png?raw=true" style="width: 100%; height: auto;" alt="Scenes Figure">

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Motivation section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🚀 Motivation and J-ORA
</div>

In Japanese human-robot interaction (HRI), understanding ambiguous object references and predicting suitable actions are essential. 
Japanese presents unique linguistic challenges like elliptical expressions and non-linear syntax. 
Yet few datasets exist for grounded multimodal reasoning in Japanese, particularly in the context of robotic perception and manipulation.

J-ORA is designed to support **fine-grained multimodal understanding** and **language grounding** for tasks critical to domestic service robots, including object recognition, action prediction, and spatial reference disambiguation.

<img src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/PaperIntroFig.png?raw=true" style="width: 100%; height: auto;" alt="Intro Figure">

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Datasets section  ++++++++ ++++++++ ++++++++ --> 

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  📦 Dataset Summary
</div>

**J-ORA** contains **142 real-world image-dialogue pairs** annotated with rich multimodal information:

- **Objects**: 1,817 object attribute annotations from 160 unique object classes.
- **Dialogues**: 142 dialogues recorded in the real world.
- **Dialogue Turns**: 15 Average turns per dialogue.
- **Utterances**: 2,131 dialogue utterances.
- **Languages**: All instructions are provided in **Japanese**, with gold-standard annotations.
- **Tasks**: Each instance supports three core tasks:
  - **Object Identification**: "Which object is being referred to?"
  - **Reference Reference**: "Which visual region best matches a given referring expression?"
  - **Action Prediction**: "What action is implied or requested?"

Each image is paired with:
- Dialogue texts in Japanese.
- Object-attribute annotations.
- Bounding boxes and category labels.
- Reference relationships.

Each object is described using these features: **category, color, shape, size, material, surface texture, position, state, functionality, brand, interactivity, and proximity to the person**.

<img src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/DataCollectionPipeline.png?raw=true" style="width: 100%; height: auto;" alt="Data Collection Pipeline Figure"> 

## 📊 Quantitative Summary of the J-ORA dataset

| **Feature**                    | **Value**              |
|--------------------------------|------------------------|
| Hours                          | 3 hrs 3 min 44 sec     |
| Unique dialogues               | 93                     |
| Total dialogues                | 142                    |
| Utterances                     | 2,131                  |
| Sentences                      | 2,652                  |
| Average turns per dialogue     | 15                     |
| Average duration per turn      | 77 sec                 |
| Total turns                    | 2,131                  |
| Image-Dialogue pairs           | 142                    |
| Unique object classes          | 160                    |
| Object attribute annotations   | 1,817                  |
| Languages                      | Japanese               |


<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Tasks Definitions section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🧠 Tasks in Detail
</div>

### 🟡 Object Identification (OI)
> Given a Japanese utterance and an image, identify all objects mentioned in the utterance.

### 🔵 Reference Resolution (RR)
> Given a Japanese utterance and an image, and the object mentions from the OI task above, describe where in the image the mentioned objects occur.

### 🔴 Next Action Prediction (AP)
> From the objects identified in the dialogue utterance in the OI task, and the locations in the image described in the RR task, predict the next most probable high-level action (e.g., pick, move, discard) implied by the instruction.

The three tasks are framed as an **end-to-end multimodal perception** problem and are performed by VLMs. Performance is evaluated with standard accuracy as the major metric.

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Evaluations section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  🧪 Evaluation and Baselines
</div>

We benchmark these leading **Vision-Language Models (VLMs)** on J-ORA, including:
- **(i) Proprietary:** Claude 3.5 Sonnet, Gemini 1.5 Pro, GPT-4o. 
- **(ii) General open-source:** Llava v1.6 Mistral 7B, Llava v1.6 Mistral 13B, Qwen2VL 7B Instruct.
- **(iii) Japanese open-source:** EvoVLM-JP-v1-7B, Japanese-stable-vlm, Bilingual-gpt-neox-4b-minigpt4.

We compare **zero-shot and fine-tuned** settings for VLMs **with or without object attributes**.
<div align="center">
<img src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/results1.png?raw=true" style="width: 60%; height: auto;" alt="Data Collection Pipeline Figure">
<img src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/results2.png?raw=true" style="width: 60%; height: auto;" alt="Data Collection Pipeline Figure">  
<img src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/results3.png?raw=true" style="width: 70%; height: auto;" alt="Data Collection Pipeline Figure">  
<img src="https://github.com/jatuhurrra/OpenPerception/blob/main/assets/results4.png?raw=true" style="width: 70%; height: auto;" alt="Data Collection Pipeline Figure">  
</div>

<!-- ++++++++++++++++ ++++++++ ++++++++ This is the Key Findings section  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
 🔍 Key Findings
</div>

- **Multilingual Gaps Persist**: Despite multilingual training, open-source VLMs showed steep performance drops compared to GPT-4o across all tasks.
- **Fine-Tuning Helps**: Language-specific tuning significantly improves model performance, but gains vary by task.
- **Reference Resolution Bottleneck**: Across models, RR lags behind OI and AP due to the multimodal reasoning required.
- **Limited Object Affordance**: With or without finetuning, object affordance remains a challenge for all VLMs.

<!-- ++++++++++++++++ ++++++++ ++++++++ Below are the Use Cases/Resources/Citation/Contact sections  ++++++++ ++++++++ ++++++++ -->

<div style="width: 100%; background-color: #b2d8d8; color: #800080; text-align: center; padding: 0.75em 0; font-size: 1.5em; font-weight: bold; margin: 2em 0;">
  Use Cases, Resources, Citation, Contact & License
</div>

## 🔍 Use Cases

J-ORA supports research in:

- Japanese robot instruction following
- Multilingual grounding and reference resolution
- Vision-language model benchmarking
- Embodied AI with language grounding
- Fine-tuning of Japanese VLMs for HRI tasks

## 🛠 Resources

- [**Code**](https://github.com/jatuhurrra/OpenPerception): Task pipelines, training scripts, and evaluation metrics.
- [**Dataset**](https://huggingface.co/datasets/jatuhurrra/J-ORA): Full annotations and image data.

  The data introduced in this project extends the [**J-CRe3**](https://github.com/riken-grp/J-CRe3) data.

## 📄 Citation

If you use J-ORA in your research, please cite these two resources:

```
IROS 2025 coming soon...
```

And J-CRe3 below

```
@inproceedings{ueda-2024-j-cre3,
  title     = {J-CRe3: A Japanese Conversation Dataset for Real-world Reference Resolution},
  author    = {Nobuhiro Ueda and Hideko Habe and Yoko Matsui and Akishige Yuguchi and Seiya Kawano and Yasutomo Kawanishi and Sadao Kurohashi and Koichiro Yoshino},
  booktitle = {Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  month     = may,
  year      = {2024},
  url       = {https://aclanthology.org/2024.lrec-main.829},
  pages     = {9489--9502},
  address   = {Turin, Italy},
}
```

## 📬 Contact

For questions and collaboration inquiries:

- **Jesse Atuhurra** — `atuhurra.jesse.ag2@naist.ac.jp`  
- **Koichiro Yoshino** — `koichiro.yoshino@riken.jp`

## 📜 License

The dataset and code are released under: **CC BY-SA 4.0**,  i.e., Attribution-ShareAlike International License
