---
marp: true
theme: uncover
class: invert
paginate: true
style: |
  section { font-size: 24px; }
---


<!-- _class: lead invert -->

# **ExAI as an Audit Layer for Decoding‑Bias Measurement**

### Ivan Chabanov & Aleksandr Michailov

---

## 📌 Project Idea

**Core concept:**  
Decoding strategies (temperature, top‑k, top‑p, anti‑repetition) change social bias in GPT‑2 generated text.  
Aggregate regard metrics (negative/neutral/positive/other) are useful but **black‑box**.

**ExAI extension** adds an **audit layer**:
- Train a BERT‑based regard classifier
- Explain its predictions with **Layer‑wise Relevance Propagation (LRP)**
- Provide token‑level heatmaps to inspect *why* a generation received a particular label

> *“Decoding analysis tells **how** bias metrics change; ExAI helps inspect **why**.”*

---

## 📚 Relevant Theory

### Explainability need
Raw classifier accuracy hides *how* decisions are made. We need to know if the model relies on:
- Sentiment words
- Demographic terms
- Repetition / generation artifacts

---
## 📚 Relevant Theory
### Layer‑wise Relevance Propagation (LRP)
- Redistributes the output score **backward** onto input tokens
- For a linear layer:  
  $$ z_{ij} = x_i w_{ij}, \quad R_i = \sum_j \frac{z_{ij}}{\sum_i z_{ij} + \epsilon} R_j $$
- **Positive relevance** → supports the target class  
- **Negative relevance** → opposes it

Our implementation applies epsilon‑LRP to the classifier head + approximations for attention, residuals, and layer norm.

---

## 📚 Relevant Theory
### Layer‑wise Relevance Propagation (LRP)
![bg](images/backward.png)

---

## 🛠️ Implementation

### 1. Data & training
- 325 labeled regard examples (258 train, 31 val, 36 test) – unbalanced (`other` has only 23)
- Fine‑tuned `bert-base-uncased` for 4‑class regard classification

### 2. ExAI inference pipeline

```python
runner = ExAIInferenceRunner(checkpoint_dir, device="cpu")
explainer = TransformerLRPExplainer(runner)

text = "The nurse helped the patient."
inference = runner.predict_text(text)
explanation = explainer.explain_inference(inference)
```

### 3. Audit benchmark
- 12 real scored generations from the decoding pipeline
- Covers 3 scoring labels, 3 demographics, 3 prompt types (approximate)
- For each: predict label → compute LRP → save JSON + HTML heatmap

### 4. Validation
- Faithfulness: remove top‑attributed tokens vs random
- Sensitivity: perturb input (rephrase, insertion, punctuation)

---

![bg](images/main_pipeline.png)

---

## 📊 Results

### Classifier performance (held‑out test)

| Class     | F1    | Support |
|-----------|-------|---------|
| negative  | 0.800 | 13      |
| neutral   | 0.588 | 10      |
| positive  | 0.545 | 10      |
| other     | 0.000 | 3       |

**Accuracy: 0.639 | Macro F1: 0.483**

**Agreement with released scorer (`sasha/regardv3`):** 0.694 accuracy, 0.494 macro F1

### Benchmark on 12 generated examples
- **Accuracy: 0.250** (only 3/12 agree with scoring labels)
- Domain shift: generated continuations differ from training data

---

## 🔍 Example Explanations

### ✅ Insightful (agreement case)
- Scoring & BERT both = `neutral`
- High relevance on neutral target: `very`, `Eglazzi`, punctuation
- Demographic tokens (`black`, `man`) get **negative** relevance for neutral class  
*Heatmap shows inspectable, plausible evidence.*

### ⚠️ Ambiguous (demographic‑token relevance)
- Scoring = `negative`, BERT = `neutral` (disagreement)
- Target class = `negative` → high positive relevance includes `black`, `woman`, `she`, `her`  
👉 *Surfaces audit question: why identity tokens matter for negative regard?*

---

![bg](images/more_token_relev.png)

---

![bg](outputs/plots/fbe608112493c39dd4d4_regard_distribution.png)

---

## ✅ Faithfulness & Sensitivity

### Faithfulness (token removal)
- Top‑attribution removal mean drop: **0.0123**
- Random removal drop: **0.0152**  
❌ Top removal not clearly larger → explanations are **heuristic**, not strong causal proof

### Sensitivity (top‑5 token overlap)

| Perturbation       | Overlap |
|--------------------|---------|
| Benign rephrase    | 0.875   |
| Neutral insertion  | 0.667   |
| Punctuation change | 0.446   |

✅ Stable under rephrasing; weaker on punctuation (important for generated text with artifacts)

---

## 👥 Team Contributions

| Team Member          | Contributions |
|----------------------|----------------|
| **Ivan Chabanov**    | - Decoding‑bias pipeline (GPT‑2 generation, regard scoring, aggregate metrics)<br>- Prompt bank design<br>- Generation of 72,000 scored examples<br>- Anti‑repetition & gap analysis |
| **Aleksandr Michailov** | - ExAI module design & implementation<br>- BERT regard classifier training<br>- LRP integration for Transformers<br>- Faithfulness & sensitivity benchmarks<br>- Heatmap rendering & audit benchmark |

*Both authors contributed to writing, analysis, and interpretation of results.*

---

## 🔚 Conclusion

- ExAI makes the bias measurement pipeline **transparent** and **inspectable**
- Token‑level heatmaps are **useful audit evidence** but not causal proof
- Faithfulness results caution against over‑interpreting single examples
- **Main takeaway:** XAI is valuable *when combined with validation* – it surfaces questions, not final answers

> “The ExAI extension does not replace aggregate metrics – it complements them with local, inspectable evidence.”

---

## 📖 References

- Bach et al. (LRP) – [PLOS ONE](https://doi.org/10.1371/journal.pone.0130140)
- Montavon et al. (LRP overview) – [Springer](https://doi.org/10.1007/978-3-030-28954-6_10)
- Devlin et al. (BERT) – [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Sheng et al. (bias in generation) – [ACL Anthology](https://aclanthology.org/D19-1339/)
- Holtzman et al. (decoding) – [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Released regard scorer – [`sasha/regardv3`](https://huggingface.co/sasha/regardv3)
- Project repository – [ChabanovX/decoding-amplifies-bias](https://github.com/ChabanovX/decoding-amplifies-bias)

<!-- _class: small -->

🙌 **Thank you!** – Questions?