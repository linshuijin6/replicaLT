# replicaLT

A two-stage MRI-to-PET generation framework with plasma-guided conditioning.

> - **`plasma_train.py` is the main method of the project**
> - **root `train.py` is only the legacy / baseline implementation** and should not be treated as the primary entrypoint for the current method

---

## 1. Project overview

`replicaLT` is a **two-stage MRI-to-PET generation framework**.

The overall design is:

1. **Stage A: representation alignment**
   - Learn reusable multimodal alignment across TAU image tokens, MRI tokens, diagnosis labels, and plasma values.
   - Produce the semantic basis for later plasma-conditioned generation.

2. **Stage B: plasma-guided PET generation**
   - Load precomputed `plasma_emb` for each subject.
   - Use `plasma_emb` as a conditional token together with modality-specific text features.
   - Train a **3D MRI-to-PET generator** in a diffusion / rectified-flow-style formulation.

This repository should therefore **not** be interpreted as a single end-to-end script. The alignment stage and the generation stage are deliberately decoupled.

---

## 2. Main entrypoints in this branch

### Current project mainline
- **`plasma_train.py`**

This is the main training script for the current method.

Its role is to:
- load precomputed `plasma_emb`
- combine plasma embeddings with modality-specific text features
- build a fixed `(B, 2, 512)` conditioning tensor
- perform **3D MRI -> PET** generation

### Upstream alignment stage
- **`adapter_v2/train.py`**

This is the main entrypoint for the representation/alignment stage.

Its role is to:
- train multimodal alignment using TAU, MRI, diagnosis, and plasma information
- provide the semantic basis used to derive plasma-guided conditioning in Stage B

### Legacy / comparison-only code
- **`train.py`**

This script is **not** the primary method in the current branch.

It should be treated as:
- a **legacy implementation**
- a **baseline / comparison method**
- a historical reference for earlier conditioning strategies

When discussing the current project method, always start from **`plasma_train.py`**, not from `train.py`.

---

## 3. Stage A: alignment training (`adapter_v2/train.py`)

### Purpose
Stage A learns a reusable multimodal alignment model across:
- TAU image tokens
- MRI image tokens
- diagnosis labels / class prompts
- plasma numerical values

The goal of this stage is **not** final PET synthesis. Its purpose is to build a strong semantic representation layer, especially for **plasma-related conditioning**.

### Core code path
When analyzing Stage A, prioritize the following files:

1. `adapter_v2/train.py`
2. `adapter_v2/dataset.py`
3. `adapter_v2/models.py`
4. `adapter_v2/losses.py`
5. `adapter_v2/precompute_cache.py`

### Practical interpretation
This stage provides the upstream representational foundation used later by Stage B. In other words:

- Stage A learns **how plasma, image, and diagnosis information align**
- Stage B consumes the resulting plasma-side semantic information via precomputed embeddings

---

## 4. Stage B: plasma-guided generation (`plasma_train.py`)

### Purpose
This is the **main method** of the current branch.

Its objective is to synthesize PET from MRI while using plasma-derived semantic conditioning.

### Key idea
The generation model does **not** use plasma as a supervised regression target.

Instead, plasma is injected as a **conditioning token**:

- **Token 0:** precomputed `plasma_emb`
- **Token 1:** tracer-specific modality text embedding

These two tokens are concatenated into a conditioning tensor of shape:

- **`(B, 2, 512)`**

This design allows the conditional information to be injected **without modifying the core diffusion UNet architecture**.

### Why this matters
The key contribution of this  is not simply “MRI to PET”.
It is:

> **MRI-to-PET generation under plasma-guided conditioning**

That means `plasma_train.py` should be read as the definitive implementation of the current project idea.

---

## 5. Precomputed plasma embeddings

### Required preprocessing
Before running `plasma_train.py`, the project expects precomputed plasma embeddings.

Relevant script:
- **`precompute_plasma_emb.py`**

### Role in the pipeline
This script generates subject-level embedding cache files, typically one embedding per subject.

`plasma_train.py` then loads those cached embeddings and uses them directly during generation.

### Design implication
This confirms that the project is **explicitly two-stage and decoupled**:

- Stage A learns alignment / representation
- Stage B reads the cached plasma representation

So, when debugging or evaluating the model, keep in mind:
- improvements may come from the **generator architecture**, or
- from better **upstream plasma embeddings**, or
- from both

---

## 6. Training formulation in the main 

### Generation granularity
The current mainline generation stage is **3D**, not 2D slice generation.

Even if the upstream representation stage may use token/slice-style processing, the final synthesis stage uses:
- 3D image preprocessing
- 3D model components
- 3D MRI/PET volumes

So the project should be described as:

> **a two-stage 3D MRI-to-PET generation framework with plasma-guided conditioning**

### Optimization style
`plasma_train.py` follows a **rectified-flow-style / velocity-field regression** formulation rather than a plain direct voxel regression.

At a high level, the script constructs an interpolation state between MRI and PET and learns a target update / velocity field.

When documenting or comparing methods, avoid describing the current mainline as just “a standard diffusion baseline”.
A more accurate description is:

> **3D conditional diffusion / rectified-flow-style MRI-to-PET generation with plasma-guided conditioning**

---

## 7. Tracer strategy

### Framework capability
The code structure supports independent generation for multiple tracers, including:
- FDG
- AV45
- TAU

Each tracer has its own modality-specific text token construction.

### Current  default
In the latest branch, the active/default experimental configuration is **TAU-focused**.

So when reading experiment code or logs, do not assume that the branch is currently running a fully joint multi-tracer training setup.

A safer statement is:

> The framework supports multiple tracer-specific targets, but the current latest branch is configured primarily for **TAU generation**.

---

## 8. How to read this repository efficiently

### Recommended order for current-method understanding
1. **`plasma_train.py`**
2. **`precompute_plasma_emb.py`**
3. **`adapter_v2/train.py`**
4. **`train.py`** (legacy only)

### If your question is about...

#### A. Current method / main generation pipeline
Start with:
- `plasma_train.py`

Focus on:
- how `plasma_emb` is loaded
- how condition tokens are constructed
- how MRI/PET data are preprocessed
- how the 3D generator is trained
- how validation and visualization are implemented

#### B. Where plasma semantics come from
Start with:
- `precompute_plasma_emb.py`
- `adapter_v2/train.py`

Focus on:
- how Stage A learns multimodal alignment
- how embeddings are computed and cached

#### C. Legacy baseline behavior
Start with:
- `train.py`

Only use it to answer:
- what the old conditioning design looked like
- how the legacy baseline differs from `plasma_train.py`

---

## 9. Fast code navigation summary

### Current method
- `plasma_train.py`

### Plasma embedding cache generation
- `precompute_plasma_emb.py`

### Alignment training
- `adapter_v2/train.py`
- `adapter_v2/dataset.py`
- `adapter_v2/models.py`
- `adapter_v2/losses.py`
- `adapter_v2/precompute_cache.py`

### Legacy / baseline
- `train.py`

---

## 10. Recommended terminology for future discussion

To avoid confusion in later experiments, notes, or papers, use the following naming consistently.

### Whole project
**replicaLT** = a two-stage MRI-to-PET generation framework

### Stage A
**alignment stage** / **adapter stage**

Main file:
- `adapter_v2/train.py`

### Stage B
**plasma-guided generation stage**

Main file:
- `plasma_train.py`

### Legacy baseline
**legacy method** / **baseline**

Main file:
- `train.py`

If any summary, script note, or experiment description contradicts this, the latest branch code should be considered the source of truth.
