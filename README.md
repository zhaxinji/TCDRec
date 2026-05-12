# **Tri-channel Conditional Diffusion with Cross-modal Causal Alignment for Multimodal Recommendation**

## 📚 Overview of TCDRec

Most existing multimodal recommenders rely on **associative cross-modal fusion**, which entangles genuine modality effects with spurious correlations induced by **latent confounders** (e.g., a brand-driven aesthetic style or a category-level design convention) that simultaneously govern visual and textual feature generation while independently shaping user preferences. This opens a back-door path $V \leftarrow C \rightarrow T$ together with $C \rightarrow Y$, distorting preference estimation.

We propose **TCDRec**, a causal multimodal recommendation framework that disentangles the latent confounder along its **entire generative pathway** through two coordinated interventions:

- **Causal confounder identification (post-fusion).** Building on a Structural Causal Model (SCM) of multimodal preference generation, TCDRec exploits the conditional independence $V \perp T \mid C$ implied by the causal structure. Two complementary proxies of the confounder $C$ are produced by **bidirectional cross-attention** ($t\tov$ and $v\tot$) and aligned through a single InfoNCE-style **causal contrastive loss**, retaining the cross-modally invariant signal at no extra parameter cost.
- **Tri-channel conditional diffusion (generative source).** Three parallel diffusion channels with $x_0$-prediction and a gated-MLP denoiser refine the **ID**, **visual**, and **textual** embeddings under channel-specific priors: the ID channel is conditioned on a **learnable retrieval prior** aggregated from semantic neighbors; the visual and textual channels are conditioned on **stop-gradient ID priors** that prevent modality features from collapsing the ID embedding into a low-rank shadow.
- **Strengthened homogeneous graphs.** For both users and items, a **co-occurrence graph** (collaborative co-pattern) is fused with a **multimodal kNN graph** (visual + textual similarity), so that hierarchical propagation captures behavioral affinity and semantic similarity simultaneously, mitigating the popularity path through the interaction graph $G$.

Together, the post-fusion contrastive identification and the upstream conditional generative refinement disentangle the latent confounder along its causal pathway, recovering preference estimates faithful to the direct visual and textual effects.

## 📝 Environment Requirement

The code has been tested under Python 3.9. The required packages are as follows:

- pytorch >= 1.13.0
- numpy >= 1.24.4
- scipy >= 1.10.1
- tqdm
- tensorboardX
- matplotlib

All experiments are conducted on a single NVIDIA RTX 4090D GPU (24 GB).

## 📥 Data

Full data can be downloaded from HuggingFace:

- [Baby](https://huggingface.co/datasets/MrShouxingMa/Baby/tree/main)
- [Sports](https://huggingface.co/datasets/MrShouxingMa/Sports/tree/main)
- [Clothing](https://huggingface.co/datasets/MrShouxingMa/Clothing/tree/main)

We follow the same 5-core preprocessing as LATTICE / FREEDOM / LGMRec on three Amazon Review categories.


| #Dataset | #Users | #Items | #Interactions | Modality | Sparsity |
| -------- | ------ | ------ | ------------- | -------- | -------- |
| Baby     | 19,445 | 7,050  | 160,792       | V, T     | 99.88%   |
| Sports   | 35,598 | 18,357 | 296,337       | V, T     | 99.95%   |
| Clothing | 39,387 | 23,033 | 278,677       | V, T     | 99.97%   |


## 🚀 Example to Run the Codes

- e.g., Baby dataset

```
python main.py --dataset baby
```

## 📂 Folder Structure

The released code consists of the following files.

```
--data
    --baby
    --clothing
    --sports
--dataloader.py        
--main.py              
--model.py            
--params.py           
--training.py          
--utils.py            
```

