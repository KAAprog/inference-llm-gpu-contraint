# Inférence LLM sur GPU contraint

**Mémoire de Master II Recherche**  
**Option :** Informatique Fondamentale  
**Candidat :** KANA AZONGMO ADELINE (Matricule : CM-UDS-21SCI0551)  
**Direction :** Dr. YANKAM Yannick Florian, Chargé de Cours  
**Établissement :** Université de Dschang — École Doctorale DSSTS — Unité de Recherche en Informatique Fondamentale, Ingénierie et Applications (URFIA)  

---

## 📌 Contexte et Problématique

L'essor des grands modèles de langage (LLM) basés sur l'architecture Transformer (ex: *Meta-Llama-3-8B* / *Llama-3.1-8B*) se heurte au **mur de la mémoire** lors du déploiement local sur des accélérateurs grand public (VRAM entre 8 Go et 16 Go) :
1. **Saturation statique des poids :** Le modèle en FP16 exige ~15,5 Go de VRAM, provoquant un crash immédiat (OOM) sur GPU de 8 Go et saturant la marge sur 16 Go.
2. **Saturation dynamique du KV-cache :** L'empreinte mémoire croît linéairement avec la longueur de séquence ($L_{ctx}$), saturant la bande passante et la capacité résiduelle.

Pour relâcher ce double goulot, le projet étudie l'interaction physique et logicielle entre deux familles de leviers :
- **Compression arithmétique (Quantization) :** Réduction de la précision des poids ($W$) et/ou du KV-cache ($KV$) en INT8, INT4, INT2 (via BitsAndBytes, HQQ, Quanto).
- **Attention I/O-aware :** Algorithme FlashAttention-2 (tiling en SRAM rapide) versus mode Math naïf ($\mathcal{O}(N^2)$).

---

## 🔬 Hypothèses de Recherche

- **H1 (Effet non monotone de la quantization agressive) :** L'INT4 réduit la VRAM mais son impact sur la latence n'est pas monotone en raison du surcoût de déquantisation.
- **H2 (Réduction des gains de FlashAttention) :** Les gains de FlashAttention sont réduits ou annulés sous quantization agressive en raison de l'overhead arithmétique dans le chemin critique.
- **H3 (Interaction non additive) :** L'effet conjoint de la quantification et de FlashAttention n'est pas additif (super-additif sur l'admissibilité VRAM, sous-additif sur le débit de décodage).
- **H4 (Généralisation de l'heuristique) :** Une heuristique légère entraînée sur une grille restreinte prédit l'optimum sur des points non vus.

---

## ⚙️ Architecture du Pipeline V2 & Orchestration

Le dépôt contient l'orchestrateur de campagne V2 (`run_campaign.py`) qui supervise l'exécution rigoureuse de 100 configurations factorielles :
- **Phase 1 (Poids) :** $W$ variable ($W_{int4}, W_{int8}, W_{fp16}$), cache KV fixe FP16 natif.
- **Phase 2 (Cache KV) :** Poids fixes FP16, cache KV variable ($KV_{int8}, KV_{int4}, KV_{int2}$ via HQQ/Quanto).
- **Phase 3 (Conjointe) :** $W$ et $KV$ quantifiés conjointement.

### Fonctionnalités de l'orchestrateur :
- Isolation inter-processus complète (purge 100% VRAM entre chaque test).
- Reprise automatique sur incident (analyse des CSV de sortie).
- Surveillance active du GPU (`--gpu-wait` via `nvidia-smi`) et respect des pauses thermiques.

---

## 🚀 Utilisation

```bash
# Lancer la campagne complète (dry-run pour prévisualiser la grille)
python run_campaign.py --phase all --dry-run

# Lancer la campagne avec attente de libération du GPU (60 min max)
python run_campaign.py --phase all --gpu-wait 60
```
