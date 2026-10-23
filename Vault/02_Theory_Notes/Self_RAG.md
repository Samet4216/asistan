---
title: 'Self-RAG: Self-Reflective Retrieval-Augmented Generation'
category: Theory_Notes
status: approved
source: lab_curator
tags:
- theory-note
- deep-learning
- rag
- self-reflection
- critique
- retrieval
---

# Self-RAG: Self-Reflective Retrieval-Augmented Generation

Standard RAG indiscriminately retrieves a fixed number of documents regardless of whether external knowledge is needed. Self-RAG introduces adaptive retrieval and self-critique via special reflection tokens.

## 📐 Reflection Token Taxonomy
1. `[Retrieve]`: Predicts whether external retrieval is needed ($\text{yes}, \text{no}, \text{continue}$). 
2. `[IsRel]`: Evaluates whether the retrieved passage provides relevant knowledge.
3. `[IsSup]`: Checks whether the generated sentence is fully supported by the passage (factual attribution).
4. `[IsUse]`: Rates the overall utility and helpfulness of the response.

## 🔗 Interconnected Nodes
- Paper Card: [[2310.11511]] (Self-RAG)
- Preference Optimization: [[Direct_Preference_Optimization]]
- Knowledge Retrieval: [[Self_Attention]]

