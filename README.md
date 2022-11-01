# Zero-Label Prompt Selection

This repository contains the official code for ZPS.

In our paper, we provide ZPS, an algorithm for zero-label prompt selection. ZPS choose an optimal prompt from the set of manual prompt candidates without and labeled data or parameter update.

You should be able to reproduce these main results with our code.
- Zero-label
    ![](./ZL.png)
- Few-shot
    ![](./FS.png)

## Contents
We provide code for both zero-label setting and few-shot setting. Including
- Zero-label
    - ZPS
    - Self-training
- Few-shot
    - ICL
    - GPS
    - GRIPS
    - Model-tuning
    - Prompt-tuning


The details to reproduce the main results can be found in [SRC](src/README.md).

