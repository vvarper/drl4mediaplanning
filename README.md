# Deep Reinforcement Learning for Media Planning

Source code associated to the paper "Deep reinforcmenet learning in agent-based simultions for
optimal media planning", co-authored by Víctor A. Vargas-Pérez, Pablo Mesejo, Manuel Chica and Oscar
Cordón.

This repository comprises only the top layer of the "DRL for media planning" framework.
Specifically, this is the DRL part, for which we use the RL4J module of the suite Deeplearning4j
(DL4J, https://deeplearning4j.konduit.ai/).

## WARNING

This source code is currently not runnable because the MDP environment is incomplete: its operation
requires interacting with the marketing agent-based model (ABM).
Unfortunately, we cannot share neither the code of that ABM nor the data used in the
experiments of the paper due to confidentiality clauses with third parties.

However, we plan to update this repository in the future and completely decouple it from the
marketing ABM, replacing the latter with a toy model. In this way, we will be able to
provide a runnable toy example of this framework.