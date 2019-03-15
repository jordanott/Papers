# Biologically Inspired Neural Networks



## Biologically Plausible Backpropagation  
[Theories of Error Back-Propagation in the Brain](https://www.sciencedirect.com/science/article/pii/S1364661319300129)  
Overview of models proposing biological backprop solutions  
* Locality of error signal, weight transport, unrealistic models of neurons
* Learning rules in biologically plausible models can be implemented with different types of spike-time-dependent plasticity
* Dynamics and plasticity of the models can be described within a common framework of energy minimization

[Random synaptic feedback weights support error backpropagation for deep learning](https://www.nature.com/articles/ncomms13276)  Solves weight transport problem
* Replace transpose of weight matrix, *W<sup>T</sup>* with random weights *B*

[Backpropagation through time and the brain](https://reader.elsevier.com/reader/sd/pii/S0959438818302009?token=5BD5FEC9246457E7B82A1D05CF9552D0A225AB8D2019AAA0B40DCC24F5578198F246669B22A804E1B99C966BAA4EF6C6)  
Review of BPTT; general overview of attention and memory mechanisms used for Temporal credit assignment (TCA)
* Progress in solving difficult TCA has been made by memory- and attention-based architectures and algorithms
* Strengthens BPTT’s position as a guide for thinking about TCA in artificial and biological systems


## Spiking Neural Networks
[Deep Continuous Local Learning]()

## Learning in the Machine
[Weight Sharing]()

## Reinforcement Learning
[Reinforcement learning in artificial and biological systems](https://www.nature.com/articles/s42256-019-0025-4#ref-CR69)
* Connections from cortex to striatum
  * Cortex represents available choices, strength of cortical connections on striatal cells represents value of choices
  * Dopamine encodes Reward Prediction Error (RPE)
  * Change in dopamine concentration drives synaptic plasticity on frontal striatal synapses
* Learning on different timescales
  * Plasticity in amygdala operates on fast timescales; activity dependent
    * Susceptible to noise because of fast changes
  * Plasticity in striatum operates on slower timescales; dopamine dependent
