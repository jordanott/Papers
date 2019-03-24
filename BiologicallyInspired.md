# Biologically Inspired Neural Networks

## Biologically Plausible Backpropagation  
**[Theories of Error Back-Propagation in the Brain](https://www.sciencedirect.com/science/article/pii/S1364661319300129)**  
Learning rules in biologically plausible models can be implemented with different types of spike-time-dependent plasticity and *approximate* backprop
* **Temporal error models:** 
  * Contrastive learning (Box 2): Approximates backprop but requires control signal of when target is present
    * 2 stages; 1. unlearn (anti-Hebbian) existing association between input and prediction 2. learn new association between input and target  
    ![](https://latex.codecogs.com/gif.latex?%5Ccenter%20%5CDelta%20W%20%5Csim%20%28t%20-%20x_L%29x%5ET_%7BL-1%7D%20%3D%20-%20x_L%20x%5ET_%7BL-1%7D%20&plus;%20t%20x%5ET_%7BL-1%7D%20%5Cnewline%20%5Ccenter%20%5CDelta%20W%20%5Csim%20%5Cdelta_L%20x%5ET_%7BL-1%7D)
* **Explicit error models:** 
  * Predictive coding (Box 3): 
    * Requires one-to-one connectivity between error and value nodes  
    ![](https://latex.codecogs.com/gif.latex?%5Ctext%7BError%20Node%3A%20%7D%20%5Cdelta_l%20%3D%20x_l%20-%20W_%7Bl-1%7D%20x_%7Bl-1%7D%20%5Cnewline%20%5Ctext%7BValue%20Node%3A%20%7D%5Cdot%7Bx%7D%20%3D%20-%20%5Cdelta_l%20&plus;%20W_l%5ET%20%5Cdelta_%7Bl&plus;1%7D)
  * Dendritic error model (Box 4): Extension from predictive coding; "Errors computed in apical dendrites"
    * Decay, feed forward, feedback, intra-layer  
    <a href="https://www.codecogs.com/eqnedit.php?latex=\dot{x}&space;=&space;-x_l&space;&plus;&space;W_{l-1}x_{l-1}&space;&plus;&space;W_l^T&space;x_{l&plus;1}&space;-&space;W_l^T&space;W_l&space;x_l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dot{x}&space;=&space;-x_l&space;&plus;&space;W_{l-1}x_{l-1}&space;&plus;&space;W_l^T&space;x_{l&plus;1}&space;-&space;W_l^T&space;W_l&space;x_l" title="\dot{x} = -x_l + W_{l-1}x_{l-1} + W_l^T x_{l+1} - W_l^T W_l x_l" /></a>
* Can be described in terms of mimizing energy (Hopfield energy &rarr; contrastive learning & free energy &rarr; Explicit models)
    
**[Random synaptic feedback weights support error backpropagation for deep learning](https://www.nature.com/articles/ncomms13276)**  
Solves weight transport problem
* Replace transpose of weight matrix, *W<sup>T</sup>* with random weights *B*

**[Backpropagation through time and the brain](https://reader.elsevier.com/reader/sd/pii/S0959438818302009?token=5BD5FEC9246457E7B82A1D05CF9552D0A225AB8D2019AAA0B40DCC24F5578198F246669B22A804E1B99C966BAA4EF6C6)**  
Review of BPTT; general overview of attention and memory mechanisms used for Temporal credit assignment (TCA)
* Progress in solving difficult TCA has been made by memory- and attention-based architectures and algorithms
* Strengthens BPTT’s position as a guide for thinking about TCA in artificial and biological systems


## Spiking Neural Networks
**[Deep Continuous Local Learning]()**

## Learning in the Machine
**[Learning in the Machine: To Share or not to Share]()**

## Unsupervised
**[Unsupervised Learning by Competing Hidden Units](https://arxiv.org/pdf/1806.10181.pdf)**  
* Can good/useful early layer representations be learned without supervision; using only a local “biological” synaptic plasticity rule?
* Local learning rule, that incorporate both LTP and LTD types of plasticity and global inhibition in the hidden layer

## Reinforcement Learning
**[Reinforcement learning in artificial and biological systems](https://www.nature.com/articles/s42256-019-0025-4#ref-CR69)**
* Connections from cortex to striatum
  * Cortex represents available choices, strength of cortical connections on striatal cells represents value of choices
  * Dopamine encodes Reward Prediction Error (RPE)
  * Change in dopamine concentration drives synaptic plasticity on frontal striatal synapses
* Learning on different timescales
  * Plasticity in amygdala operates on fast timescales; activity dependent
    * Susceptible to noise because of fast changes
  * Plasticity in striatum operates on slower timescales; dopamine dependent
