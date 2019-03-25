# Biologically Inspired Neural Networks

## Questions
* Poisson spike train

## Notes
* Want to learn to predict active neurons
* Minimize energy consumption of neurons (evolutionarily advantageous)
  * Cite work showing decrease in glucose(?) usage in areas of brain after learning

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

**[Towards Biologically Plausible Deep Learning](https://arxiv.org/pdf/1502.04156.pdf?utm_content=buffer22202&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer)**
* Weights change if there is a pre-synaptic spike in the temporal vecinity of a post-synaptic spike
<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;W_{ij}&space;\propto&space;S_i&space;\dot{V_j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;W_{ij}&space;\propto&space;S_i&space;\dot{V_j}" title="\Delta W_{ij} \propto S_i \dot{V_j}" /></a>

* **If the change ∆Vj corresponds to improving some objective function J, then STDP corresponds to approximate stochastic gradient descent in that objective function**

* J comes out of a variational bound on the likelihood of the data (observed variable x; latent variable h)

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Joint:&space;}p(x,h);&space;\text{Inference&space;mechanism:&space;}q^*(H|x)&space;\\&space;\log&space;p(x)&space;=&space;\log&space;p(x)&space;\sum_h&space;q^*(h|x)&space;\\&space;=&space;\log&space;p(x)&space;\sum_h&space;q^*(h|x)&space;\log&space;\frac{p(x,h)&space;q^*(h|x)}{p(h|x)q^*(h|x)}&space;\\&space;=&space;E_{q^*(H|x)}&space;[\log&space;p(x,H)]&space;&plus;&space;H[q^*(H|x)]&space;&plus;&space;KL(q^*(H|x)&space;||&space;p(H|x))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Joint:&space;}p(x,h);&space;\text{Inference&space;mechanism:&space;}q^*(H|x)&space;\\&space;\log&space;p(x)&space;=&space;\log&space;p(x)&space;\sum_h&space;q^*(h|x)&space;\\&space;=&space;\log&space;p(x)&space;\sum_h&space;q^*(h|x)&space;\log&space;\frac{p(x,h)&space;q^*(h|x)}{p(h|x)q^*(h|x)}&space;\\&space;=&space;E_{q^*(H|x)}&space;[\log&space;p(x,H)]&space;&plus;&space;H[q^*(H|x)]&space;&plus;&space;KL(q^*(H|x)&space;||&space;p(H|x))" title="\text{Joint: }p(x,h); \text{Inference mechanism: }q^*(H|x) \\ \log p(x) = \log p(x) \sum_h q^*(h|x) \\ = \log p(x) \sum_h q^*(h|x) \log \frac{p(x,h) q^*(h|x)}{p(h|x)q^*(h|x)} \\ = E_{q^*(H|x)} [\log p(x,H)] + H[q^*(H|x)] + KL(q^*(H|x) || p(H|x))" /></a>
* Bound log-likelihood with cross-entropy term. Suggests q<sup>\*</sup> should approximate p(H|x)


1. Back-propagation computation is purely linear, whereas biological neurons interleave linear and non-linear operations
2. If the feedback paths were used to propagate credit assignment by backprop, they would need precise knowledge of the derivatives of the non-linearities at the operating point used in the corresponding feedforward computation on the feedforward
path
3. Feedback paths would have to use exact symmetric weights (with the same connectivity,
transposed) of the feedforward connections (weight transport)
4. Real neurons communicate by (possibly stochastic) binary values
(spikes)
5. Computation would have to be precisely clocked to alternate between feedforward and back-propagation phases (backprop requires forward results)
6. It is not clear where the output targets would come from

Future work:
* Lateral connections

## Spiking Neural Networks
**[Synaptic Plasticity Dynamics for Deep Continuous Local Learning](https://arxiv.org/pdf/1811.10766.pdf)**  
<a href="https://www.codecogs.com/eqnedit.php?latex=\centering&space;u_i&space;=&space;\sum_j&space;w_{ij}&space;(\epsilon&space;*&space;s_j)&space;&plus;&space;\eta&space;*&space;s_i;&space;s_i&space;=&space;\theta&space;(u_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\centering&space;u_i&space;=&space;\sum_j&space;w_{ij}&space;(\epsilon&space;*&space;s_j)&space;&plus;&space;\eta&space;*&space;s_i;&space;s_i&space;=&space;\theta&space;(u_i)" title="\centering u_i = \sum_j w_{ij} (\epsilon * s_j) + \eta * s_i; s_i = \theta (u_i)" /></a>

* Conditional probability of output spike (s<sub>i</sub>=1); given input spike vector **s**  
<a href="https://www.codecogs.com/eqnedit.php?latex=P(s_i&space;=&space;1|&space;s)&space;=&space;\sigma(u_i(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(s_i&space;=&space;1|&space;s)&space;=&space;\sigma(u_i(t))" title="P(s_i = 1| s) = \sigma(u_i(t))" /></a>

* Gradient-based optimization of a target loss L as a function of a<sub>i</sub>  
<a href="https://www.codecogs.com/eqnedit.php?latex=a_i&space;=&space;\sigma(u_i);&space;\frac{\partial&space;L}{\partial&space;w_{ij}}&space;=&space;\frac{\partial&space;L}{\partial&space;a_{i}}&space;\frac{\partial&space;a_{i}}{\partial&space;w_{ij}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_i&space;=&space;\sigma(u_i);&space;\frac{\partial&space;L}{\partial&space;w_{ij}}&space;=&space;\frac{\partial&space;L}{\partial&space;a_{i}}&space;\frac{\partial&space;a_{i}}{\partial&space;w_{ij}}" title="a_i = \sigma(u_i); \frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_{i}} \frac{\partial a_{i}}{\partial w_{ij}}" /></a>

* Three factor rules  
<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Modulatory:&space;}&space;\frac{\partial&space;L}{\partial&space;u_i};&space;\text{post-synaptic:&space;}&space;\sigma^{'}(u_i);&space;\text{pre-synaptic:&space;}(\epsilon&space;*&space;s_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Modulatory:&space;}&space;\frac{\partial&space;L}{\partial&space;u_i};&space;\text{post-synaptic:&space;}&space;\sigma^{'}(u_i);&space;\text{pre-synaptic:&space;}(\epsilon&space;*&space;s_j)" title="\text{Modulatory: } \frac{\partial L}{\partial u_i}; \text{post-synaptic: } \sigma^{'}(u_i); \text{pre-synaptic: }(\epsilon * s_j)" /></a>

* Learn local errors from random classifiers at each layer;  no error information propagates downwards through the layer stack, the layers indirectly learn useful hierarchical features that end up minimizing the cost at the top layer

* Layerwise loss  
<a href="https://www.codecogs.com/eqnedit.php?latex=\centering&space;\text{fixed&space;random&space;matricies:&space;}&space;g_{ij}^n;&space;\text{pseudotarget&space;for&space;layer&space;n:&space;}&space;\hat{y}_i^n&space;\\&space;L^n&space;=&space;\sum_i&space;\int_{-\inf}^{T}df(\sum_j&space;g_{ij}^n&space;a_j^n&space;-&space;\hat{y}_i^n)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\centering&space;\text{fixed&space;random&space;matricies:&space;}&space;g_{ij}^n;&space;\text{pseudotarget&space;for&space;layer&space;n:&space;}&space;\hat{y}_i^n&space;\\&space;L^n&space;=&space;\sum_i&space;\int_{-\inf}^{T}df(\sum_j&space;g_{ij}^n&space;a_j^n&space;-&space;\hat{y}_i^n)^2" title="\centering \text{fixed random matricies: } g_{ij}^n; \text{pseudotarget for layer n: } \hat{y}_i^n \\ L^n = \sum_i \int_{-\inf}^{T}df(\sum_j g_{ij}^n a_j^n - \hat{y}_i^n)^2" /></a>

**[SuperSpike]()**

## Learning in the Machine
**[Learning in the Machine: To Share or not to Share]()**

## Unsupervised
**[Unsupervised Learning by Competing Hidden Units](https://arxiv.org/pdf/1806.10181.pdf)**  
* Can good/useful early layer representations be learned without supervision; using only a local “biological” synaptic plasticity rule?
* Local learning rule, that incorporate both LTP and LTD types of plasticity and global inhibition in the hidden layer
* Train with unsupervised, local rule; then use coding produced to train another layer using SGD

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

**[Prefrontal cortex as a meta-reinforcement learning system]()**
* **System architecture:** PFC, together with the basal ganglia and thalamic nuclei with which it connects, as forming a recurrent neural network. Inputs: perceptual data, which either contains or is accompanied by information about executed actions and received rewards. On the output side, the network triggers actions and also emits estimates of state value

* **Learning:** synaptic weights in the prefrontal network, including its striatal components, are adjusted by a model-free RL procedure, in which DA conveys a RPE signal. Via this role, the DA-based RL procedure shapes the activation dynamics of the recurrent prefrontal network

* **Task environment:** RL takes place not on a single task, but instead in a dynamic environment posing a series of interrelated tasks. The learning system is thus
required to engage in ongoing inference and behavioral adjustment
