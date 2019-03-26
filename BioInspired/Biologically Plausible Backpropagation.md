# Biologically Plausible Backpropagation  

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

**[Towards Biologically Plausible Deep Learning]()**
* Weights change if there is a pre-synaptic spike in the temporal vecinity of a post-synaptic spike

\Delta W_{ij} \prop S_i \dot{V_j}

* **If the change ∆Vj corresponds to improving some objective function J, then STDP corresponds to approximate stochastic gradient descent in that objective function**

* J comes out of a variational bound on the likelihood of the data (observed variable x; latent variable h)

\text{Joint: }p(x,h); \text{Inference mechanism: }q^\*(H|x)

\log p(x) = \log p(x) \sum_h q^\*(h|x)
= \log p(x) \sum_h q^\*(h|x) \log \frac{p(x,h) q^\*(h|x)}{p(h|x)q^\*(h|x)}
= E_{q^\*(H|x)} [\log p(x,H)] + H[q^\*(H|x)] + KL(q^\*(H|x) || p(H|x))
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
