# Spiking Neural Networks
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

# Unsupervised
**[Unsupervised Learning by Competing Hidden Units](https://arxiv.org/pdf/1806.10181.pdf)**  
* Can good/useful early layer representations be learned without supervision; using only a local “biological” synaptic plasticity rule?
* Local learning rule, that incorporate both LTP and LTD types of plasticity and global inhibition in the hidden layer
* Train with unsupervised, local rule; then use coding produced to train another layer using SGD
