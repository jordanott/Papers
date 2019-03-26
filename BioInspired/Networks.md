# Spiking Neural Networks
**[Deep Continuous Local Learning]()**
u_i = \sum_j w_{ij} (\epsilon * s_j) + \eta * s_i
s_i = \theta (u_i)

* Conditional probability of output spike (s<sub>i</sub>=1); given input spike vector **s**
P(s_i = 1| s) = \sigma(u_i(t))

* Gradient-based optimization of a target loss L as a function of a<sub>i</sub>

a_i = \sigma(u_i); \frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_{i}} \frac{\partial a_{i}}{\partial w_{ij}}

* Three factor rules
\text{Modulatory: } \frac{\partial L}{\partial u_i} \text{post-synaptic: } \sigma^'(u_i) \text{pre-synaptic: }(\epsilon * s_j)

* Learn local errors from random classifiers at each layer;  no error information propagates downwards through the layer stack, the layers indirectly learn useful hierarchical features that end up minimizing the cost at the top layer

* Layerwise loss
\text{fixed random matricies: } g_{ij}^n; \text{pseudotarget for layer n: } \hat{y}_i^n;  L^n = \sum_i \int_{-\inf}^{T}df(\sum_j g_{ij}^n a_j^n - \hat{y}_i^n)^2

**[SuperSpike]()**

# Unsupervised
**[Unsupervised Learning by Competing Hidden Units](https://arxiv.org/pdf/1806.10181.pdf)**  
* Can good/useful early layer representations be learned without supervision; using only a local “biological” synaptic plasticity rule?
* Local learning rule, that incorporate both LTP and LTD types of plasticity and global inhibition in the hidden layer
* Train with unsupervised, local rule; then use coding produced to train another layer using SGD
