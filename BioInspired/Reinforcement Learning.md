# Reinforcement Learning
**[Reinforcement learning in artificial and biological systems](https://www.nature.com/articles/s42256-019-0025-4#ref-CR69)**
* Connections from cortex to striatum
  * Cortex represents available choices, strength of cortical connections on striatal cells represents value of choices
  * Dopamine encodes Reward Prediction Error (RPE)
  * Change in dopamine concentration drives synaptic plasticity on frontal striatal synapses
* Learning on different timescales
  * Plasticity in amygdala operates on fast timescales; activity dependent
    * Suscep## Learning in the Machine
**[Learning in the Machine: To Share or not to Share]()**
tible to noise because of fast changes
  * Plasticity in striatum operates on slower timescales; dopamine dependent

**[Prefrontal cortex as a meta-reinforcement learning system]()**
* **System architecture:** PFC, together with the basal ganglia and thalamic nuclei with which it connects, as forming a recurrent neural network. Inputs: perceptual data, which either contains or is accompanied by information about executed actions and received rewards. On the output side, the network triggers actions and also emits estimates of state value

* **Learning:** synaptic weights in the prefrontal network, including its striatal components, are adjusted by a model-free RL procedure, in which DA conveys a RPE signal. Via this role, the DA-based RL procedure shapes the activation dynamics of the recurrent prefrontal network

* **Task environment:** RL takes place not on a single task, but instead in a dynamic environment posing a series of interrelated tasks. The learning system is thus
required to engage in ongoing inference and behavioral adjustment
