# Mixture-of-Agents

![Mixture of Agents Workflow](assets/workflow.jpg)

This is a practical implementation of the paper titled *"Mixture-of-Agents Enhances Large Language Model Capabilities"*. [Read the paper](https://arxiv.org/abs/2406.04692)

### Role of Each Layer in the Architecture:

- **Layer 1:** Initial brainstorming by reference models.
- **Layer 2+:** Refined reasoning by the same models on the user's prompt but this time using prior layer's ideas.
- **Last Layer:** Final synthesis performed by a dedicated aggregator model.
