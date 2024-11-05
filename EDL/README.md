# Usage of the code #
All modules are presented in the EDL_modules folder. The main modules are:
- EDL_callbacks.py: contains the callbacks for updating the epoch in the loss while training.
- EDL_models.py: contains the models for the Evidential Networks. The model has a linear output layer.
- EDL_losses.py: contains the losses for the Evidential Networks. The output of the model is mapped to evidence through a ReLU and then to a Dirichlet inside the loss.
- EDL_metrics.py: contains the metrics for the Evidential Networks these are the accuracy, the mean evidence, the mean evidence of success and the mean evidence of fails. The output of the model is modified inside each metric depending on the metric's needs.
- EDL_utils.py: contains the auxiliary functions for importing datasets and plotting results.

To run the EDL framework with mse loss it is only necessary to turn the TEST_EQ5=True in the EDL_script.py file
and run it.

# Evidential Networks #

In this approach we assume the is **no OOD data available in training**. This Dirichlet-based model for neural networks is based on 
**subjective logic** framework. This framework formalizes the Dempster-Shafer theory's notion of belief assignments over a frame of
discernement as a Dirichlet distribution.

> The Dempster-Shafer theory of evidence [(P. Dempster, 2008)](https://www.stat.purdue.edu/~chuanhai/projects/DS/docs/68RSS.pdf) is a generalization of the Bayesian theory to
> subjective probabilities. It assigns belief masses to subsets of a frame of discernment, which denotes the set of exclusive possible states, e.g., possible class labels for
> a sample.

It boils down to account for an overall uncertainty mass of $u$ added to belief classes $b_c$: $$u+\sum_{c=1}^Cb_c=1 \label{eq1.2}\tag{1}$$
where $u\geq 0$ and $\forall c\in y, b_c\geq 0$. A belief mass $b_c$ for a singleton $c$ is computed using the evidence for the singleton.
Let $e_c\geq 0$ be the evidence derived for the $c^{th}$ singleton, then the belief $b_c$ and the uncertainty $u$ are computed as:
$$b_c=\frac{e_c}{S} \quad \text{and} \quad u=\frac{C}{S} \label{eq2.2}\tag{2}$$
where $S=\sum_{c=1}^C(e_c+1)$. Note that the uncertainty is inversely proportional to the total evidence.

The link to Dirichlet comes from the fact that the concentration parameters of the distribution $p(\mu|x,\theta)$ correspond to evidences
$\alpha_c=e_c+1$. Then, $S=\alpha_0$, which represents the spread of the distribution.

[Sensoy et al., 2018](https://arxiv.org/abs/1806.01768) propose to model the concentration parameters by a neural network output hence:
$$\alpha = \text{ReLU}\big(f(x,\theta)\big) + 1$$
which differs from the modelization of Prior Networks by replacing the softmax layer with a ReLU activation layer to ensure non-negative outputs.

## Training an Evidential Neural Network ##
Authors propose the minimization of the Bayes risk of the MSE loss with respect to the "class predictor":
$$\mathcal{L} _{ENN}(\theta) = \mathbb{E} _{p(x,y)}\Big[\int ||y-\mu||^2\cdot p(\mu|x,\theta)d\mu + \lambda_t\cdot KL\big(\text{Dir}(\mu|\tilde{\alpha})||\text{Dir}(\mu|u)\big) \Big] \label{eq3}\tag{3}$$

The extra KL-Divergence serves as a regularizer which penalizes "unknown" predictive distributions. $\text{Dir}(\mu|u)$ is the Dirichlet uniform distribution and $\tilde{\alpha}=y+(1-y)\odot\alpha=1 + (1-y)\odot e$ are the Dirichlet.
parameters after removal of the non-misleading evidence from predicted parameters $\alpha$. 
The annealing coefficient $\lambda_t=\text{min}(1,t/10)\in[0,1]$ depends on the index of the current epoch, $t$.

An interesting property of the Bayes risk minimization is that, thanks to the variance identity, it can be reduced to:
$$\int ||y-\mu||^2\cdot p(\mu|x,\theta)d\mu = \sum_{c=1}^C(y_c-\frac{\alpha_c}{\alpha_0})^2+\frac{\alpha_c(\alpha_0-\alpha_c)}{\alpha_0^2(\alpha_c+1)} \label{eq4}\tag{4}$$
This representation shows the joint goals of minimizing the prediction error and the variance of the Dirichlet distribution output by the neural network.

