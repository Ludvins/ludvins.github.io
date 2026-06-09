---
layout: base
title: Publications
nav_key: publications
permalink: /publications/
description: Publication archive for Luis A. Ortega.
---

<main class="page-shell publications-page">
  <section class="page-hero">
    <p class="section-kicker">Publication Archive</p>
    <h1>Published work and active research threads.</h1>
    <p>
      Browse peer-reviewed papers, workshop papers, current preprints, and
      code-linked research projects by year or topic.
    </p>
  </section>

  <section class="publication-tools" aria-label="Publication filters">
    <label class="screen-reader-text" for="publication-search">Search publications</label>
    <input id="publication-search" type="search" placeholder="Search title, venue, method, or repository" autocomplete="off" />
    <div class="filter-chips" aria-label="Theme filters">
      <button type="button" class="chip is-active" data-filter="all">All</button>
      <button type="button" class="chip" data-filter="bayesian-dl">Bayesian DL</button>
      <button type="button" class="chip" data-filter="inference">Inference</button>
      <button type="button" class="chip" data-filter="bounds">Bounds</button>
      <button type="button" class="chip" data-filter="code">Code available</button>
    </div>
  </section>

  <section class="archive-layout">
    <aside class="archive-sidebar" aria-label="Archive navigation">
      <h2>Archive</h2>
      <div class="year-list">
        <button type="button" class="year-button is-active" data-year="all"><span>All</span><span>13</span></button>
        <button type="button" class="year-button" data-year="2026"><span>2026</span><span>3</span></button>
        <button type="button" class="year-button" data-year="2025"><span>2025</span><span>1</span></button>
        <button type="button" class="year-button" data-year="2024"><span>2024</span><span>3</span></button>
        <button type="button" class="year-button" data-year="2023"><span>2023</span><span>1</span></button>
        <button type="button" class="year-button" data-year="2022"><span>2022</span><span>2</span></button>
        <button type="button" class="year-button" data-year="ongoing"><span>Ongoing</span><span>3</span></button>
      </div>
      <div class="archive-summary">
        <h3>At a glance</h3>
        <span>10 published papers</span>
        <span>3 ongoing projects</span>
        <span>8 code-linked works</span>
      </div>
    </aside>

    <div class="publication-list" id="publication-list">
      <article class="publication-card accent-oxblood" id="scalable-linearized-laplace" data-year="2026" data-tags="bayesian-dl inference code uncertainty ntk lla" data-title="scalable linearized laplace approximation surrogate neural kernel esann spotlight bayesipy">
        <div class="pub-year"><span>2026</span></div>
        <div class="pub-content">
          <h2>Scalable Linearized Laplace Approximation via Surrogate Neural Kernel</h2>
          <p class="authors">Luis A. Ortega, Simon Rodriguez-Santana, Daniel Hernandez-Lobato</p>
          <p class="venue">ESANN 2026 &middot; Spotlight talk</p>
          <p>Learns a surrogate neural kernel to avoid large Jacobians while estimating uncertainty for pre-trained networks.</p>
          <div class="tag-row compact">
            <span>LLA</span>
            <span>NTK</span>
            <span>Uncertainty</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              We introduce a scalable method to approximate the kernel of the Linearized Laplace Approximation (LLA). For this, we use a surrogate deep neural network (DNN) that learns a compact feature representation whose inner product replicates the Neural Tangent Kernel (NTK). This avoids the need to compute large Jacobians. Training relies solely on efficient Jacobian-vector products, allowing to compute predictive uncertainty on large-scale pre-trained DNNs. Experimental results show similar or improved uncertainty estimation and calibration compared to existing LLA approximations. Notwithstanding, biasing the learned kernel significantly enhances out-of-distribution detection. This remarks the benefits of the proposed method for finding better kernels than the NTK in the context of LLA to compute prediction uncertainty given a pre-trained DNN.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://github.com/Ludvins/BayesiPy">Code</a>
        </div>
      </article>

      <article class="publication-card no-actions accent-oxblood" id="quadratic-linearized-laplace" data-year="2026" data-tags="bayesian-dl inference laplace regression" data-title="improving linearized laplace approximation quadratic approximations esann qla">
        <div class="pub-year"><span>2026</span></div>
        <div class="pub-content">
          <h2>Improving the Linearized Laplace Approximation via Quadratic Approximations</h2>
          <p class="authors">Pedro Jimenez, Luis A. Ortega, Pablo Morales-Alvarez, Daniel Hernandez-Lobato</p>
          <p class="venue">ESANN 2026</p>
          <p>Rank-one quadratic factors improve fidelity to the full Laplace posterior while keeping prediction linearized.</p>
          <div class="tag-row compact">
            <span>QLA</span>
            <span>Laplace</span>
            <span>Regression</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              Deep neural networks (DNNs) often produce overconfident out-of-distribution predictions, motivating Bayesian uncertainty quantification. The Linearized Laplace Approximation (LLA) achieves this by linearizing the DNN and applying Laplace inference to the resulting model. Importantly, the linear model is also used for prediction. We argue this linearization in the posterior may degrade fidelity to the true Laplace approximation. To alleviate this problem, without increasing significantly the computational cost, we propose the Quadratic Laplace Approximation (QLA). QLA approximates each second order factor in the approximate Laplace log-posterior using a rank-one factor obtained via efficient power iterations. QLA is expected to yield a posterior precision closer to that of the full Laplace without forming the full Hessian, which is typically intractable. For prediction, QLA also uses the linearized model. Empirically, QLA yields modest yet consistent uncertainty estimation improvements over LLA on five regression datasets.
            </p>
          </details>
        </div>
      </article>

      <article class="publication-card accent-oxblood" id="sgd-implicit-bias" data-year="2026" data-tags="bounds code generalization sgd ldt" data-title="large deviation theory analysis implicit bias sgd neurocomputing code">
        <div class="pub-year"><span>2026</span></div>
        <div class="pub-content">
          <h2>A Large Deviation Theory Analysis on the Implicit Bias of SGD</h2>
          <p class="authors">Luis A. Ortega, Andres R. Masegosa</p>
          <p class="venue">Neurocomputing 2026</p>
          <p>Uses large-deviation theory to explain why mini-batch SGD may prefer better-concentrated solutions.</p>
          <div class="tag-row compact">
            <span>SGD</span>
            <span>LDT</span>
            <span>Generalization</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              Stochastic Gradient Descent (SGD) is the primary optimization method used in deep learning, yet the reasons behind its ability to select models that generalize effectively remain unclear. This paper develops a new perspective based on Large Deviation Theory (LDT). We show that the generalization error can be decomposed into three terms: the expected loss, a component that reflects the concentration of the empirical loss around its mean, and a component that captures the abnormality of deviations arising from stochastic sampling. This decomposition highlights a key difference between optimization methods: while full-batch Gradient Descent tends to exploit poorly concentrated and abnormal fluctuations&mdash;often leading to overfitting&mdash;mini-batch SGD naturally biases the search towards models with tighter concentration and fewer abnormal deviations. The analysis relies on standard assumptions such as i.i.d data and smooth loss functions. Experiments with deep convolutional networks support the theoretical findings, showing that smaller batch sizes and l2 regularization reinforce the preference for models with smaller generalization error. These results position LDT as a useful tool for understanding implicit regularization in SGD and suggest directions for extending this perspective to broader machine learning settings.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://github.com/Ludvins/sgd-implicit-bias-ldt">Code</a>
        </div>
      </article>

      <article class="publication-card accent-oxblood" id="pac-chernoff-bounds" data-year="2025" data-tags="bounds code generalization pac-chernoff interpolation" data-title="pac chernoff bounds understanding generalization interpolation regime jair ecai spotlight">
        <div class="pub-year"><span>2025</span></div>
        <div class="pub-content">
          <h2>PAC-Chernoff Bounds: Understanding Generalization in the Interpolation Regime</h2>
          <p class="authors">Andres R. Masegosa, Luis A. Ortega</p>
          <p class="venue">JAIR &middot; ECAI 2025 Spotlight</p>
          <p>A distribution-dependent PAC-Chernoff bound and smoothness framework for over-parameterized interpolators.</p>
          <div class="tag-row compact">
            <span>PAC-Chernoff</span>
            <span>Interpolation</span>
            <span>Bounds</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              This paper introduces a distribution-dependent PAC-Chernoff bound that exhibits perfect tightness for interpolators, even within over-parameterized model classes. This bound, which relies on basic principles of Large Deviation Theory, defines a natural measure of the smoothness of a model, characterized by simple real-valued functions. Building upon this bound and the new concept of smoothness, we present an unified theoretical framework revealing why certain interpolators show an exceptional generalization, while others falter. We theoretically show how a wide spectrum of modern learning methodologies, encompassing techniques such as l2-norm, distance-from-initialization and input-gradient regularization, in combination with data augmentation, invariant architectures, and over-parameterization, collectively guide the optimizer toward smoother interpolators, which, according to our theoretical framework, are the ones exhibiting superior generalization performance. This study shows that distribution-dependent bounds serve as a powerful tool to understand the complex dynamics behind the generalization capabilities of over-parameterized interpolators.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://www.jair.org/index.php/jair/article/view/17036">PDF</a>
          <a href="https://github.com/Ludvins/2024_PAC-Chernoff-Bound">Code</a>
        </div>
      </article>

      <article class="publication-card accent-olive" id="pac-bayes-chernoff" data-year="2024" data-tags="bounds bayesian-dl pac-bayes unbounded losses" data-title="pac bayes chernoff bounds unbounded losses neurips">
        <div class="pub-year"><span>2024</span></div>
        <div class="pub-content">
          <h2>PAC-Bayes-Chernoff Bounds for Unbounded Losses</h2>
          <p class="authors">Ioar Casado, Luis A. Ortega, Aritz Perez, Andres R. Masegosa</p>
          <p class="venue">NeurIPS 2024</p>
          <p>Extends Cramer-Chernoff style bounds to PAC-Bayesian settings with unbounded losses.</p>
          <div class="tag-row compact">
            <span>PAC-Bayes</span>
            <span>Bounds</span>
            <span>Unbounded loss</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              We introduce a new PAC-Bayes oracle bound for unbounded losses that extends Cram&eacute;r-Chernoff bounds to the PAC-Bayesian setting. The proof technique relies on controlling the tails of certain random variables involving the Cram&eacute;r transform of the loss. Our approach naturally leverages properties of Cram&eacute;r-Chernoff bounds, such as exact optimization of the free parameter in many PAC-Bayes bounds. We highlight several applications of the main theorem. Firstly, we show that our bound recovers and generalizes previous results. Additionally, our approach allows working with richer assumptions that result in more informative and potentially tighter bounds. In this direction, we provide a general bound under a new model-dependent assumption from which we obtain bounds based on parameter norms and log-Sobolev inequalities. Notably, many of these bounds can be minimized to obtain distributions beyond the Gibbs posterior and provide novel theoretical coverage to existing regularization techniques.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://openreview.net/forum?id=CyzZeND3LB">PDF</a>
        </div>
      </article>

      <article class="publication-card accent-olive" id="variational-linearized-laplace" data-year="2024" data-tags="bayesian-dl inference code variational gp laplace" data-title="variational linearized laplace approximation bayesian deep learning icml code">
        <div class="pub-year"><span>2024</span></div>
        <div class="pub-content">
          <h2>Variational Linearized Laplace Approximation for Bayesian Deep Learning</h2>
          <p class="authors">Luis A. Ortega, Simon Rodriguez-Santana, Daniel Hernandez-Lobato</p>
          <p class="venue">ICML 2024</p>
          <p>Approximates LLA via sparse variational Gaussian processes with sub-linear training costs.</p>
          <div class="tag-row compact">
            <span>Bayesian DL</span>
            <span>Variational GP</span>
            <span>LLA</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              The Linearized Laplace Approximation (LLA) has been recently used to perform uncertainty estimation on the predictions of pre-trained deep neural networks (DNNs). However, its widespread application is hindered by significant computational costs, particularly in scenarios with a large number of training points or DNN parameters. Consequently, additional approximations of LLA, such as Kronecker-factored or diagonal approximate GGN matrices, are utilized, potentially compromising the model's performance. To address these challenges, we propose a new method for approximating LLA using a variational sparse Gaussian Process (GP). Our method is based on the dual RKHS formulation of GPs and retains, as the predictive mean, the output of the original DNN. Furthermore, it allows for efficient stochastic optimization, which results in sub-linear training time in the size of the training dataset. Specifically, its training cost is independent of the number of training points. We compare our proposed method against accelerated LLA (ELLA), which relies on the Nystr&ouml;m approximation, as well as other LLA variants employing the sample-then-optimize principle. Experimental results, both on regression and classification datasets, show that our method outperforms these already existing efficient variants of LLA, both in terms of the quality of the predictive distribution and in terms of total computational time.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://proceedings.mlr.press/v235/ortega24a.html">PDF</a>
          <a href="https://github.com/Ludvins/Variational-LLA">Code</a>
        </div>
      </article>

      <article class="publication-card accent-olive" id="cold-posterior-effect" data-year="2024" data-tags="bayesian-dl code posterior underfitting" data-title="cold posterior effect indicates underfitting tmlr code">
        <div class="pub-year"><span>2024</span></div>
        <div class="pub-content">
          <h2>The Cold Posterior Effect Indicates Underfitting</h2>
          <p class="authors">Yijie Zhang, Yi-Shan Wu, Luis A. Ortega, Andres R. Masegosa</p>
          <p class="venue">TMLR 2024</p>
          <p>Reframes the cold posterior effect as evidence of underfitting in misspecified Bayesian posteriors.</p>
          <div class="tag-row compact">
            <span>Bayesian DL</span>
            <span>Posterior</span>
            <span>Underfitting</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              The cold posterior effect (CPE) (Wenzel et al., 2020) in Bayesian deep learning shows that, for posteriors with a temperature T&lt;1, the resulting posterior predictive could have better performance than the Bayesian posterior (T=1). As the Bayesian posterior is known to be optimal under perfect model specification, many recent works have studied the presence of CPE as a model misspecification problem, arising from the prior and/or from the likelihood. In this work, we provide a more nuanced understanding of the CPE as we show that misspecification leads to CPE only when the resulting Bayesian posterior underfits. In fact, we theoretically show that if there is no underfitting, there is no CPE. Furthermore, we show that these tempered posteriors with (T&lt;1) are indeed proper Bayesian posteriors with a different combination of likelihood and prior parameterized by T. This observation validates the adjustment of the temperature hyperparameter T as a straightforward approach to mitigate underfitting in the Bayesian posterior. In essence, we show that by fine-tuning the temperature T we implicitly utilize alternative Bayesian posteriors, albeit with less misspecified likelihood and prior distributions.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://openreview.net/forum?id=GZORXGxHHT">PDF</a>
          <a href="https://github.com/pyijiezhang/cpe-underfit">Code</a>
        </div>
      </article>

      <article class="publication-card accent-olive" id="deep-variational-implicit-processes" data-year="2023" data-tags="bayesian-dl inference code implicit processes" data-title="deep variational implicit processes iclr code">
        <div class="pub-year"><span>2023</span></div>
        <div class="pub-content">
          <h2>Deep Variational Implicit Processes</h2>
          <p class="authors">Luis A. Ortega, Simon Rodriguez-Santana, Daniel Hernandez-Lobato</p>
          <p class="venue">ICLR 2023</p>
          <p>A multi-layer implicit-process generalization for flexible function-space inference.</p>
          <div class="tag-row compact">
            <span>DVIP</span>
            <span>Implicit Processes</span>
            <span>Inference</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              Implicit processes (IPs) are a generalization of Gaussian processes (GPs). IPs may lack a closed-form expression but are easy to sample from. Examples include, among others, Bayesian neural networks or neural samplers. IPs can be used as priors over functions, resulting in flexible models with well-calibrated prediction uncertainty estimates. Methods based on IPs usually carry out function-space approximate inference, which overcomes some of the difficulties of parameter-space approximate inference. Nevertheless, the approximations employed often limit the expressiveness of the final model, resulting, e.g., in a Gaussian predictive distribution, which can be restrictive. We propose here a multi-layer generalization of IPs called the Deep Variational Implicit process (DVIP). This generalization is similar to that of deep GPs over GPs, but it is more flexible due to the use of IPs as the prior distribution over the latent functions. We describe a scalable variational inference algorithm for training DVIP and show that it outperforms previous IP-based methods and also deep GPs. We support these claims via extensive regression and classification experiments. We also evaluate DVIP on large datasets with up to several million data instances to illustrate its good scalability and performance.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://openreview.net/forum?id=8aeSJNbmbQq">PDF</a>
          <a href="https://github.com/Ludvins/2023-ICLR-DVIP">Code</a>
        </div>
      </article>

      <article class="publication-card accent-olive" id="diversity-generalization-ensembles" data-year="2022" data-tags="bayesian-dl code ensembles generalization" data-title="diversity and generalization neural network ensembles aistats code">
        <div class="pub-year"><span>2022</span></div>
        <div class="pub-content">
          <h2>Diversity and Generalization in Neural Network Ensembles</h2>
          <p class="authors">Luis A. Ortega, Rafael Cabanas, Andres R. Masegosa</p>
          <p class="venue">AISTATS 2022</p>
          <p>Connects ensemble diversity, generalization error, and common model-combination strategies.</p>
          <div class="tag-row compact">
            <span>Ensembles</span>
            <span>Diversity</span>
            <span>Generalization</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              Ensembles are widely used in machine learning and, usually, provide state-of-the-art performance in many prediction tasks. From the very beginning, the diversity of an ensemble has been identified as a key factor for the superior performance of these models. But the exact role that diversity plays in ensemble models is poorly understood, specially in the context of neural networks. In this work, we combine and expand previously published results in a theoretically sound framework that describes the relationship between diversity and ensemble performance for a wide range of ensemble methods. More precisely, we provide sound answers to the following questions: how to measure diversity, how diversity relates to the generalization error of an ensemble, and how diversity is promoted by neural network ensemble algorithms. This analysis covers three widely used loss functions, namely, the squared loss, the cross-entropy loss, and the 0-1 loss; and two widely used model combination strategies, namely, model averaging and weighted majority vote. We empirically validate this theoretical analysis with neural network ensembles.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://proceedings.mlr.press/v151/ortega22a.html">PDF</a>
          <a href="https://github.com/PGM-Lab/2022-AISTATS-diversity">Code</a>
        </div>
      </article>

      <article class="publication-card accent-olive" id="sparse-implicit-processes" data-year="2022" data-tags="bayesian-dl inference code implicit processes" data-title="correcting model bias sparse implicit processes icml workshop code">
        <div class="pub-year"><span>2022</span></div>
        <div class="pub-content">
          <h2>Correcting Model Bias with Sparse Implicit Processes</h2>
          <p class="authors">Simon Rodriguez-Santana, Luis A. Ortega, Daniel Hernandez-Lobato, Bryan Zaldivar</p>
          <p class="venue">ICML Workshop 2022</p>
          <p>Shows sparse implicit processes can correct model bias when the assumed mechanism differs from the data.</p>
          <div class="tag-row compact">
            <span>SIP</span>
            <span>Implicit Processes</span>
            <span>Inference</span>
            <span>Code</span>
          </div>
          <details class="publication-abstract">
            <summary>Abstract</summary>
            <p>
              Model selection in machine learning (ML) is a crucial part of the Bayesian learning procedure. Model choice may impose strong biases on the resulting predictions, which can hinder the performance of methods such as Bayesian neural networks and neural samplers. On the other hand, newly proposed approaches for Bayesian ML exploit features of approximate inference in function space with implicit stochastic processes (a generalization of Gaussian processes). The approach of Sparse Implicit Processes (SIP) is particularly successful in this regard, since it is fully trainable and achieves flexible predictions. Here, we expand on the original experiments to show that SIP is capable of correcting model bias when the data generating mechanism differs strongly from the one implied by the model. We use synthetic datasets to show that SIP is capable of providing predictive distributions that reflect the data better than the exact predictions of the initial, but wrongly assumed model.
            </p>
          </details>
        </div>
        <div class="pub-actions">
          <a href="https://arxiv.org/abs/2207.10673">PDF</a>
          <a href="https://github.com/simonrsantana/sparse-implicit-processes">Code</a>
        </div>
      </article>

      <article class="publication-card accent-oxblood ongoing-card" data-year="ongoing" data-tags="bayesian-dl inference" data-title="fixed mean gaussian processes ad hoc bayesian deep learning">
        <div class="pub-year"><span>Ongoing</span></div>
        <div class="pub-content">
          <h2>Fixed-Mean Gaussian Processes for ad-hoc Bayesian Deep Learning</h2>
          <p>Converting models to Bayesian predictors by creating a Gaussian process with fixed predictive mean.</p>
        </div>
        <div class="pub-actions">
          <a href="https://arxiv.org/abs/2412.04177">Preprint</a>
        </div>
      </article>

      <article class="publication-card no-actions accent-oxblood ongoing-card" data-year="ongoing" data-tags="bounds bayesian-dl" data-title="regularization as estimation pac bayes chernoff approach">
        <div class="pub-year"><span>Ongoing</span></div>
        <div class="pub-content">
          <h2>Regularization as Estimation, A PAC-Bayes-Chernoff Approach</h2>
          <p>A prescriptive framework that reframes regularization as a statistical estimation problem.</p>
        </div>
      </article>

      <article class="publication-card no-actions accent-oxblood ongoing-card" data-year="ongoing" data-tags="bayesian-dl bounds" data-title="revisiting marginal likelihood pac bayesian lens">
        <div class="pub-year"><span>Ongoing</span></div>
        <div class="pub-content">
          <h2>Revisiting the Marginal Likelihood through a PAC-Bayesian Lens</h2>
          <p>Generalization in Bayesian models depends on factors beyond marginal likelihood alone.</p>
        </div>
      </article>

      <p class="no-results" hidden>No publications match the current filters.</p>
    </div>
  </section>

</main>
