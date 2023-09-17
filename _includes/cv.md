<p align='justify'>
I am a PhD Student at the
<b>Autonomous University of Madrid</b>
and study foundational topics in <b>machine learning</b> and
<b>function space inference</b>.
My research focuses on studying implicit processes and methods of approximating the intractable distributions
they define.
</p><br>


## <i class="fa fa-chevron-right"></i> Current Position

<table class="table table-hover">
  <tr>
    <td style='padding-right:0;'>
      <span class='cvdate'>2021&nbsp;-&nbsp;Present</span>
      <p markdown="1" style='margin: 0'><strong>Research Personnel, Ph.D. Student granted with FPI-UAM Scholarship</strong>, <em>Autonomous University</em>          , Madrid
<br>
          <p style='margin-top:-1em;margin-bottom:0em' markdown='1'>
          <br> Thesis: *New Learning Methods based on Implicit Processes*
          <br> Advisor: <a href="https://dhnzl.org" target="_blank">Daniel Hernández Lobato</a>
          </p>
      </p>
    </td>
  </tr>
</table>


## <i class="fa fa-chevron-right"></i> Education

<table class="table table-hover">
  <tr>
    <td>
      <span class='cvdate'>2021&nbsp;-&nbsp;Present</span>
      <strong>Ph.D. Student</strong>, <em>Autonomous University of Madrid</em>
      <br>
        <p style='margin-top:-1em;margin-bottom:0em' markdown='1'>
        <br> Thesis: *New Learning Methods based on Implicit Processes*
        <br> Advisor: <a href="https://dhnzl.org" target="_blank">Daniel Hernández Lobato</a>
        </p>
    </td>
  </tr>
  <tr>
    <td>
      <span class='cvdate'>2020&nbsp;-&nbsp;2022</span>
      <strong>M.S. in Data Science</strong>, <em>Autonomous University of Madrid</em>
      <br>
    </td>
  </tr>
  <tr>
    <td>
      <span class='cvdate'>2015&nbsp;-&nbsp;2020</span>
      <strong>B.S. in Computer Science</strong>, <em>University of Granada</em>
      <br>
    </td>
  </tr>
  <tr>
    <td>
      <span class='cvdate'>2015&nbsp;-&nbsp;2020</span>
      <strong>B.S. in Mathematics</strong>, <em>University of Granada</em>
      <br>
    </td>
  </tr>
</table>


## <i class="fa fa-chevron-right"></i> Previous Positions
<table class="table table-hover">
<tr>
  <td style='padding-right:0;'>
<span class='cvdate'>2021&nbsp;-&nbsp;2021</span>
<p markdown="1" style='margin: 0'><strong>Research Assistant</strong>, <em>University of Almería</em><span markdown="1" style="color:grey;font-size:1.3rem;margin: 0">
(with <a href="https://andresmasegosa.github.io/" target="_blank">Andrés R. Masegosa</a> studing the effect of diversity on Deep Neural Network ensembles.)
</span></p>
  </td>
</tr>
</table>


## <i class="fa fa-chevron-right"></i> Honors & Awards
<table class="table table-hover">
<tr>
  <td>
  <div style='float: right'>2021</div>
  <div>
    Granted FPI-UAM Scholarship. Competitive Predoctoral Contract for Training Research Personnel
    <br><p style="color:grey;font-size:1.2rem">Department of Computer Science, Autonomous University of Madrid</p>
  </div>
  </td>
  <!-- <td class='col-md-2' style='text-align:right;'>2021</td> -->
</tr>
<tr>
  <td>
  <div style='float: right'>2020</div>
  <div>
    Research Collaboration Scholarship
    <br><p style="color:grey;font-size:1.2rem">Department of Computer Science, Autonomous University of Madrid</p>
  </div>
  </td>
  <!-- <td class='col-md-2' style='text-align:right;'>2020</td> -->
</tr>
<tr>
  <td>
  <div style='float: right'>2020</div>
  <div>
    Granted Highest Mark on Bachelor's Thesis, 10/10. Statistical Models with Variational Methods
    <br><p style="color:grey;font-size:1.2rem">Department of Computer Science and Faculty of Science, Granada</p>
  </div>
  </td>
  <!-- <td class='col-md-2' style='text-align:right;'>2020</td> -->
</tr>
</table>


## <i class="fa fa-chevron-right"></i> Publications

<!-- [<a href="https://github.com/bamos/cv/blob/master/publications/all.bib">BibTeX</a>] -->
<!-- Representative publications that I am a primary author on are -->
<!-- <span style='background-color: #ffffd0'>highlighted.</span> -->
<br>
<!-- [<a href="https://scholar.google.com/citations?user=1Ly8qeoAAAAJ">Google Scholar</a>; 14+ citations, h-index: 1+] -->

<h2>2023</h2>
<table class="table table-hover">

<tr id="tr-ortega2023deep" >
<td align='right' style='padding-left:0;padding-right:0;'>
1.
</td>
<td>
<a href='https://openreview.net/forum?id=8aeSJNbmbQq' target='_blank'><img src="images/publications/ortega2023deep.png" onerror="this.style.display='none'" class="publicationImg" /></a> 
<em><a href='https://openreview.net/forum?id=8aeSJNbmbQq' target='_blank'>Deep Variational Implicit Processes</a> </em> 
[<a href='javascript:;'
    onclick='$("#abs_ortega2023deep").toggle()'>abs</a>] [<a href='https://github.com/Ludvins/DeepVariationalImplicitProcesses' target='_blank'>code</a>] <br>
<strong>Luis&nbsp;A.&nbsp;Ortega</strong>, Simón&nbsp;Rodríguez-Santana, and <a href='https://dhnzl.org' target='_blank'>Daniel&nbsp;Hernández-Lobato</a><br>
International Conference on Learning Representations, 2023 2023  <br>

<div id="abs_ortega2023deep" style="text-align: justify; display: none" markdown="1">
Implicit processes (IPs) are a generalization of Gaussian processes (GPs). IPs may lack a closed-form expression but are easy to sample from. Examples include, among others, Bayesian neural networks or neural samplers. IPs can be used as priors over functions, resulting in flexible models with well-calibrated prediction uncertainty estimates. Methods based on IPs usually carry out function-space approximate inference, which overcomes some of the difficulties of parameter-space approximate inference. Nevertheless, the approximations employed often limit the expressiveness of the final model, resulting, e.g., in a Gaussian predictive distribution, which can be restrictive. We propose here a multi-layer generalization of IPs called the Deep Variational Implicit process (DVIP). This generalization is similar to that of deep GPs over GPs, but it is more flexible due to the use of IPs as the prior distribution over the latent functions. We describe a scalable variational inference algorithm for training DVIP and show that it outperforms previous IP-based methods and also deep GPs. We support these claims via extensive regression and classification experiments. We also evaluate DVIP on large datasets with up to several million data instances to illustrate its good scalability and performance. 
selected  = false
</div>

</td>
</tr>

</table>
<h2>2022</h2>
<table class="table table-hover">

<tr id="tr-pmlr-v151-ortega22a" >
<td align='right' style='padding-left:0;padding-right:0;'>
2.
</td>
<td>
<a href='https://proceedings.mlr.press/v151/ortega22a.html' target='_blank'><img src="images/publications/pmlr-v151-ortega22a.png" onerror="this.style.display='none'" class="publicationImg" /></a> 
<em><a href='https://proceedings.mlr.press/v151/ortega22a.html' target='_blank'>Diversity and Generalization in Neural Network Ensembles</a> </em> 
[<a href='javascript:;'
    onclick='$("#abs_pmlr-v151-ortega22a").toggle()'>abs</a>] [<a href='https://github.com/PGM-Lab/2022-AISTATS-diversity' target='_blank'>code</a>] <br>
<strong>Luis&nbsp;A.&nbsp;Ortega</strong>, <a href='https://www.linkedin.com/in/rcabanasdepaz' target='_blank'>Rafael&nbsp;Cabañas</a>, and <a href='https://andresmasegosa.github.io/' target='_blank'>Andrés&nbsp;R.&nbsp;Masegosa</a><br>
AISTATS 2022  <br>

<div id="abs_pmlr-v151-ortega22a" style="text-align: justify; display: none" markdown="1">
Ensembles are widely used in machine learning and, usually, provide state-of-the-art performance in many prediction tasks. From the very beginning, the diversity of an ensemble has been identified as a key factor for the superior performance of these models. But the exact role that diversity plays in ensemble models is poorly understood, specially in the context of neural networks. In this work, we combine and expand previously published results in a theoretically sound framework that describes the relationship between diversity and ensemble performance for a wide range of ensemble methods. More precisely, we provide sound answers to the following questions: how to measure diversity, how diversity relates to the generalization error of an ensemble, and how diversity is promoted by neural network ensemble algorithms. This analysis covers three widely used loss functions, namely, the squared loss, the cross-entropy loss, and the 0-1 loss; and two widely used model combination strategies, namely, model averaging and weighted majority vote. We empirically validate this theoretical analysis with neural network ensembles.
</div>

</td>
</tr>


<tr id="tr-santana2022correcting" >
<td align='right' style='padding-left:0;padding-right:0;'>
3.
</td>
<td>
<a href='https://arxiv.org/abs/2207.10673' target='_blank'><img src="images/publications/santana2022correcting.png" onerror="this.style.display='none'" class="publicationImg" /></a> 
<em><a href='https://arxiv.org/abs/2207.10673' target='_blank'>Correcting Model Bias with Sparse Implicit Processes</a> </em> 
[<a href='javascript:;'
    onclick='$("#abs_santana2022correcting").toggle()'>abs</a>] [<a href='https://github.com/simonrsantana/sparse-implicit-processes' target='_blank'>code</a>] <br>
Simón&nbsp;Rodríguez-Santana, <strong>Luis&nbsp;A.&nbsp;Ortega</strong>, <a href='https://dhnzl.org' target='_blank'>Daniel&nbsp;Hernández-Lobato</a>, and <a href='https://www.linkedin.com/in/bryan-zaldivar/' target='_blank'>Bryan&nbsp;Zaldívar</a><br>
Beyond Bayes Workshop, ICML 2022  <br>

<div id="abs_santana2022correcting" style="text-align: justify; display: none" markdown="1">
Model selection in machine learning (ML) is a crucial part of the Bayesian learning procedure. Model choice may impose strong biases on the resulting predictions, which can hinder the performance of methods such as Bayesian neural networks and neural samplers. On the other hand, newly proposed approaches for Bayesian ML exploit features of approximate inference in function space with implicit stochastic processes (a generalization of Gaussian processes). The approach of Sparse Implicit Processes (SIP) is particularly successful in this regard, since it is fully trainable and achieves flexible predictions. Here, we expand on the original experiments to show that SIP is capable of correcting model bias when the data generating mechanism differs strongly from the one implied by the model. We use synthetic datasets to show that SIP is capable of providing predictive distributions that reflect the data better than the exact predictions of the initial, but wrongly assumed model.
</div>

</td>
</tr>

</table>


## <i class="fa fa-chevron-right"></i> Open Source Repositories
<table class="table table-hover">
<tr>
  <td align='right' style='padding-right:0;padding-left:0;'>1.</td>
  <td>
    <span class='cvdate'>2020</span>
    <a href="https://github.com/Ludvins/Statistical-Models-with-Variational-Methods">Ludvins/Statistical-Models-with-Variational-Methods</a> |
    <i class="fa fas fa-star"></i> 8 |
    <em>Bachelor's Thesis, Statistical Models with Variational Methods.</em>
    <!--  -->
    <!--     Ludvins/Statistical-Models-with-Variational-Methods  -->
    <!--  -->
  </td>
</tr>
<tr>
  <td align='right' style='padding-right:0;padding-left:0;'>2.</td>
  <td>
    <span class='cvdate'>2017</span>
    <a href="https://github.com/libreim/apuntesDGIIM">libreim/apuntesDGIIM</a> |
    <i class="fa fas fa-star"></i> 79 |
    <em>Divulgation Group Destinated to the Double Degree in Computer Science and Mathematics, Granada</em>
    <!--  -->
    <!--     libreim/apuntesDGIIM  -->
    <!--  -->
  </td>
</tr>
</table>
