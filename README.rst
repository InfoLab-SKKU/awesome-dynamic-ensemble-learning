Awesome Dynamic Ensemble Learning
=========================
 
.. image:: https://img.shields.io/github/license/yzhao062/awesome-ensemble-learning.svg?color=blue
   :target: https://github.com/yzhao062/awesome-ensemble-learning/blob/master/LICENSE
   :alt: License

.. image:: https://awesome.re/badge-flat2.svg
   :target: https://awesome.re/badge-flat2.svg
   :alt: Awesome



Table of Contents
=================

- `Introduction <#introduction>`_
- `Steps of Dynamic Ensemble Selection <#steps-of-dynamic-ensemble-selection>`_
- `Existing Popular DES Techniques <#existing-popular-des-techniques>`_
- `Python Libraries <#python-libraries>`_
- `Existing Libraries and Included Techniques <#existing-libraries-and-included-techniques>`_
- `Benchmark <#benchmark>`_
- `References <#references>`_


Introduction
============
`Dynamic Ensemble Learning <https://en.wikipedia.org/wiki/Ensemble_learning>`_
(often referred to as *Dynamic Ensemble Selection (DES)* or *Dynamic Classifier Selection (DCS)*)
is an advanced branch of ensemble learning focused on selecting the most
competent classifier or a subset of classifiers for each query sample.
Unlike static ensemble methods, dynamic ensemble learning adapts the decision
process according to the local characteristics of the input space, often leading
to improve robustness and predictive performance [#Ko2008Dynamic]_.

Dynamic ensemble learning has gained significant attention in machine learning
research due to its effectiveness in handling complex, imbalanced, noisy, and
non-stationary datasets. It has been successfully applied in various domains,
including medical diagnosis, cybersecurity, fault detection, computer vision,
and autonomous systems [#Cruz2018Dynamic]_.


Steps of Dynamic Ensemble Selection: 
======================================

.. image:: images/pipeline.png
   :width: 600px
   :align: center
   :alt: Dynamic Ensemble Selection Pipeline


The Dynamic Ensemble Selection (DES) framework can be divided into four
main stages, as illustrated in the workflow diagram.

🔹 **Data Preparation**
    This stage includes the initial processing of the dataset before
    training. It consists of preprocessing the data (Step 1), such as
    handling missing values, normalization, or feature transformation,
    followed by partitioning the dataset (Step 2) into training,
    validation (DSEL), and testing subsets.

🔹 **DES Training Phase**
    During training, a diverse pool of classifiers is generated (Step 3)
    and trained using the designated training data (Step 4). For
    meta-learning-based DES methods, an additional meta-training stage
    (Step 5) is performed to learn how to estimate classifier competence
    in different regions of the feature space.

🔹 **DES Selection Phase**
    Given a query sample, the algorithm first determines its Region of
    Competence (RoC) (Step 6), typically using the nearest neighbors in
    the DSEL. The competence of each
    classifier is then estimated within this region (Step 7). Based on
    these competence estimates, the most suitable classifiers are
    selected to form a dynamic ensemble (Step 8).

🔹 **DES Prediction Phase**
    The selected ensemble produces predictions for the query sample,
    which are combined using an aggregation strategy such as majority
    voting, weighted voting, or probability averaging (Step 9). The
    aggregated result constitutes the final prediction of the DES system.

Existing Popular DES techniques: 
=====================================

+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| Technique         | RoC Definition         | Selection Criteria                   | Reference                                                        | Year |
+===================+========================+======================================+==================================================================+======+
| DES_Clustering    | Clustering             | Accuracy & Diversity                 | Soares et al. [#soares2006using]_                                | 2006 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| DES-KNN           | K-NN                   | Accuracy & Diversity                 | Soares et al. [#soares2006using]_                                | 2006 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| KNORA-E           | K-NN                   | Oracle                               | Ko et al. [#Ko2008Dynamic]_                                      | 2008 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| KNORA-U           | K-NN                   | Oracle                               | Ko et al. [#Ko2008Dynamic]_                                      | 2008 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| DES-RRC           | Potential function     | Probabilistic                        | Woloszynski et al. [#woloszynski2011probabilistic]_              | 2011 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| DES-KL            | Potential function     | Probabilistic                        | Woloszynski et al. [#woloszynski2012measure]_                    | 2012 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| DES-P             | Potential function     | Probabilistic                        | Woloszynski et al. [#woloszynski2012measure]_                    | 2012 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| KNOP              | K-NN                   | Behavior                             | Cavalin et al. [#cavalin2013dynamic]_                            | 2013 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| META-DES          | K-NN                   | Meta-Learning                        | Cruz et al. [#cruz2015meta]_                                     | 2015 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| FIRE-DES          | K-NN                   | Pruning-based                        | Oliveira et al. [#oliveira2017online]_                           | 2017 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| DES-MI            | K-NN                   | Weighted Accuracy (minority-aware)   | Garcia et al. [#garcia2018dynamic]_                              | 2018 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| FIRE-DES++        | K-NN                   | Pruning-based                        | Cruz et al. [#cruz2019fire]_                                     | 2019 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| GNN-DES           | Graph (Decision space) | Meta-Learning                        | Souza et al. [#souza2024dynamic]_                                | 2024 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| FH-DES            | Fuzzy hyperboxes       | Fuzzy logic                          | Davtalab et al. [#davtalab2024scalable]_                         | 2024 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| imDEF             | Decision space         | Referee System                       | Zhu et al. [#zhu2025dynamic]_                                    | 2025 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| DES-AS            | K-NN                   | Synergy                              | Zhang et al. [#zhang2025dynamic]_                                | 2025 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+
| IncA-DES          | K-d Tree               | Adaptive selection                   | Barboza et al. [#barboza2025inca]_                               | 2025 |
+-------------------+------------------------+--------------------------------------+------------------------------------------------------------------+------+

+-------------------+-----------------------------------------------------------+
| Technique         | GitHub                                                    | 
+===================+===========================================================+
| FH-DES            | `Repository <https://github.com/redavtalab/FH-DES>`_      |
+-------------------+-----------------------------------------------------------+
| IncA-DES          | `Repository <https://github.com/eduardovlb/IncA-DES>`_    |
+-------------------+-----------------------------------------------------------+


Python Libraries: 
===================================== 

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg?style=flat&logo=python&logoColor=white
   :target: https://www.python.org
   :alt: Python Version 

1. DESLib [#Cruz2020DESlib]_   Github: `here <https://github.com/scikit-learn-contrib/deslib>`_
2. InfoDESLib [#juraev2024infodeslib]_  Github: `here <https://github.com/InfoLab-SKKU/infodeslib>`_

Existing Libraries and included techniques: 
==================================================================

+----------------------------------------+-------+----------+-------------+
| Technique                              | Type  | DESLib   | InfoDESLib  |
+========================================+=======+==========+=============+
| Modified Classifier Rank (MR)          | DCS   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| Overall Local Accuracy (OLA)           | DCS   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| Local Class Accuracy (LCA)             | DCS   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| Modified Local Accuracy (MLA)          | DCS   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| Multiple Classifier Behaviour (MCB)    | DCS   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| A Priori Selection                     | DCS   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| A Posteriori Selection                 | DCS   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| DES-KNN                                | DES   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| KNORA-E                                | DES   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| KNORA-U                                | DES   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| KNORA-E-W                              | DES   | --       | ✓           |
+----------------------------------------+-------+----------+-------------+
| KNORA-U-W                              | DES   | --       | ✓           |
+----------------------------------------+-------+----------+-------------+
| DES-P                                  | DES   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| KNOP                                   | DES   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| DES-RRC                                | DES   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| DES-KL                                 | DES   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| DES-Exponential                        | DES   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| DES-Logarithmic                        | DES   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| DES-Minimum Difference                 | DES   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| DES-Clustering                         | DES   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| DES-MI                                 | DES   | ✓        | --          |
+----------------------------------------+-------+----------+-------------+
| FIRE-DES (DFP)                         | DES   | ✓        | ✓           |
+----------------------------------------+-------+----------+-------------+
| Late Fusion / Feature Sets             | --    | --       | ✓           |
+----------------------------------------+-------+----------+-------------+
| Explainable AI (XAI) support           | --    | --       | ✓           |
+----------------------------------------+-------+----------+-------------+

Benchmark: 
========================= 

We conducted a large-scale benchmark comparing 16 DES methods and 3 classical static ensemble techniques across 25 diverse tabular datasets from the PMLB repository. All methods were evaluated under a unified experimental protocol using a heterogeneous pool of 17 base classifiers, a 60/20/20 train–DSEL–test split, and macro-F1 as the primary evaluation metric.

Overall, DES methods consistently outperformed static ensemble approaches, confirming the advantage of dynamically selecting competent classifiers for individual test instances. Classical DES methods such as KNOP, META-DES, and KNORA-U achieved the highest average rankings, while recent approaches, including IncA-DES, GNN-DES, and FH-DES, also demonstrated strong performance.

From a computational perspective, most DES techniques exhibited inference times, training costs, and memory requirements comparable to simple voting-based ensembles. Only probabilistic and meta-learning-based DES variants incurred noticeable computational overhead due to their additional competence estimation and modeling components. These findings demonstrate that DES can provide substantial predictive improvements while remaining computationally practical and scalable for real-world applications.

.. image:: images/benchmark_results_1.png
   :width: 700px
   :align: center
   :alt: Dynamic Ensemble Selection Benchmark Results 

We also evaluated the sensitivity of the ensemble methods to three key parameters shared across most DES frameworks: the RoC size, DSEL size, and classifier pool size. Experiments were conducted on the Fars dataset using the default configuration of (k=7), 20% DSEL, and 17 base classifiers.

Overall, the results indicate that RoC size has minimal influence on predictive performance and computational cost. In contrast, DSEL size substantially affects all evaluated metrics: larger DSEL sets generally improve accuracy but increase training time, inference time, and memory consumption, while smaller DSEL sets provide better efficiency at the cost of some performance degradation. The classifier pool size also plays an important role; reducing the number of base classifiers consistently improves computational efficiency and, in many cases, slightly increases F1-score, suggesting that excessively large classifier pools may introduce redundancy rather than additional diversity. These findings highlight DSEL size and pool size as the most influential factors for balancing predictive performance and computational efficiency in dynamic ensemble systems.

.. image:: images/benchmark_results_2.png
   :width: 700px
   :align: center
   :alt: Dynamic Ensemble Selection Benchmark Results 


References: 
========================= 

.. [#Ko2008Dynamic] Ko, A. H. R., Sabourin, R., and Britto Jr, A. S. "From Dynamic Classifier Selection to Dynamic Ensemble Selection." Pattern Recognition, 2008.

.. [#Cruz2018Dynamic] Cruz, R. M. O., Sabourin, R., and Cavalcanti, G. D. C. "Dynamic Classifier Selection: Recent Advances and Perspectives." Information Fusion, 2018.

.. [#Cruz2020DESlib] Cruz, R. M. O., Hafemann, L. G., Sabourin, R., and Cavalcanti, G. D. C. "DESlib: A Dynamic Ensemble Selection Library in Python." Journal of Machine Learning Research, 2020.

.. [#juraev2024infodeslib] Juraev, F., Shaker El-Sappagh, Tamer Abuhmed. "Infodeslib: Python library for dynamic ensemble learning using late fusion of multimodal data." KDD 2024 Workshop KiL. 

.. [#soares2006using] Soares, R. G., et al. "Using accuracy and diversity to select classifiers to build ensembles". The 2006 IEEE international joint conference on neural network proceedings 2006.
.. [#woloszynski2011probabilistic] Woloszynski, T., & Kurzynski, M. "A probabilistic model of classifier competence for dynamic ensemble selection". Pattern Recognition, 2011.
.. [#woloszynski2012measure] Woloszynski, T., et al. "A measure of competence based on random classification for dynamic ensemble selection". Information Fusion, 2012.
.. [#cavalin2013dynamic] Cavalin, P. R., et al. "Dynamic selection approaches for multiple classifier systems". Neural computing and applications (2013).
.. [#cruz2015meta] Cruz, R. M., et al. "META-DES: A dynamic ensemble selection framework using meta-learning", Pattern Recognition (2015).
.. [#oliveira2017online] Oliveira, D. V., et al. "Online pruning of base classifiers for dynamic ensemble selection". Pattern Recognition (2017).
.. [#garcia2018dynamic] Garcia, S., et al. "Dynamic ensemble selection for multi-class imbalanced datasets". Information Sciences (2018).
.. [#cruz2019fire] Cruz, R. M., et al. "FIRE-DES++: Enhanced online pruning of base classifiers for dynamic ensemble selection". Pattern Recognition (2019).
.. [#souza2024dynamic] Souza, J. V., et al. "A dynamic multiple classifier system using graph neural network for high dimensional overlapped data". Information Fusion (2024).
.. [#davtalab2024scalable] Davtalab, R., et al. "A scalable dynamic ensemble selection using fuzzy hyperboxes". Information Fusion (2024).
.. [#zhu2025dynamic] Zhu, Y., et al. "Dynamic ensemble framework for imbalanced data classification". IEEE Transactions on Knowledge and Data Engineering (2025).
.. [#zhang2025dynamic] Zhang, Y., et al. "DES-AS: Dynamic ensemble selection based on algorithm Shapley". Pattern Recognition (2025).
.. [#barboza2025inca] Barboza, E., et al. "IncA-DES: An incremental and adaptive dynamic ensemble selection approach using online Kd tree neighborhood search for data streams with concept drift". Information Fusion (2025).

You can check: `Awesome Ensemble Learning <https://github.com/yzhao062/awesome-ensemble-learning#gomes2017a>`_ 
