Awesome Dynamic Ensemble Learning
=========================
 
.. image:: https://img.shields.io/github/license/yzhao062/awesome-ensemble-learning.svg?color=blue
   :target: https://github.com/yzhao062/awesome-ensemble-learning/blob/master/LICENSE
   :alt: License

.. image:: https://awesome.re/badge-flat2.svg
   :target: https://awesome.re/badge-flat2.svg
   :alt: Awesome





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


Existing Popular DES techniques: 
================================================================== 


   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | **Technique** | **RoC Definition** | **Selection Criteria** | **Reference** | **Year** |
   +========================================+=======+==========+=============+=============+
   | DES_Clustering                         | Clustering               | Accuracy & Diversity                | Soares et al. [#soares2006using]_   | 2006     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | DES-KNN                                | K-NN                     | Accuracy & Diversity                | Soares et al. [#soares2006using]_   | 2006     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | KNORA-E                                | K-NN                     | Oracle                              | Ko et al. [#ko2008dynamic]_         | 2008     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | KNORA-U                                | K-NN                     | Oracle                              | Ko et al. [#ko2008dynamic]_         | 2008     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | DES-RRC                                | Potential function       | Probabilistic                       | Woloszynski et al. [#wolo2011]_     | 2011     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | DES-KL                                 | Potential function       | Probabilistic                       | Woloszynski et al. [#wolo2012]_     | 2012     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | DES-P                                  | Potential function       | Probabilistic                       | Woloszynski et al. [#wolo2012]_     | 2012     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | KNOP                                   | K-NN                     | Behavior                            | Cavalin et al. [#cavalin2013]_      | 2013     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | META-DES                               | K-NN                     | Meta-Learning                       | Cruz et al. [#cruz2015meta]_        | 2015     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | FIRE-DES                               | K-NN                     | Pruning-based                       | Oliveira et al. [#oliveira2017]_    | 2017     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | DES-MI                                 | K-NN                     | Weighted Accuracy (minority-aware)  | S. Garcia et al. [#garcia2018]_     | 2018     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | FIRE-DES++                             | K-NN                     | Pruning-based                       | Cruz et al. [#cruz2019fire]_        | 2019     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | GNN-DES                                | Graph (Decision space)   | Meta-Learning                       | Souza et al. [#souza2024]_          | 2024     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | FH-DES                                 | Fuzzy hyperboxes         | Fuzzy logic                         | Davtalab et al. [#davtalab2024]_    | 2024     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | imDEF                                  | Decision space           | Referee System                      | Zhu et al. [#zhu2025]_              | 2025     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | DES-AS                                 | K-NN                     | Synergy                             | Zhang et al. [#zhang2025]_          | 2025     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+
   | IncA-DES                               | K-d Tree                 | Adaptive selection                  | Barboza et al. [#barboza2025]_      | 2025     |
   +----------------------------------------+--------------------------+-------------------------------------+-------------------------------------+----------+



Python Libraries: 

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg?style=flat&logo=python&logoColor=white
   :target: https://www.python.org
   :alt: Python Version 

1. DESLib [#Cruz2020DESlib]_   Github: https://github.com/scikit-learn-contrib/deslib 
2. InfoDESLib [#juraev2024infodeslib]_  Github: https://github.com/InfoLab-SKKU/infodeslib 

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

References: 
========================= 

.. [#Ko2008Dynamic] Ko, A. H. R., Sabourin, R., and Britto Jr, A. S. "From Dynamic Classifier Selection to Dynamic Ensemble Selection." Pattern Recognition, 2008.

.. [#Cruz2018Dynamic] Cruz, R. M. O., Sabourin, R., and Cavalcanti, G. D. C. "Dynamic Classifier Selection: Recent Advances and Perspectives." Information Fusion, 2018.

.. [#Cruz2020DESlib] Cruz, R. M. O., Hafemann, L. G., Sabourin, R., and Cavalcanti, G. D. C. "DESlib: A Dynamic Ensemble Selection Library in Python." Journal of Machine Learning Research, 2020.

.. [#juraev2024infodeslib] Juraev, F., Shaker El-Sappagh, Tamer Abuhmed. "Infodeslib: Python library for dynamic ensemble learning using late fusion of multimodal data." KDD 2024 Workshop KiL. 

.. [#soares2006using] Soares, R. G., et al. (2006).
.. [#wolo2011] Woloszynski, T., & Kurzynski, M. (2011).
.. [#wolo2012] Woloszynski, T., et al. (2012).
.. [#cavalin2013] Cavalin, P. R., et al. (2013).
.. [#cruz2015meta] Cruz, R. M., et al. (2015).
.. [#oliveira2017] Oliveira, D. V., et al. (2017).
.. [#garcia2018] Garcia, S., et al. (2018).
.. [#cruz2019fire] Cruz, R. M., et al. (2019).
.. [#souza2024] Souza, J. V., et al. (2024).
.. [#davtalab2024] Davtalab, R., et al. (2024).
.. [#zhu2025] Zhu, Y., et al. (2025).
.. [#zhang2025] Zhang, Y., et al. (2025).
.. [#barboza2025] Barboza, E., et al. (2025).

You can check: `Awesome Ensemble Learning <https://github.com/yzhao062/awesome-ensemble-learning#gomes2017a>`_ 
