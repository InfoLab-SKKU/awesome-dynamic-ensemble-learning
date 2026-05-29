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

You can check: `Awesome Ensemble Learning <https://github.com/yzhao062/awesome-ensemble-learning#gomes2017a>`_ 
