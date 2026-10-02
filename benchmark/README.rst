Dynamic Ensemble Selection Benchmark
====================================

This repository provides a reproducible Python benchmark for Dynamic Ensemble
Selection (DES). It accompanies the empirical component of our DES survey and
is intended to make the benchmark easy to inspect, reproduce, and extend.

The conceptual and systematic review is the primary contribution of the study;
this benchmark serves as a complementary empirical analysis under a unified
experimental protocol.

Quick Start
-----------

1. Install the required packages::

    pip install -r requirements.txt

2. Check the environment and resolved benchmark configuration::

    python experiments.py --preflight

3. Run the complete benchmark::

    python experiments.py

The default command runs all configured datasets, all benchmark methods, and
the five predefined random seeds.

Example: Run Selected Methods
-----------------------------

Run selected methods on the ``fars`` dataset::

    python experiments.py \
        --datasets fars \
        --methods KNORAU DESFH METADES \
        --seeds 0 42 123 2021 7 \
        --output results

Example: Explicit Full Run
--------------------------

::

    python experiments.py \
        --datasets all \
        --methods all \
        --seeds 0 42 123 2021 7 \
        --k 7 \
        --pool-profile current13 \
        --output results

Useful Commands
---------------

List all available datasets::

    python experiments.py --list-datasets

List all available methods::

    python experiments.py --list-methods

Check dependencies without running experiments::

    python experiments.py --preflight

Stop immediately if any method fails::

    python experiments.py --fail-fast

Repository Structure
--------------------

::

    .
    |-- experiments.py
    |-- requirements.txt
    |
    |-- des_benchmark/
    |   |-- __init__.py
    |   |-- config.py
    |   |-- data.py
    |   |-- models.py
    |   |-- runner.py
    |   |-- results.py
    |   |
    |   `-- methods/
    |       |-- __init__.py
    |       |-- des_as.py
    |       |-- fh_des.py
    |       |-- gnn_des.py
    |       |-- im_def.py
    |       `-- inca_des.py
    |
    `-- results/

Benchmark Protocol
------------------

The benchmark uses the following default protocol:

* **Datasets:** 25 tabular datasets from PMLB.
* **Split:** stratified 60/20/20 train/DSEL/test split.
* **Repeated evaluation:** five independent random seeds:
  ``0, 42, 123, 2021, 7``.
* **Feature scaling:** ``MinMaxScaler`` fitted only on the training partition.
* **Region of Competence:** ``k = 7`` where applicable.
* **Primary predictive metric:** macro F1-score.
* **Training time:** method-level fitting/adaptation time.
* **Inference time:** total prediction time over the complete test partition.
* **Memory:** peak Python allocation during method fitting and prediction,
  measured with ``tracemalloc``.

For DES methods, shared base-classifier pool training is measured separately
from DES fitting/adaptation. Therefore, reported DES training times should be
interpreted as method-level fitting costs rather than end-to-end training costs.

Methods
-------

The benchmark includes classical DES methods from ``DESlib``, recent/custom DES
implementations, and static ensemble baselines.

DESlib methods
~~~~~~~~~~~~~~

* KNOP
* DES-Clustering
* DES-KNN
* KNORA-E
* KNORA-U
* FIRE-DES
* DES-MI
* DES-RRC
* DES-KL
* DES-P
* META-DES

Recent / custom implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* GNN-DES
* FH-DES
* imDEF
* DES-AS
* IncA-DES

Static ensemble baselines
~~~~~~~~~~~~~~~~~~~~~~~~~

* Hard Voting
* Soft Voting
* Stacking

Base Classifier Pool
--------------------

Two pool profiles are available.

``current13``
~~~~~~~~~~~~~

This is the default profile and reproduces the classifier dictionary used in
the current benchmark code.

``full17``
~~~~~~~~~~

This profile additionally includes four Naive Bayes classifiers.

Select a profile with::

    python experiments.py --pool-profile current13

or::

    python experiments.py --pool-profile full17

The pool profile used for a run is recorded automatically in
``results/run_config.json``.

Datasets
--------

The configured benchmark contains 25 PMLB datasets:

* Wine Quality Red
* Wine Quality White
* Led 24
* Led 7
* Yeast
* Ecoli
* Vehicle
* Satimage
* Adult
* Glass
* Wine Recognition
* Collins
* Soybean
* Penguins
* Heart Statlog
* Ionosphere
* Sonar
* Phoneme
* Texture
* Car Evaluation
* Breast Cancer
* DNA
* Shuttle
* Magic
* Fars

Use ``python experiments.py --list-datasets`` to display the exact dataset
names and PMLB identifiers accepted by the command-line interface.

Output Files
------------

Each run writes results to the selected output directory.

::

    results/
    |-- raw_results.csv
    |-- summary.csv
    |-- summary_numeric.csv
    |-- run_config.json
    `-- benchmark.log

``raw_results.csv``
    Seed-level results for every dataset-method run.

``summary.csv``
    Human-readable aggregated benchmark summary.

``summary_numeric.csv``
    Numeric aggregated results for downstream analysis.

``run_config.json``
    Exact benchmark configuration used for the run, including datasets,
    methods, seeds, neighborhood size, pool profile, split protocol, scaling,
    metric, and timing/memory definitions.

``benchmark.log``
    Execution log containing progress information and any method-level errors.

Reproducibility
---------------

The command-line interface is designed so that the benchmark can be reproduced
without executing notebook cells manually. The complete configuration of each
run is saved automatically, and intermediate results are checkpointed during
execution.

For the benchmark configuration used in the manuscript, use the same dataset
set, random seeds, neighborhood size, and base-classifier pool profile reported
with the released results.

Requirements
------------

The main dependencies are:

* NumPy
* pandas
* SciPy
* scikit-learn
* PMLB
* DESlib
* tqdm
* PyTorch

Install all required packages with::

    pip install -r requirements.txt

Command-Line Options
--------------------

Run::

    python experiments.py --help

to see the complete set of available options.

Citation
--------

If you use this benchmark in your research, please cite the accompanying DES
survey paper.

The full citation and BibTeX entry can be added here after publication.

Notes
-----

The benchmark is intended as a transparent and extensible empirical reference,
not as a definitive ranking of DES algorithms. Results should be interpreted
under the reported datasets, classifier pool, preprocessing, split protocol,
and computational environment.
