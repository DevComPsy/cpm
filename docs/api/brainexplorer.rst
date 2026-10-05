cpm.brainexplorer
=================

.. currentmodule:: cpm.brainexplorer

Descriptive statistics and exclusion criteria for the games of
`BrainExplorer <https://brainexplorer.net/>`__. Each class reads the data of
one game, from a pandas DataFrame or a CSV file, computes the metrics
of each participant with ``metrics()``, applies the participant-level
exclusion criteria with ``clean_data()``, and describes every metric in its
codebook, ``get_codebook()``.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   bandits.MilkyWay
   information_gathering.TreasureHunt
   perceptual_decision_making.SpaceObserver
   risky_decision_making.Scavenger
