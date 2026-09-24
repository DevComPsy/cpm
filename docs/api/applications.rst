cpm.applications
================

.. currentmodule:: cpm.applications

Ready-made models from the literature. Most are subclasses of
:class:`~cpm.generators.Wrapper`, so they can be simulated and fitted like any
model you build yourself.

Reinforcement learning
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   reinforcement_learning.RLRW
   reinforcement_learning.HybridMBMF

Decision making
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   decision_making.PTSM
   decision_making.PTSM1992
   decision_making.PTSM2025

Metacognition
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   signal_detection.EstimatorMetaD
   signal_detection.fit_metad
   signal_detection.metad_nll
