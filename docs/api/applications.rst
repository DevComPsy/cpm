cpm.applications
================

.. currentmodule:: cpm.applications

Ready-made models from the literature. Most are subclasses of
:class:`~cpm.generators.Wrapper`, so they can be simulated and fitted like any
model you build yourself. Each of them has a session version (named with
``Session`` appended), which gives the same results many times faster; see
:doc:`/how-to/fast-session-models`.

Reinforcement learning
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   reinforcement_learning.RLRW
   reinforcement_learning.RLRWSession
   reinforcement_learning.HybridMBMF
   reinforcement_learning.HybridMBMFSession

Decision making
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   decision_making.PTSM
   decision_making.PTSMSession
   decision_making.PTSM1992
   decision_making.PTSM1992Session
   decision_making.PTSM2025
   decision_making.PTSM2025Session

Metacognition
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   signal_detection.EstimatorMetaD
   signal_detection.fit_metad
   signal_detection.metad_nll
