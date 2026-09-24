cpm.models
==========

.. currentmodule:: cpm.models

The components that models are built from. Combine them inside a model
function and pass that to :class:`~cpm.generators.Wrapper`.

Learning rules
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   learning.DeltaRule
   learning.SeparableRule
   learning.QLearningRule
   learning.HumbleTeacher
   learning.SARSATrace

Decision rules
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   decision.Softmax
   decision.Sigmoid
   decision.GreedyRule
   decision.ChoiceKernel

Activation functions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   activation.SigmoidActivation
   activation.CompetitiveGating
   activation.ProspectUtility
   activation.Offset

Attention
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   attention.RapidAttentionShift

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   utils.Nominal
