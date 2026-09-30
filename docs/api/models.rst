cpm.models
==========

.. currentmodule:: cpm.models

The components that models are built from. Combine them inside a model
function and pass that to :class:`~cpm.generators.Wrapper`. Their formulas are
in :mod:`cpm.models.kernels`, as functions that the classes compute with and
that loops compiled with numba can call.

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

Kernels for compiled models
---------------------------

.. automodule:: cpm.models.kernels
   :no-members:

.. currentmodule:: cpm.models.kernels

.. autosummary::
   :toctree: generated/
   :nosignatures:

   softmax
   log_softmax
   p_second
   logistic
   irreducible_noise
   softmax_noise
   sigmoid
   greedy
   choice_kernel
   choose
   delta_rule
   separable_rule
   q_learning
   humble_teacher_change
   humble_teacher
   sarsa_trace
   sarsa_trace_update
   sigmoid_activation
   gating_gain
   gate
   competitive_gating
   prospect_utility
   prospect_weight
   weight_tk
   weight_power
   weight_prelec
   weight_gw
   expected_utility
   add_offset
   offset
   rapid_attention_shift
