cpm.generators
==============

.. currentmodule:: cpm.generators

The building blocks of every model: parameters and their priors, the
:class:`Wrapper` that runs a model over the trials of one participant (or the
:class:`SessionWrapper` for models that compute all trials at once), and the
:class:`Simulator` that runs it over many.

Parameters
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Parameters
   Value
   LogParameters

Wrappers and simulators
-----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Wrapper
   SessionWrapper
   Simulator
