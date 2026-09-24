{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

{% set links = tutorial_links.get(fullname, []) %}
{% if links %}
.. seealso::

{% for doc in links %}
   - :doc:`/{{ doc }}`
{% endfor %}
{% endif %}

