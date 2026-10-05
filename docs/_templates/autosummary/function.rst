{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autofunction:: {{ objname }}

{% set links = tutorial_links.get(fullname, []) %}
{% if links %}
.. seealso::

{% for doc in links %}
   - :doc:`/{{ doc }}`
{% endfor %}
{% endif %}

