.. This is A COPY OF the main index.rst file which is rendered into the landing page of your documentation.
   Follow the inline instructions to configure this for YOUR next project.



Welcome to cfmining documentation !
=========================================================
Cfmining is a tool to explore counterfactual antecedents in in machine learning models.

The source code is available `here <https://github.com/visual-ds/cfmining>`_.

Algorithms
----------

The algorithms are divided into two categories: algorithms: general algorithms designed to any type of classifier (model-agnostic) and mip-algorithms: they were designed to specific classes of algorithms and explore mixed-integeter programming.

.. glossary::
    :doc:`algorithms`
         General algorithms designed to any type of classifier (model-agnostic)

    :doc:`mip_algorithms`
         Algorithms designed to specific classes of algorithms and explore mixed-integeter programming.


Auxiliar/Wrapper classes
------------------------

A counterfactual antecedent algorithm usually needs three other properties to find the actions:

.. glossary::
    :doc:`actions_set`
        Action Set consists on a structure that is capable of organizing the feasible changes on a counterfactual antecedent. It consists on a _ActionElement which is a variable and its possible changes and an ActionSet, that organizes all ActionElements. The discretization of continuous features can be done using a pre-determined grid or by finding the discretization in decision trees.

    :doc:`predictors`
        The predictors are classes that behave like wrappers for sklearn-like classifier (and other structures), they provide the additional information needed by the algorithms.

    :doc:`criteria`
        The critera are classes that evaluate a counterfactual antecedent and calculate its cost.
        
Visualization
-------------

.. glossary::
    :doc:`visualization`
        Promote visalization tools to help understanding the counterfactual antecedents.

Glossary
-------------


.. toctree::
   :maxdepth: 2
   :caption: Algorithms:

   algorithms.rst
   mip_algorithms.rst

.. toctree::
   :maxdepth: 2
   :caption: Processes organization:

   actions_set.rst
   predictors.rst
   criteria.rst

.. toctree::
   :maxdepth: 2
   :caption: Visualization:

   visualization.rst

.. toctree::
   :maxdepth: 2
   :caption: Summary:

   summary.rst

.. Delete this line until the * to generate index for your project: * :ref:`genindex`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


This documentation was last updated on |today|.

.. Finished personalizing all the relevant details? Great! Now make this your main index.rst,
   And run `make clean html` from your documentation folder :)
