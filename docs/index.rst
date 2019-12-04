.. pandas ml utils documentation master file, created by
   sphinx-quickstart on Sat Nov  2 16:34:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pandas ml utils's documentation!
===========================================
Pandas ML Utils is intended to help you through your journey of applying statistical oder machine learning models
to data while you never need to leave the world of pandas.

I was really sick of converting data frames to numpy arrays back and forth just to try out a
simple logistic regression. So I have started this library where everything you need should be
reachable as a function on the DataFrame.

Install:

.. code-block:: bash

   pip install pandas-ml-utils


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   feature_analysis
   apply_models_on_dataframes
   api


General Concept
===============
.. automodule:: pandas_ml_utils
   :members:
   :noindex:

Check the `component tests <https://github.com/KIC/pandas_ml_utils/blob/master/test/z_component_tests/>`_
for some more concrete examples.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

A note of caution
=================
This is a one man show hobby project in pre-alpha state mainly
serving my own needs. Any help turnng this into a mainstream library is appreciated!
