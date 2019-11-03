API
===
The core two classes needed to work with pandas-ml-utils are subclass of
:class:`.Model` and :class:`.FeaturesAndLabels`. After a model is fitted you get back a
:class:`.Fit` object.

Methods available via pandas DataFrame's
----------------------------------------

The general pattern is:

.. code-block:: python

   import pands as pd
   import pandas_ml_utils as pmu
   from sklearn.neural_network import MLPClassifier
   from hyperopt import hp, fmin

   fit = df.fit_classifier(pmu.SkitModel(MLPClassifier(activation='tanh', random_state=42), fAndL_v1),
                           test_size=0.2,
                           hyper_parameter_space={'learning_rate_init': hp.uniform('learning_rate_init', 0.0001, 0.01),
                                                  'alpha': hp.uniform('alpha', 0.00001, 0.01),
                                                  'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(60,50), (110), (30,30,30)]),
                                                  'early_stopping': True,
                                                  'max_iter': 50,
                                                  '__max_evals': 100})

Where the available arguments are:

.. autofunction:: pandas_ml_utils.fit_classifier

The available arguments are the same for all *fit_\** functions. In general the signature
of all functions is the exact same whether they are classifier, regressors or agents.

.. automodule:: pandas_ml_utils
   :members:


Model
-----
.. autoclass:: pandas_ml_utils.Model
   :members:


FeaturesAndLabels
-----------------
.. autoclass:: pandas_ml_utils.FeaturesAndLabels
   :members:

   .. automethod:: __init__

Fit
---
.. autoclass:: pandas_ml_utils.Fit
   :members:
