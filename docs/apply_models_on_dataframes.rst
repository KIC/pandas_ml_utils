Apply Models to DataFrames
============================

While all the possible arguments are documented in :doc:`api` the general pattern follwows
along the lines.

.. jupyter-execute::
   :hide-code:

   %matplotlib inline
   import sys
   sys.path.append("../")

.. jupyter-execute::

   import pandas as pd
   import pandas_ml_utils as pmu
   from sklearn.linear_model import LogisticRegression

   df = pd.read_csv('_static/burritos.csv')
   df["with_fires"] = df["Fries"].apply(lambda x: str(x).lower() == "x")
   df["price"] = df["Cost"] * -1
   df = df[["Tortilla", "Temp", "Meat", "Fillings", "Meat:filling", "Uniformity", "Salsa", "Synergy", "Wrap", "overall", "with_fires", "price"]].dropna()
   fit = df.fit_classifier(pmu.SkitModel(LogisticRegression(solver='lbfgs'),
                                         pmu.FeaturesAndLabels(["Tortilla", "Temp", "Meat", "Fillings", "Meat:filling",
                                                                "Uniformity", "Salsa", "Synergy", "Wrap", "overall"],
                                                               ["with_fires"],
                                                               targets=("price", "price"))))

   fit

From here you can save and reuse it like so:

.. jupyter-execute::

   fit.save_model('/tmp/burrito.model')
   df.classify(pmu.Model.load('/tmp/burrito.model')).tail()


This is basically all you need to know. The same patterns are applied to regressors or
agents for reinforcement learning.