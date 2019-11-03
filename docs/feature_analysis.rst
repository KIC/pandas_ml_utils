Analyze the Feature Space
==========================

.. autofunction:: pandas_ml_utils.feature_selection

.. jupyter-execute::
   :hide-code:

   %matplotlib inline
   import sys
   sys.path.append("../")

.. jupyter-execute::

   import pandas_ml_utils as pmu
   import pandas as pd

   df = pd.read_csv('_static/burritos.csv')[["Tortilla", "Temp", "Meat", "Fillings", "Meat:filling", "Uniformity", "Salsa", "Synergy", "Wrap", "overall"]]
   df.feature_selection(label_column="overall")

