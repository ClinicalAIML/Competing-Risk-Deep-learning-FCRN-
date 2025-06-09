# Competing-Risk-Deep-learning-FCRN-
Official implementation of FCRN model

We develop two pipelines to model the cause-specific and Fine-Grey models for the competing risks modeling.


## How to run the model
To run the code, you can use `python simple_spec.py` to run the cause-specific model.

Use `python simple_sub.py` to run the Fine-Grey model.

Use `python simple_spec_fda.py` to run the cause-specific model with functional data input.

Use `python simple_sub_fda.py` to run the Fine-Grey model with functional data input.

Use `python simple_spec_missing.py` to run the cause-specific model with missing data.

Use `python simple_sub_missing.py` to run the Fine-Grey model with missing data.


The example data are sved in the `raw_data` file, you can change the path and use your own data.
