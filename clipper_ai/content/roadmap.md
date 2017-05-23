+++
date = "2017-05-23T08:40:43-07:00"
title = "Status and Roadmap"
icon = "<b>5. </b>"
weight = 5

+++

Clipper is currently under active development and has released an 0.1 alpha version of the system.

### 0.1 Release

The 0.1 release focused on providing a reliable, robust system for serving
predictions for single model applications.

+ First class support for application and model management via a management REST API and accompanying Python client-side management library
+ Robust system implementation to minimize application downtime
+ First class support for serving Scikit-Learn models, Spark ML and Spark MLLib models, and arbitrary Python functions with pre-implemented model containers
+ Extensible metrics library for measuring and reporting system performance metrics


### Beyond 0.1

The priorities of Clipper in the near-term are to improve support for the entire
machine-learning application lifecycle, including the ongoing maintenance and evolution
of deployed machine-learning applications and support for new types of models and specialized
hardware support. Critical features include:

+ First class support for TensorFlow models including Docker support for
running on GPUs.
+ Support for selection policies and multi-model applications including the use of adversarial bandit algorithms
+ Model performance monitoring to detect and prevent application performance degradation over time
+ New scheduler design to leverage the model and resource heterogeneity common to machine learning applications
