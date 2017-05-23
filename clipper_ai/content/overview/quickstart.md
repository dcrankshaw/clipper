+++
date = "2017-05-20T18:01:34-07:00"
icon = "<b>1. </b>"
title = "Quickstart"
weight = 1
+++

The easiest way to get started using Clipper is to install the `clipper_admin` pip package and use it interactively from a Python
REPL.

```py
$ pip install clipper_admin
$ python
```

Once you have the `clipper_admin` package installed, you can use it to start a Clipper instance and deploy your
first model.

From the Python REPL:

```py
>>> import clipper_admin.clipper_manager as cm
# Start
>>> clipper = cm.Clipper("localhost")
Checking if Docker is running...
>>> clipper.start()
Clipper is running
>>> clipper.register_application("test_app", "test_model", "doubles", "-1.0", 100000)
Success!
>>> clipper.get_all_apps()
[u'test']
>>> def pred(xs):
...     return [str(np.sum(x)) for x in xs]
>>> clipper.deploy_predict_function("test_model", 1, pred, ["test"], "doubles")
```






<!-- This package contains utilities to start and manage a Clipper instance, as well as to deploy models to Clipper. -->




<!-- __Link to API docs__ -->
