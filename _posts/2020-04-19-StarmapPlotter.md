---
layout: post
title: "Parallel Plotting in Python"
author: "Archis Joglekar"
categories: journal
tags: [matplotlib, multiprocessing, Parallel Computing, Scientific Computing]
image: rc-1.778e-05.png
---

The ability to visualize information convincingly is an essential part of any computational scientists toolkit. There
are plenty of libraries out there, and many are based on `matplotlib`. In my case, I've primarily stuck to the
boilerplate-heavy `matplotlib`. On top of the boilerplate, writing a figure to a file can be quite slow. Plotting 100s
of figures sequentially can become really painful.

For example, you may be plotting resonance curves, like the one near the title of this page, but for a range of
wavenumbers and damping rates. The above curve was generated using the following code

```python
def plot_resonance_curve(wax, actual, predicted, config):
    """
    This function creates a figure, and then adds a 1D plot to that figure
    However, it could be a contourf, a histogram, or any other plot.
    It also closes the figure after it's done to avoid opening too many figures

    :param wax:
    :param actual:
    :param predicted:
    :param config:
    :return:
    """
    fig = plt.figure(figsize=(9, 4))
    rplt = fig.add_subplot(111)
    rplt.plot(wax, actual, label="Actual")
    rplt.plot(wax, predicted, label="CauchyNet")
    rplt.legend()
    rplt.set_xlabel(config["xlabel"], fontsize=16)
    rplt.set_ylabel(config["ylabel"], fontsize=16)
    rplt.set_title(config["title"], fontsize=18)
    rplt.grid()

    fig.savefig(config["filepath"], bbox_inches="tight")
    plt.close(fig)
```

The ability to generate these in parallel becomes particularly important when your code is running on a GPU machine,
and valuable GPU cycles are being wasted because your code is plodding along on a single CPU core. To improve on this,
I've used `multiprocessing` (or `mp`) and it's many calls to parallelize, but I've settled on `starmap` for now.

`starmap` takes in for inputs: a function (`plot_resonance_curve` above) and a list of tuples. Each tuple is comprised
of arguments to the function you want to parallelize.


This list of tuples can be very customized for each application but the following code can be used as an example

```python
def get_plotting_args(training_data, infer_data, folders):
    """
    This function gets all the arguments necessary for the plotting function
    and creates a list of tuples using those arguments.

    The output of this function will be fed to the multiprocessing based process-dispatcher

    :param training_data:
    :param infer_data:
    :param folders:
    :return:
    """
    args_list = []

    for k0, nu in product(
        training_data.coords["wavenumber"].data, training_data.coords["cee"].data
    ):
        k0_dir = os.path.join(folders["plots"], "k0=" + str(round(k0, 2)))
        os.makedirs(k0_dir, exist_ok=True)

        actual = training_data.loc[{"cee": nu, "wavenumber": k0}].data.copy()
        predicted = infer_data.loc[{"cee": nu, "wavenumber": k0}].data.copy()

        nu_str = np.format_float_scientific(nu, unique=False, precision=3)
        config = get_config()
        config["title"] = r"$k_0 = $ " + str(k0) + r", $\nu$ = " + nu_str
        config["filepath"] = os.path.join(k0_dir, "rc-" + nu_str + ".png")

        args_list.append(
            (training_data.coords["frequency"].data.copy(), actual, predicted, config)
        )

    return args_list
```

Finally, the `starmap` call brings this all together in the following function.

```python
def plot_resonance_curves_for_inferences(training_data, infer_data, folders):
    """
    This function is a high level function that creates the arguments for the plotter
    and dispatches the plotter in parallel using starmap

    :param training_data:
    :param infer_data:
    :param folders:
    :return:
    """

    args_list = get_plotting_args(training_data, infer_data, folders)
    cpu_pool = mp.Pool(processes=mp.cpu_count() - 1 or 1)
    cpu_pool.starmap(plot_resonance_curve, args_list)
    cpu_pool.close()
    cpu_pool.join()
```

First, you get a list of all the arguments for each figure. Then, you make a `Pool` of `mp.cpu_count() - 1 or 1`
processes. I typically subtract 1 because it leaves 1 core open for my use, in case this is running locally.

Another thing to keep in mind is that in OSX, you have to set `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` in 
your profile, or at least, before your `python` call. This doesn't have to be done on any versions of linux that I've
found so far, and as you'll see from the link, seems to be an OSX issue [2].

Additional Reading:
1. [Amazingly comprehensive SO post about multiprocessing](https://stackoverflow.com/questions/53751050/python-multiprocessing-understanding-logic-behind-chunksize)
2. [SO post about crashes with multiprocessing <> OSX](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr)