"""Integrate `torchtraining` with `neptune.ai <https://neptune.ai/>`__ experiment management tool.

.. note::

    **IMPORTANT**: This module is experimental and may not be working
    correctly. Use at your own risk and report any issues you find.

.. note::

    **IMPORTANT**: This module needs `neptune-client` Python package to be available.
    You can install it with `pip install -U torchtraining[neptune]`

Usage is similar to `torchtraining.callbacks.Tensorboard`, except creating `project`
instead of `torch.utils.tensorboard.SummaryWriter`.

Example::

    import torchtraining as tt
    import torchtraining.callbacks.neptune as neptune


    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            # Dummy step
            images, labels = sample
            return images


    project = neptune.project()
    experiment = neptune.experiment(project)

    step = TrainStep(criterion, device)

    # Experiment is optional
    # You have to split `tensor` as only single image can be logged
    # Also you need to cast to `numpy`
    step ** tt.OnSplittedTensor(tt.cast.Numpy() ** neptune.Image(experiment))


"""

import neptune

from .. import _base


def project(project_qualified_name=None, api_token=None, proxies=None, backend=None):
    """Initialize Neptune client library to work with specific project.

    Authorize user, sets value of global variable project to Project object
    that can be used to create or list experiments, notebooks, etc.

    Extensive documentation (explained optional parameters) is located
    `here <https://docs.neptune.ai/neptune-client/docs/neptune.html#neptune.init>`__ .

    Returns
    -------
    neptune.Project
        Object that is used to create or list experiments, notebooks, etc.

    """
    return neptune.init(project_qualified_name, api_token, proxies, backend)


def experiment(
    self,
    project=None,
    name=None,
    description=None,
    params=None,
    properties=None,
    tags=None,
    upload_source_files=None,
    abort_callback=None,
    logger=None,
    upload_stdout=True,
    upload_stderr=True,
    send_hardware_metrics=True,
    run_monitoring_thread=True,
    handle_uncaught_exceptions=True,
    git_info=None,
    hostname=None,
    notebook_id=None,
    notebook_path=None,
):
    """Create and start Neptune experiment.

    Create experiment, set its status to running and append it to the top of the experiments view.
    All parameters are optional.

    Extensive documentation (explained optional parameters) is located
    `here <https://docs.neptune.ai/neptune-client/docs/project.html#neptune.projects.Project.create_experiment>`__

    Returns:
    --------
        `neptune.experiments.Experiment` object that is used to manage experiment
        and log data to it. See `original documentation <https://docsneptune.ai/neptune-client/docs/experiment.html#neptune.experiments.Experiment.log_artifact>`__)

    Raises:
    ------
        `ExperimentValidationError`: When provided arguments are invalid.
        `ExperimentLimitReached`: When experiment limit in the project has been reached.

    """
    if project is None:
        return neptune.create_experiment(
            name,
            description,
            params,
            properties,
            tags,
            upload_source_files,
            abort_callback,
            logger,
            upload_stdout,
            upload_stderr,
            send_hardware_metrics,
            run_monitoring_thread,
            handle_uncaught_exceptions,
            git_info,
            hostname,
            notebook_id,
            notebook_path,
        )
    return project.create_experiment(
        name,
        description,
        params,
        properties,
        tags,
        upload_source_files,
        abort_callback,
        logger,
        upload_stdout,
        upload_stderr,
        send_hardware_metrics,
        run_monitoring_thread,
        handle_uncaught_exceptions,
        git_info,
        hostname,
        notebook_id,
        notebook_path,
    )


class _NeptuneOperation(_base.Operation):
    """Base class for neptune.ai operations.

    Based on `experiment` (and whether it's `None`) calls `neptune.method`
    or `experiment.method` (to be used in `forward`).

    """

    def __init__(self, experiment, method_name):
        super().__init__()
        self.experiment = experiment
        if self.experiment is None:
            self._method = getattr(neptune, method_name)
        else:
            self._method = getattr(self.experiment, method_name)


class Artifact(_NeptuneOperation):
    """Save an artifact (file) in experiment storage.

    Parameters
    ----------
    destination: str, optional
        Destination path to save artifact. If `None` artifact name will be used.
        Default: `None`
    experiment: `neptune.experiments.Experiment`, optional
        Instance of experiment to use. If `None`, global `experiment` will be used.
        Default: `None`


    Returns
    -------
    data
        Data without any modification

    """

    def __init__(self, destination=None, experiment=None):
        super().__init__(experiment, "send_artifact")
        self.destination = destination

    def forward(self, data):
        """
        Arguments
        ---------
        data: str | IO.Object
            A path to the file in local filesystem or IO object.
            It can be open file descriptor or in-memory buffer like io.StringIO or io.BytesIO.
        """
        self._method(data, self.destination)
        return data


class Image(_NeptuneOperation):
    """Log image data in Neptune.

    See `original documentation <https://docs.neptune.ai/_modules/neptune/experiments.html#Experiment.log_image>`__.

    Example::

        import torchtraining as tt
        import torchtraining.callbacks.neptune as neptune


        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                # Dummy step
                images, labels = sample
                # Images is of shape [batch, 1, 28, 28], say MNIST
                return images


        project = neptune.project()
        experiment = neptune.experiment(project)

        step = TrainStep(criterion, device)

        # Experiment is optional
        # You have to split `tensor` as only single image can be logged
        # Also you need to cast to `numpy`
        step ** tt.OnSplittedTensor(tt.cast.Numpy() ** neptune.Image(experiment))

    Parameters
    ----------
    log_name: str
        Name of log (group of images), e.g. "generated images".
    image_name: str, optional
        Name of this specific image received in `data`.
        If `None` consecutive numbers will be used.
        Default: `None`
    description: str, optional
        Textual description of image. If `None` no description.
        Default: `None`
    timestamp: time, optional
        Timestamp to be associated with log entry. Must be Unix time.
        If None is passed, time.time() (Python 3.6 example) is invoked to obtain timestamp.
        Default `None`
    experiment: `neptune.experiments.Experiment`, optional
        Instance of experiment to use. If `None`, global `experiment` will be used.
        Default: `None`

        Returns
    -------
    data
        Data without any modification

    """

    def __init__(
        self,
        log_name: str,
        image_name: str = None,
        description: str = None,
        timestamp=None,
        experiment=None,
    ):
        super().__init__(experiment, "send_image")
        self.log_name = log_name
        self.image_name = image_name
        self.description = description
        self.timestamp = timestamp

    def forward(self, data):
        """
        Arguments
        ---------
        data: PIL image | matplotlib.figure.Figure | str | np.array
            Can be one of:
            * :obj:`PIL image`
              `Pillow docs <https://pillow.readthedocs.io/en/latest/reference/Image.html#image-module>`_
            * :obj:`matplotlib.figure.Figure`
              `Matplotlib 3.1.1 docs <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.figure.Figure.html>`_
            * :obj:`str` - path to image file
            * 2-dimensional :obj:`numpy.array` - interpreted as grayscale image
            * 3-dimensional :obj:`numpy.array` - behavior depends on last dimension

                * if last dimension is 1 - interpreted as grayscale image
                * if last dimension is 3 - interpreted as RGB image
                * if last dimension is 4 - interpreted as RGBA image

            You may need to `transpose` and  transform PyTorch `tensors` to
            fit the above format.

        """
        self._method(
            self.log_name, data, None, self.image_name, self.description, self.timestamp
        )
        return data


class Scalar(_NeptuneOperation):
    """Log scalar data in Neptune.

    Calls `experiment.log_metric` under the hood.
    See `original documentation <https://docs.neptune.ai/_modules/neptune/experiments.html#Experiment.log_metric>`__.

    Example::

        import torchtraining as tt
        import torchtraining.callbacks.neptune as neptune


        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                # Dummy step
                # Calculate loss
                ...
                return loss


        project = neptune.project()
        experiment = neptune.experiment(project)

        step = TrainStep(criterion, device)

        # Experiment is optional
        # You have to split `tensor` as only single image can be logged
        # Also you need to cast to `numpy`
        step ** tt.cast.Item() ** neptune.Image(experiment))

    Parameters
    ----------
    log_name: str
        Name of log (group of images), e.g. "generated images".
    timestamp: time, optional
        Timestamp to be associated with log entry. Must be Unix time.
        If None is passed, time.time() (Python 3.6 example) is invoked to obtain timestamp.
        Default `None`
    experiment: `neptune.experiments.Experiment`, optional
        Instance of experiment to use. If `None`, global `experiment` will be used.
        Default: `None`

    Returns
    -------
    data
        Data without any modification

    """

    def __init__(
        self, log_name: str, timestamp=None, experiment=None,
    ):
        super().__init__(experiment, "send_metric")
        self.log_name = log_name
        self.timestamp = timestamp

    def forward(self, data):
        """
        Arguments
        ---------
        data: double
            Single element Python `double` type

        """
        self._method(self.log_name, data, None, self.timestamp)
        return data


class Text(_NeptuneOperation):
    """Log text data in Neptune.

    See `original documentation <https://docs.neptune.ai/_modules/neptune/experiments.html#Experiment.log_text>`__.

    Example::

        import torchtraining as tt
        import torchtraining.callbacks.neptune as neptune


        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                # You returning text for some reason...
                return loss, text


        project = neptune.project()
        neptune.experiment(project)

        step = TrainStep(criterion, device)

        # Experiment is optional
        # You have to split `tensor` as only single image can be logged
        # Also you need to cast to `numpy`
        step ** tt.Select(text=1) ** neptune.Text())

    Parameters
    ----------
    log_name: str
        Name of log (group of images), e.g. "generated images".
    timestamp: time, optional
        Timestamp to be associated with log entry. Must be Unix time.
        If None is passed, time.time() (Python 3.6 example) is invoked to obtain timestamp.
        Default `None`
    experiment: `neptune.experiments.Experiment`, optional
        Instance of experiment to use. If `None`, global `experiment` will be used.
        Default: `None`

    Returns
    -------
    data
        Data without any modification

    """

    def __init__(
        self, log_name: str, timestamp=None, experiment=None,
    ):
        super().__init__(experiment, "send_text")
        self.log_name = log_name
        self.timestamp = timestamp

    def forward(self, data):
        """
        Arguments
        ---------
        data: str
            Text to be logged
        """
        self._method(self.log_name, data, None, self.timestamp)
        return data


class Reset(_NeptuneOperation):
    """Resets the log.

    Removes all data from the log and enables it to be reused from scratch.
    See `original documentation <https://docs.neptune.ai/_modules/neptune/experiments.html#Experiment.reset_log>`__.

    Example::

        import torchtraining as tt
        import torchtraining.callbacks.neptune as neptune


        class TrainStep(tt.steps.Train):
            def forward(self, module, sample):
                ...
                # You returning text for some reason...
                return loss, text


        project = neptune.project()
        neptune.experiment(project)

        step = TrainStep(criterion, device)

        # Experiment is optional
        # You have to split `tensor` as only single image can be logged
        # Also you need to cast to `numpy`
        step ** tt.Select(text=1) ** neptune.Text())

    Parameters
    ----------
    log_name: str
        Name of log (group of images), e.g. "generated images".
        If `log` does not exist, error `ChannelDoesNotExist` will be raised.

    Returns
    -------
    data
        Anything which was passed into it.

    """

    def __init__(
        self, log_name: str, experiment=None,
    ):
        super().__init__(experiment, "reset_log")
        self.log_name = log_name

    def forward(self, data):
        """
        Arguments
        ---------
        data: Any
            Anything as data will be forwarded
        """
        self._method(self.log_name)
        return data
