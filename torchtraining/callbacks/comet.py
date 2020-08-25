"""Integrate `torchtraining` with `comet.ml <https://www.comet.ml/site/>`__ experiment management tool.

.. note::

    **IMPORTANT**: This module is experimental and may not be working
    correctly. Use at your own risk and report any issues you find.

.. note::

    **IMPORTANT**: This module needs `comet-ml` Python package to be available.
    You can install it with `pip install -U torchtraining[neptune]`

Usage is similar to `torchtraining.callbacks.Tensorboard`, except creating `Experiment`
instead of `torch.utils.tensorboard.SummaryWriter`.

Example::

    import torchtraining as tt
    import torchtraining.callbacks.comet as comet


    class TrainStep(tt.steps.Train):
        def forward(self, module, sample):
            images, labels = sample
            ...
            return loss, images


    project = comet.Experiment()
    step = TrainStep(criterion, device)

    # You have to split `tensor` as only single image can be logged
    step ** tt.Select(1) ** tt.OnSplittedTensor(comet.Image(experiment))
    step ** tt.Select(0) ** tt.cast.Item() ** comet.Scalar(experiment)


"""

import typing

import comet_ml

from .. import _base

Experiment = comet_ml.Experiment
ExistingExperiment = comet_ml.ExistingExperiment
OfflineExperiment = comet_ml.OfflineExperiment


class Clean(_base.Operation):
    """Clean experiment loggers

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment

    Returns
    -------
    Any
        Data passed initially to the operation.

    """

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def forward(self, data):
        """
        Arguments
        ---------
        data: Any
            Anything, will be forwarded
        """
        self.experiment.clean()
        return data


class Asset(_base.Operation):
    """Logs the Asset passed during call.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    name: str, optional
        A custom file name to be displayed. If not provided the `filename`
        from the file_data argument will be used.
    overwrite: bool, optional
        If True will overwrite all existing assets with the same name.
        Default: `False`
    copy_to_tmp: bool, optional
        If file_data is a file-like object, then this flag determines if the file is
        first copied to a temporary file before upload. If `copy_to_tmp`
        is False, then it is sent directly to the cloud.
        Default: `True`
    step: int, optional
        Used to associate the asset to a specific step. Default: `None`
    metadata: typing.Dict, optional
        Optional. Some additional data to attach to the the asset data.
        Must be a JSON-encodable dict. Default: `None`

    Returns
    -------
    str | File-like
        Data passed initially to the operation.

    """

    def __init__(
        self,
        experiment,
        name=None,
        overwrite=False,
        copy_to_tmp=True,
        step=None,
        metadata=None,
    ):
        super().__init__()
        self.experiment = experiment
        self.name = name
        self.overwrite = overwrite
        self.copy_to_tmp = copy_to_tmp
        self.step = step
        self.metadata = metadata

    def forward(self, data):
        """
        Arguments
        ---------
        data: str | File-like
            Either the file path of the file you want to log, or a file-like asset.
        """
        self.experiment.log_asset(
            data, self.name, self.overwrite, self.copy_to_tmp, self.step, self.metadata,
        )
        return data


class AssetData(_base.Operation):
    """Log given data (`str`, `binary` or `JSON`).

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    name: str, optional
        A custom file name to be displayed. If not provided the `filename`
        from the file_data argument will be used.
    overwrite: bool, optional
        If True will overwrite all existing assets with the same name.
        Default: `False`
    step: int, optional
        Used to associate the asset to a specific step. Default: `None`
    metadata: typing.Dict, optional
        Optional. Some additional data to attach to the the asset data.
        Must be a JSON-encodable dict. Default: `None`


    Returns
    -------
    str | File-like
        Data passed initially to the operation.

    """

    def __init__(
        self, experiment, name=None, overwrite=False, step=None, metadata=None,
    ):
        super().__init__()
        self.experiment = experiment
        self.name = name
        self.overwrite = overwrite
        self.step = step
        self.metadata = metadata

    def forward(self, data):
        """
        Arguments
        ---------
        data: str | JSON
            Data to log
        """
        self.experiment.log_asset_data(
            data, self.name, self.overwrite, self.step, self.metadata,
        )
        return data


class AssetFolder(_base.Operation):
    """Logs all the files located in the given folder as assets.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    step: int, optional
        Used to associate the asset to a specific step. Default: `None`
    log_file_name: bool, optional
        If True, log the file path with each file. Default: `False`
    recursive: bool, optional
        If `true` recurse folder and save file names.
        Default: `False`

    Returns
    -------
    folder: str
        Data passed initially to the operation.

    """

    def __init__(
        self, experiment, step=None, log_file_name=False, recursive=False,
    ):
        super().__init__()
        self.experiment = experiment
        self.step = step
        self.log_file_name = log_file_name
        self.recursive = recursive

    def forward(self, data):
        """
        Arguments
        ---------
        folder: str
            Path to the folder to be logged.

        """
        self.experiment.log_asset_folder(
            data, self.step, self.log_file_name, self.recursive
        )
        return data


class Audio(_base.Operation):
    """Logs the audio Asset determined by audio data.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    sample_rate: int, optional
        The sampling rate given to scipy.io.wavfile.write for creating the wav file.
    name: str, optional
        A custom file name to be displayed. If not provided the `filename`
        from the file_data argument will be used.
    overwrite: bool, optional
        If True will overwrite all existing assets with the same name.
        Default: `False`
    copy_to_tmp: bool, optional
        If file_data is a file-like object, then this flag determines if the file is
        first copied to a temporary file before upload. If `copy_to_tmp`
        is False, then it is sent directly to the cloud.
        Default: `True`
    step: int, optional
        Used to associate the asset to a specific step. Default: `None`
    metadata: typing.Dict, optional
        Optional. Some additional data to attach to the the asset data.
        Must be a JSON-encodable dict. Default: `None`

    Returns
    -------
    data: str | np.array
        Data passed initially to the operation.

    """

    def __init__(
        self,
        experiment,
        sample_rate=None,
        name=None,
        overwrite=False,
        copy_to_tmp=True,
        step=None,
        metadata=None,
    ):
        super().__init__()
        self.experiment = experiment
        self.sample_rate = sample_rate
        self.name = name
        self.overwrite = overwrite
        self.copy_to_tmp = copy_to_tmp
        self.step = step
        self.metadata = metadata

    def forward(self, data):
        """
        Arguments
        ---------
        data: str | np.array
            Either the file path of the file you want to log, or a numpy array given to
            `scipy.io.wavfile.write` for wav conversion.

        """
        self.experiment.log_audio(
            data,
            self.sample_rate,
            self.name,
            self.metadata,
            self.overwrite,
            self.copy_to_tmp,
            self.step,
        )
        return data


class ConfusionMatrix(_base.Operation):
    """Logs confusion matrix.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    title: str, optional
        A custom name to be displayed. By default, it is "Confusion Matrix".
    row_label: str, optional
        Label for rows. By default, it is "Actual Category".
    column_label: str, optional
        Label for columns. By default, it is "Predicted Category".
    max_example_per_cell: int, optional
        Maximum number of examples per cell. By default, it is 25.
    max_categories: int, optional
        Max number of columns and rows to use. By default, it is 25.
    winner_function: Callable(List) -> List, optional
        A function that takes in an entire list of rows of patterns,
        and returns the winning category for each row. By default, it is argmax.
    index_to_example_function: Callable, optional
        A function that takes an index and returns either a number, a string, a URL, or a
        {"sample": str, "assetId": str} dictionary.
        See below for more info.
        By default, the function returns a number representing the index of the example.
    cache: bool, optional
        Should the results of index_to_example_function be cached and reused?
        By default, cache is `True`.
    selected: List, optional
        None, or list of selected category indices.
        These are the rows/columns that will be shown. By default, select is None.
        If the number of categories is greater than max_categories, and selected is not provided,
        then selected will be computed automatically by selecting the most confused categories.
    kwargs: optional
        any extra keywords and their values will be passed onto the index_to_example_function.

    Returns
    -------
    data: List[List[double]]
        Data passed initially to the operation.

    """

    def __init__(
        self,
        experiment,
        title="Confusion Matrix",
        row_label="Actual Category",
        column_label="Predicted Category",
        max_examples_per_cell=25,
        max_categories=25,
        winner_function=None,
        index_to_example_function=None,
        cache=True,
        file_name="confusion-matrix.json",
        overwrite=False,
        step=None,
        **kwargs
    ):
        super().__init__()
        self.experiment = experiment
        self.title = title
        self.row_label = row_label
        self.column_label = column_label
        self.max_examples_per_cell = max_examples_per_cell
        self.max_categories = max_categories
        self.winner_function = winner_function
        self.index_to_example_function = index_to_example_function
        self.cache = cache
        self.file_name = file_name
        self.overwrite = overwrite
        self.step = step
        self.kwargs = kwargs

    def forward(self, data):
        """
        Arguments
        ---------
        data: List[List[double]]
            Matrix-like list contianing `confusion` matrix.

        """
        self.experiment.log_confusion_matrix(
            None,
            None,
            data,
            None,
            self.title,
            self.row_label,
            self.column_label,
            self.max_examples_per_cell,
            self.max_categories,
            self.winner_function,
            self.index_to_example_function,
            self.cache,
            self.file_name,
            self.overwrite,
            self.step,
            **self.kwargs,
        )
        return data


class Curve(_base.Operation):
    """Log timeseries data.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    name: str
        Name of data
    overwrite: bool, optional
        If True overwrite previous log. Default: `False`
    step: int, optional
        Step value. Default: `None`

    Returns
    -------
    data: Tuple(List[Number], List[Number])
        Data passed initially to operation

    """

    def __init__(
        self, experiment, name=None, overwrite=False, step=None,
    ):
        super().__init__()
        self.experiment = experiment
        self.name = name
        self.overwrite = overwrite
        self.step = step

    def forward(self, data):
        """
        Arguments
        ---------
        data: Tuple(List[Number], List[Number])
            Either the file path of the file you want to log, or a numpy array given to
            `scipy.io.wavfile.write` for wav conversion.
        """
        self.experiment.log_curve(
            self.name, *data, self.overwrite, self.step,
        )
        return data


class Embedding(_base.Operation):
    """Log a multi-dimensional dataset and metadata for viewing with Comet's Embedding Projector.

    This feature is currently deemed experimental.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    image_data: List[Array] | Images, optional
        List of arrays or Images
    image_size: int, optional (required if image_data is given)
        The size of each image
    image_preprocess_function: Callable, optional
        If image_data is an array, apply this function to each element first
    image_transparent_color: Tuple, optional
        (red, green, blue) tuple
    image_background_color_function: Callable(int) -> Tuple(red, green, blue), optional
        A function that takes an index, and returns a (red, green, blue) color tuple
    title: str, optional
        Name of tensor
    template_filename: str, optional
        name of template JSON file

    Returns
    -------
    data: Tuple(torch.Tensor, torch.Tensor)
        Tensors to visualize in 3D and labels for each tensor.

    """

    def __init__(
        self,
        experiment,
        image_data=None,
        image_size=None,
        image_preprocess_function=None,
        image_transparent_color=None,
        image_background_color_function=None,
        title="Comet Embedding",
        template_filename="template_projector_config.json",
        group=None,
    ):

        super().__init__()
        self.experiment = experiment

        self.image_data = image_data
        self.image_size = image_size
        self.image_preprocess_function = image_preprocess_function
        self.image_transparent_color = image_transparent_color
        self.image_background_color_function = image_background_color_function
        self.title = title
        self.template_filename = template_filename
        self.group = group

    def forward(self, data):
        """
        Arguments
        ---------
        data: Tuple(torch.Tensor, torch.Tensor)
            Tensors to visualize in 3D and labels for each tensor.
        """
        self.experiment.log_embedding(
            *data,
            self.image_data,
            self.image_size,
            self.image_preprocess_function,
            self.image_transparent_color,
            self.image_background_color_function,
            self.title,
            self.template_filename,
            self.group,
        )
        return data


class Figure(_base.Operation):
    """Logs the global Pyplot figure or the passed one and upload its svg version to the backend.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    figure_name: str, optional
        Name of the figure
    overwrite: bool, optional
        If another figure with the same name exists, it will be overwritten if overwrite is set to True.
        Default: `False`
    step: int, optional
        Used to associate figure to a specific step.

    Returns
    -------
    data: figure, optional.
        Data passed initially to operation.

    """

    def __init__(self, experiment, figure_name=None, overwrite=False, step=None):

        super().__init__()
        self.experiment = experiment

        self.figure_name = figure_name
        self.overwrite = overwrite
        self.step = step

    def forward(self, data):
        """
        Arguments
        ---------
        data: figure, optional.
            The figure you want to log. If `None` passed,
            the global pyplot figure will be logged and uploaded
        """
        self.experiment.log_figure(
            self.figure_name, data, self.overwrite, self.step,
        )
        return data


class Histogram3d(_base.Operation):
    """Logs a histogram of values for a 3D chart as an asset for this experiment.

    Calling this method multiple times with the same name and incremented steps
    will add additional histograms to the 3D chart on Comet.ml.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    name: str, optional
        Name of summary
    step: int, optional.
        Used as the Z axis when plotting on Comet.ml.
    **kwargs:
        Additional keyword arguments for histogram.

    Returns
    -------
    data: List | Tuple | Array | Histogram object
        Summarization of histogram (passed to `forward`).

    """

    def __init__(self, experiment, name=None, step=None, **kwargs):
        super().__init__()
        self.experiment = experiment

        self.name = name
        self.step = step
        self.kwargs = kwargs

    def forward(self, data):
        """
        Arguments
        ---------
        data: List | Tuple | Array | Histogram object
            Summarization of histogram
        """
        self.experiment.log_histogram_3d(data, self.name, self.step, **self.kwargs)
        return data


class Image(_base.Operation):
    """Logs the image. Images are displayed on the Graphics tab on Comet.ml.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    name: str, optional
        A custom name to be displayed on the dashboard.
        If not provided the filename from the image_data argument will be used if it is a path.
    overwrite: bool, optional
        If another image with the same name exists, it will be overwritten if overwrite is set to True.
    image_format: str, optional
        Default: 'png'. If the image_data is actually something that can be turned
        into an image, this is the format used. Typical values include 'png' and 'jpg'.
    image_scale: float, optional
        Default: 1.0. If the image_data is actually something that can be turned
        into an image, this will be the new scale of the image.
    image_shape: Tuple, optional
        Default: None. If the image_data is actually something that can be
        turned into an image, this is the new shape of the array. Dimensions are (width, height).
    image_colormap: str, optional
        If the image_data is actually something that can be turned into an image,
        this is the colormap used to colorize the matrix.
    image_minmax: (Number, Number), optional
        If the image_data is actually something that can be turned into an image,
        this is the (min, max) used to scale the values.
        Otherwise, the image is autoscaled between (array.min, array.max).
    image_channels: str, optional. Default 'last'.
        If the image_data is actually something that can be turned into an image,
        this is the setting that indicates where the color information is in the format of the 2D data.
        'last' indicates that the data is in (rows, columns, channels)
        where 'first' indicates (channels, rows, columns).
    copy_to_tmp: bool, optional
        If image_data is not a file path, then this flag determines if the image
        is first copied to a temporary file before upload.
        If copy_to_tmp is False, then it is sent directly to the cloud.
        Default: `True`
    step: int, optional
        Used to associate the audio asset to a specific step. Default: `None`

    Returns
    -------
    data: Multiple objects
        See `forward` for possibilities

    """

    def __init__(
        self,
        experiment,
        name=None,
        overwrite=False,
        image_format="png",
        image_scale=1.0,
        image_shape=None,
        image_colormap=None,
        image_minmax=None,
        image_channels="last",
        copy_to_tmp=True,
        step=None,
    ):
        super().__init__()

        self.experiment = experiment

        self.name = name
        self.overwrite = overwrite
        self.image_format = image_format
        self.image_scale = image_scale
        self.image_shape = image_shape
        self.image_colormap = image_colormap
        self.image_minmax = image_minmax
        self.image_channels = image_channels
        self.copy_to_tmp = copy_to_tmp
        self.step = step

    def forward(self, data):
        """
        Arguments
        ---------
        data: Multiple objects
            One of:
                * a path (string) to an image
                * a file-like object containing an image
                * a numpy matrix
                * a TensorFlow tensor
                * a PyTorch tensor
                * a list or tuple of values
                * a PIL Image

        """
        self.experiment.log_image(
            data,
            self.name,
            self.overwrite,
            self.image_format,
            self.image_scale,
            self.image_shape,
            self.image_colormap,
            self.image_minmax,
            self.image_channels,
            self.copy_to_tmp,
            self.step,
        )
        return data


class Scalar(_base.Operation):
    """Logs scalar value under specified name.

    Usually used to log metric values (like `accuracy`)

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    name: str
        Name of scalar / metric.
    step: int, optional
        Used as the X axis when plotting on comet.ml
    epoch: int, optional
        Used as the X axis when plotting on comet.ml
    include_context: bool, optional
        If set to True (the default), the current context will be logged along
        the metric.

    Returns
    -------
    data: Number
        Value passed to `forward`

    """

    def __init__(self, experiment, name, step=None, epoch=None, include_context=True):
        super().__init__()

        self.experiment = experiment

        self.name = name
        self.step = step
        self.epoch = epoch
        self.include_context = include_context

    def forward(self, data):
        """
        Arguments
        ---------
        data: Number
            Value to log
        """
        self.experiment.log_metric(
            self.name, data, self.step, self.epoch, self.include_context
        )
        return data


# log_metrics
class Scalars(_base.Operation):
    """Logs dictionary of scalars / metrics.

    Usually used to log metric values (like `accuracy`)

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    prefix: str, optional
        Name of prefix used when logging into comet.ml
    step: int, optional
        Used as the X axis when plotting on comet.ml
    epoch: int, optional
        Used as the X axis when plotting on comet.ml

    Returns
    -------
    data: Dict
        Dictionary passed to `forward`

    """

    def __init__(self, experiment, prefix=None, step=None, epoch=None):
        super().__init__()

        self.experiment = experiment

        self.prefix = prefix
        self.step = step
        self.epoch = epoch

    def forward(self, data):
        """
        Arguments
        ---------
        data: Dict
            Dictionary with values to log
        """
        self.experiment.log_metrics(data, self.prefix, self.step, self.epoch)
        return data


class Other(_base.Operation):
    """Reports a key and value to the Other tab on Comet.ml.

    Useful for reporting datasets attributes, datasets path, unique identifiers etc.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment

    Returns
    -------
    data: Tuple[Any, Any]
        Tuple with `key` and `value`

    """

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def forward(self, data):
        """
        Arguments
        ---------
        data: Tuple[Any, Any]
            Tuple with `key` and `value`
        """
        self.experiment.log_other(*data)
        return data


class Others(_base.Operation):
    """Reports dictionary of key/values to the Other tab on Comet.ml.

    Useful for reporting datasets attributes, datasets path, unique identifiers etc.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment

    Returns
    -------
    data: Dict[Any, Any]
        Dict with any `keys` and `values`

    """

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def forward(self, data):
        """
        Arguments
        ---------
        data: Dict[Any, Any]
            Dict with any `keys` and `values`
        """
        self.experiment.log_others(data)
        return data


class Table(_base.Operation):
    """Logs tabular data.

    These strings appear on the Text Tab in the Comet UI.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    filename: str
        Filename ending in ".csv", or ".tsv"
    headers: bool | List
        If True, will add column headers automatically if tabular_data is given;
        if False, no headers will be added; if list then it will be used as headers.

    Returns
    -------
    data: tensor | List[List]
        Data received in `forward`

    """

    def __init__(
        self,
        experiment,
        filename: str,
        headers: typing.Union[bool, typing.List] = False,
    ):
        super().__init__()
        self.experiment = experiment

        self.filename = filename
        self.header = headers

    def forward(self, data):
        """
        Arguments
        ---------
        data: tensor | List[List]
            Data that can be interpreted as 2D tabular data.

        """
        self.experiment.log_table(self.filename, data, self.header)
        return data


class Text(_base.Operation):
    """Logs the text.

    These strings appear on the Text Tab in the Comet UI.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    step: int, optional
        Used to associate text to a specific step
    metadata: JSON-like, optional
        Additional data attached to text.

    Returns
    -------
    data: str
        Received text

    """

    def __init__(self, experiment, step: int = None, metadata=None):
        super().__init__()
        self.experiment = experiment

        self.step = step
        self.metadata = metadata

    def forward(self, data):
        """
        Arguments
        ---------
        data: str
            Text to be stored

        """
        self.experiment.log_text(data, self.step, self.metadata)
        return data


class Notification(_base.Operation):
    """Send yourself a notification through email when an experiment ends.

    Parameters
    ----------
    experiment: Experiment
        Object representing single experiment
    title: str
        Subject of the email
    status: str
        Final status of the experiment.
        Typically, something like "finished", "completed" or "aborted".
    additional_data: dict
        Dictionary of key/values to notify.

    Returns
    -------
    data: Any
        Anything passed to forward

    """

    def __init__(self, experiment, title, status=None, additional_data=None):
        super().__init__()
        self.experiment = experiment

        self.title = title
        self.status = status
        self.additional_data = additional_data

    def forward(self, data):
        """
        Arguments
        ---------
        data: Any
            Anything as it's not send to the function, just passes through

        """
        self.experiment.log_text(self.title, self.status, self.additional_data)
        return data
