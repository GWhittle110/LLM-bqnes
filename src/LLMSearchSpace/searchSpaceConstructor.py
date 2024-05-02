"""
Interface with LLM for construction of search space
"""

from anthropic import Anthropic
from dotenv import dotenv_values
import numpy as np
import re
from src.LLMSearchSpace.searchSpace import SearchSpace
import inspect
from torch.utils.data import Dataset
from pandas import DataFrame


class AnthropicSearchSpaceConstructor(Anthropic):
    """
    Given a set of models and their source code, constructs a search space in the form of a common coordinate system
    along with specific coordinates for each model and suggests the form of a covariance function to be used in a
    Gaussian process modelling the architecture likelihood surface across this search space.
    """

    def __init__(self, api_key: str = dotenv_values()['ANTHROPIC_API_KEY'],
                 max_tokens: int = 4096,
                 model: str = "claude-3-opus-20240229",
                 examples: list[dict[str, str]] = None,
                 *args, **kwargs):

        f"""
        Constructs search space object according to specifications
        :param api_key: str, access key to Anthropic API. By default, uses value stored under 'ANTHROPIC_API_KEY' from a
         .env file in the active project.
        :param max_tokens: int, maximum number of tokens to sample
        :param model: str, Anthropic LLM model used
        :param examples: List of examples to inform the LLM. Uses Anthropic Message API format, so must be in form
        [dict("role": "user", "content": "example prompt 1"), dict("role": "assistant"), "content": 
        "example response 1"), ... ]
        """

        super().__init__(api_key=api_key, *args, **kwargs)

        self.max_tokens = max_tokens
        self.model = model
        self.examples = [] if examples is None else examples

        self.system_prompt = f"Your purpose is to be part of a \
                               Neural Ensemble Search algorithm. This algorithm is using a Gaussian process to \
                               approximate the architecture likelihood surface with respect to some dataset over the \
                               search space, a vector space that can represent every model architecture that could \
                               potentially be used in the ensemble. Models will be selected from this search space \
                               sequentially, then their likelihood over the dataset will be calculated  and used to \
                               update the Gaussian process model for the architecture likelihood surface using a \
                               squared exponential kernel. Models that are expected to perform similarly on the given \
                               dataset should therefore be close together and very different models should be further \
                               apart to make this model for the architecture likelihood surface as accurate and \
                               informative as possible. An overview of your task is to analyse the source code for all \
                               the candidate models, suggest a coordinate system for the search space that best \
                               clusters similar models together while separating them from different models, and \
                               assign each candidate model a coordinate in this search space. You will be given model \
                               source code inside <models></models> XML tags, along with information on the specific \
                               machine learning task these models are carrying out and information on the dataset. \
                               Within these tags each model architecture's source code will be inside <source index=i>\
                               </source> XML tags where i is an integer indexing each model uniquely. Your detailed, \
                               step-by-step instructions are as follows: Task 1: Analyse the source code for all of \
                               the candidate models and contextualise them in terms of the dataset and machine \
                               learning task being carried out to understand the specific task. Give a brief overview \
                               of your understanding of the diversity of models present in the candidate set. \
                               Task 2: Using this understanding, identify features that can be used as different \
                               coordinate dimensions for the search space. These features should be fairly abstract, \
                               such as complexity, aspect ratio, spatial processing power, representation power \
                               etc. The number of features chosen should be as low as possible while still efficiently \
                               clustering similar models together. As such, this should increase only as the diversity \
                               of models in the candidate set increases and more dimensions are needed to separate \
                               the different models. The features chosen will also depend heavily on the specific \
                               machine learning task and the dataset itself, as very different models in one way may \
                               perform very similarly on certain datasets. Great care should be taken over this step \
                               and you should stop and think to get this completely right. Task 3: Assign each model a \
                               coordinate in the search space with each feature chosen in task 2 to acting as a dimension. \
                               The range of values for each dimension will be 0 to 1, with 0 representing the model in \
                               the candidate set with the lowest value for that particular feature and 1 representing \
                               the model with the highest value for that particular feature, with all other models \
                               being assigned a coordinate within this range. Coordinates should be UNIQUE for each \
                               model UNLESS two model architectures are mathematically identical, although models can have equal coordinate values for individual dimensions. \
                               It is very important models are assigned coordinates which place similar models close \
                               together and different models further apart, so take your time over this task too and \
                               make sure to get it right. Your output for this task should be a set of coordinates for \
                               each model given in <models></models> XML tags. Within these tags the coordinates for \
                               each model architecture should be given in <coordinate index=j></coordinate> XML tags, \
                               it is VERY IMPORTANT that the index j of each tag corresponds to the same index i of \
                               the source code for the model architecture this coordinate represents. It is also very \
                               important that every model has a value for every coordinate even if 0. Task 4: Analyse \
                               the coordinates you have produced and check that they accurately represent the source \
                               code for each model and meaningfully separate different models while clustering similar \
                               models together. If not, stop and think about why this is. It is likely to be because \
                               the chosen features are inappropriate, the number of features chosen is too low or \
                               both. Once you have thought about why your first attempt failed, restart from task 2 \
                               with this new understanding. Task 5: Check the produced coordinates are all between 0 \
                               and 1, and are continuous not categorical e.g. they cannot only have entries of 0 and 1 \
                               and instead must have values which are continuous between 0 and 1. This is something you \
                               are known to struggle with so pay CLOSE ATTENTION to this. You cannot use features which \
                               indicate whether or not something is true, or binary features. If this is not the \
                               case restart from task 2. Task 6: Check the produced coordinates are uncorrelated, and \
                               if they are restart from task 2 with different features."

    def construct_search_space(self, models: list[callable], task: str, dataset: Dataset,
                               predictions: DataFrame = None, reduction_factor: float = 1, nats_info: dict = None,
                               save_prompt: bool = False, _run=None) -> [SearchSpace, np.ndarray]:
        """
        :param models: list containing model objects
        :param task: str detailing the task eg "image classification"
        :param dataset: Training dataset for models
        :param predictions: Dataframe containing model predictions keyed on model name and targets
        :param reduction_factor: Factor to divide log likelihood values by (shrinks likelihoods towards 1)
        :param nats_info: Dict containing NATS Bench API, dataset name and architecture indexes
        :param save_prompt: Whether to save the prompt and response to the _run.info dict
        :param _run: Sacred run object
        :return: SearchSpace object containing coordinates, models and dataset, and array of discrete dimensions
        """

        if "classification" in task.lower():
            classes = np.unique(dataset.targets)
            dataset_info = f'Number of datapoints: {len(dataset)}, number of output classes: {len(classes)}'
        else:
            dataset_info = f'Number of datapoints: {len(dataset)}, target mean: {dataset.targets.mean().item()}, \
                             target standard deviation: {dataset.targets.std().item()}'

        if nats_info is not None:
            source_code = [str(model) for model in models]
        else:
            source_code = [inspect.getsource(type(model)) for model in models]
        n_models = len(source_code)
        source_code_prompt = "".join(
            [f"<source index={i}> {source} </source>" for i, source in enumerate(source_code)])

        message = {"role": "user", "content": f"Please analyse the following \
        {'printed model architectures' if nats_info is not None else 'source code'}: <models> {source_code_prompt} </models>\
                   These models will be used for {task}. The dataset information is: {dataset_info}. Then carry out \
                   your tasks detailed above."}

        response_message = self.messages.create(max_tokens=self.max_tokens, model=self.model, system=self.system_prompt,
                                                messages=self.examples+[message], temperature=0.)
        response = response_message.content[0].dict()["text"]
        print(response)
        coord_strings = re.findall('<coordinate index=[0-9]+>(.+?)</coordinate>', response, flags=re.DOTALL)
        coords_list = [[float(x) for x in re.findall('[0-9|.]+', coord)] for coord in coord_strings]
        coords = np.array(coords_list[-n_models:])
        discrete_dims = np.array([i for i in range(coords.shape[1]) if set(coords[:, i]).union({0., 1.}) == {0., 1.}])
        if len(discrete_dims) == 0:
            discrete_dims = None

        # Remove dimensions which contain only one value
        mask = np.array([len(set(coords[:, i])) > 1 for i in range(coords.shape[1])])
        coords = coords[:, mask]

        # Squash dimensions which have values outside the interval [0, 1] into this interval
        coords = np.array([coords[:, i] / (max(np.max(coords[:, 0]), 1) - min(np.min(coords[:, 0]), 0))
                           + min(np.min(coords[:, 0]), 0) for i in range(coords.shape[1])]).T

        if _run is not None:
            if save_prompt:
                _run.info["claude_query"] = self.examples+[message]
                _run.info["claude_response"] = response_message
            _run.info["coordinates"] = coords.tolist()
            _run.info["discrete_dims"] = discrete_dims.tolist() if discrete_dims is not None else None
        search_space = SearchSpace(models, coords, dataset, predictions=predictions, reduction_factor=reduction_factor,
                                   nats_info=nats_info, log_offset=-12971/reduction_factor-5)
        return search_space, discrete_dims
