"""
Testing potential interface between rest of NAS/NES algorithm and Anthropic API search space
This test uses chain of thought prompting and has Claude analyse source code for some already trained models, identify
a common coordinate space but not suggest a covariance function. It refines the previous example by adopting a step by
step instructional approach
"""

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import dotenv_values
from typing import Iterable, Union, Tuple
import numpy as np
import tinygp
import os
import re
import matplotlib.pyplot as plt
from sandbox.quadrature.tinyQuadGP import QuadGP


class AnthropicSearchSpace(Anthropic):
    """
    Given a set of models and their source code, constructs a search space in the form of a common coordinate system
    along with specific coordinates for each model and suggests the form of a covariance function to be used in a
    Gaussian process modelling the architecture likelihood surface across this search space.
    """

    def __init__(self, api_key: str = dotenv_values()['ANTHROPIC_API_KEY'],
                 max_tokens: int = 10000,
                 model: str = "claude-2.1",
                 examples: Union[Iterable[str], str] = None,
                 use_default_examples: bool = True,
                 *args, **kwargs) -> None:

        f"""
        Constructs search space object according to specifications
        :param api_key: str, access key to Anthropic API. By default, uses value stored under 'ANTHROPIC_API_KEY' from a
         .env file in the active project.
        :param max_tokens: int, maximum number of tokens to sample
        :param model: str, Anthropic LLM model used
        :param examples: Iterable[str], iterable of additional chain of thought prompts the user wishes to use to 
        initialise the search space. Must be in form f"{HUMAN_PROMPT} ... {AI_PROMPT} ... "
        """

        super().__init__(api_key=api_key, *args, **kwargs)

        self.max_tokens = max_tokens
        self.model = model
        self.coords = None

        self.system_prompt = f"You are a sophisticated, highly intelligent AI whose sole purpose is to be part of a \
                               Neural ensemble search algorithm. This algorithm is using a Gaussian process to \
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
                               coordinate in the search space for each feature chosen in task 2 to act as a dimension. \
                               The range of values for each dimension will be 0 to 1, with 0 representing the model in \
                               the candidate set with the lowest value for that particular feature and 1 representing \
                               the model with the highest value for that particular feature, with all other models \
                               being assigned a coordinate within this range. Coordinates should be unique for each \
                               model, although models can have equal coordinate values for individual dimensions. \
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
                               and instead must have values which are continuous between 0 and 1. If this is not the \
                               case restart from task 2. Task 6: Check the produced coordinates are uncorrelated, and \
                               if they are restart from task 2 with different features. "

        source_code_ex1 = [open('../mnistEnsembleExample/' + file).read()
                           for file in os.listdir('../mnistEnsembleExample/')
                           if file.endswith(".py") and "Train" not in file]
        source_code_prompt_ex1 = "".join([f"<source index={i}> {source} </source>"
                                          for i, source in enumerate(source_code_ex1)])

        self.default_examples = f"{HUMAN_PROMPT} I want to generate a latent space representation of this set \
                                  of models: <models>{source_code_prompt_ex1}</models> trained on the MNIST dataset, \
                                  a relatively easy image classification task with a large number of training samples. \
                                  Please generate coordinates for these models in this low dimensional latent space. \
                                  When you are done, check your answers are valid referencing the requirements in the \
                                  <requirements></requirements> XML tags. Talk me through your chain of thought. \
                                  {AI_PROMPT} This task is relatively easy so even simple models should perform okay \
                                  and therefore should not be clustered together. The number of training samples in \
                                  this task is high, so overfitting is unlikely and complex models are likely to do \
                                  better than simpler models and therefore should not be clustered together. This \
                                  suggests model complexity is likely to be a good separator and should be included as \
                                  a dimension in the latent space. \
                                  The source code provided contains models from the CNN, MLP, ViT and boosted trees \
                                  classes. As this is an image classification task and the number of model classes is \
                                  high the ability of each of these models to process images is a good separator and \
                                  should be included as a dimension in the latent space. The number of models is low, \
                                  so these 2 dimensions are sufficiently rich to represent the models. \
                                  Model 1 is a CNN with 2 convolutional layers with larger patches with many channels \
                                  so has very high image processing ability, and has a moderate number of parameters \
                                  so is moderately complex. \
                                  Model 2 is a CNN with 3 convolutional layers but smaller patches, giving good image \
                                  processing ability but not as good as Model 1. The overall parameter number is \
                                  higher than Model 1, giving a slightly higher complexity score. \
                                  Model 3 is a fairly small MLP giving low image processing ability and low complexity. \
                                  Model 4 is a more complex MLP than Model 3, so also has a low image processing \
                                  ability but much higher complexity than Model 3. \
                                  Model 5 is a vision transformer so gets a high image processing power score and has \
                                  a high number of parameters so gets a high complexity. \
                                  Model 6 is an XGBoost model so gets low image processing power. The number of trees \
                                  is unconstrained however giving maximum complexity. \
                                  I therefore suggest a 2 dimensional latent space based on image processing power and \
                                  model complexity with the following coordinates: \
                                  <models><coordinate index=1>[0.9,0.3]</coordinate> \
                                  <coordinate index=2>[0.8,0.4]</coordinate> \
                                  <coordinate index=3>[0.1,0.1]</coordinate> \
                                  <coordinate index=4>[0.1,0.2]</coordinate> \
                                  <coordinate index=5>[0.9,0.9]</coordinate> \
                                  <coordinate index=6>[0.1,0.9]</coordinate></models> \
                                  Referring to the requirements: <requirements> <requirement index=0> The latent space \
                                  is 2 dimensional which is low but can still meaningfully represent these models \
                                  </requirement><requirement index=1> All models have 2 dimensional coordinates \
                                  </requirement></requirements> Hence the results are valid."

        if isinstance(examples, str):
            self.user_examples = examples
        elif examples is not None:
            self.user_examples = "".join(examples)
        else:
            self.user_examples = ""

        if use_default_examples:
            init_prompt = "".join((self.system_prompt, self.default_examples, self.user_examples))
        else:
            init_prompt = "".join((self.system_prompt, self.user_examples))

        if use_default_examples or examples is not None:
            setup = self.completions.create(model=self.model, max_tokens_to_sample=self.max_tokens, prompt=init_prompt)
            self.starter_prompt = init_prompt + setup.completion
        else:
            self.starter_prompt = init_prompt

    def construct_search_space(self, source_code: Iterable[str], task: str, dataset_info: str)\
            -> Tuple[list[np.ndarray], tinygp.kernels.Kernel]:
        """
        :param source_code: iterable[str] containing source code for all models in the ensemble candidate set
        :param task: str detailing the task eg "image classification"
        :param dataset_info: information about the dataset, should include number of training samples "
        :return: tuple containing list of numpy arrays representing coordinates of all models in constructed search
        space and corresponding Gaussian process kernel
        """

        source_code_prompt = "".join([f"<source index={i}> {source} </source>" for i, source in enumerate(source_code)])

        prompt = f"{HUMAN_PROMPT} Please analyse the following source code: <models> {source_code_prompt} </models>\
                   These models will be used for {task}. The dataset information is: {dataset_info}. Then carry out \
                   your tasks detailed above. {AI_PROMPT}"

        complete_prompt = "".join((self.starter_prompt, prompt))
        completion = self.completions.create(model=self.model, max_tokens_to_sample=self.max_tokens,
                                           prompt=complete_prompt)
        response = completion.completion
        print(response)
        coord_strings = re.findall('<coordinate index=[0-9]+>(.+?)</coordinate>', response, flags=re.DOTALL)
        self.coords = np.array([[float(x) for x in re.findall('[0-9|.]+', coord)] for coord in coord_strings])
        return self.coords


if __name__ == "__main__":
    source_code = [open('../mnistEnsembleExample/' + file).read()
                   for file in os.listdir('../mnistEnsembleExample/')
                   if file.endswith(".py") and "Train" not in file]
    task = "image classification"
    dataset_info = "MNIST dataset, 60000 training samples, 10 classes"
    search_space = AnthropicSearchSpace(use_default_examples=False)
    coords = search_space.construct_search_space(source_code, task, dataset_info)
    print(coords)
    for X, Y in zip(coords.T[:-1], coords.T[1:]):
        plt.scatter(X, Y)
        plt.show()
    mean_neg_log_likelihoods = np.array([0.0838, 0.1055, 0.3827, 0.1170, 0.1050, 0.0584])
    n_batches = 25
    likelihoods = np.exp(-n_batches * mean_neg_log_likelihoods)

    integrand = QuadGP(coords, likelihoods, mean='avg')
    lbs = np.zeros(coords.shape[1])
    ubs = np.ones(coords.shape[1])
    integral, variance = integrand.mc_quad(1000, lbs, ubs)
    print(f'Model evidence: {integral}, estimate variance: {variance}')
    print(f'Evidence integral weights: {integrand.quad_weights}')

    ensemble_weights = integrand.quad_weights * likelihoods / integral
    print(f'Ensemble weights: {ensemble_weights}')

    epsilon = 0.8 * np.min(likelihoods)
    z = np.sqrt(2 * (likelihoods - epsilon))

    integrand_wsabi = QuadGP(coords, z)
    weights = integrand.wsabi_quad(epsilon)
    normalised_weights = np.diag(weights) / np.trace(weights)
    integral = likelihoods @ normalised_weights
    print(f'WSABI model evidence: {integral}')

    ensemble_weights = normalised_weights * likelihoods / integral
    print(f'WSABI ensemble weights: {ensemble_weights}')






