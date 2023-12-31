"""
Testing potential interface between rest of NAS/NES algorithm and Anthropic API search space
This test uses chain of thought prompting and has Claude analyse source code for some already trained models, identify
a common coordinate space but not suggest a covariance function
"""

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import dotenv_values
from typing import Iterable, Union, Tuple
import numpy as np
import tinygp
import os


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
        initialise the search space. Must be in form f"{HUMAN_PROMPT} ... {AI_PROMPT}"
        """

        super().__init__(api_key=api_key, *args, **kwargs)

        self.max_tokens = max_tokens
        self.model = model

        self.system_prompt = f"You are a sophisticated, highly intelligent AI whose sole purpose is to be part of a \
                               Neural ensemble search algorithm. This algorithm is using a Gaussian process to \
                               approximate the architecture likelihood surface with respect to some dataset over the \
                               search space, a vector space that can represent every model architecture that could \
                               potentially be used in the ensemble. Your task is to analyse the source code for a set \
                               of model architectures that could be used in the ensemble and find some common \
                               coordinate system between the model architectures and return the coordinates of each \
                               model architecture whose source code was given. Here are some VERY IMPORTANT requirements for \
                               the task: <requirements> <requirement index=0> The common coordinate system should be \
                               as low dimensional as possible while still being mathematically rich enough to \
                               meaningfully cluster similar model architectures closer together and different model \
                               architectures further apart. You can think about this task as if you are performing the \
                               role of an autoencoder, compressing a naive representation of the model architecture \
                               (the source code) into a low dimensional space. How you do this is up to you, and need \
                               not be human interpretable. </requirement> <requirement index=1> The number of axes in\
                               the coordinate system MUST be equal across all models. </requirement> </requirements> \
                               This is your input specification: You will receive source \
                               code inside <models></models> XML tags. Within these tags each model architecture's \
                               source code will be inside <source index=i></source> XML tags where i is an integer \
                               indexing each model uniquely. This is your output specification: You should give the \
                               coordinates in <models></models> XML tags. Within these tags the coordinates for each \
                               model architecture should be given in <coordinate index=j></coordinate> XML tags, where \
                               it is VERY IMPORTANT that the index j of each tag corresponds to the same index i of \
                               the source code for the model architecture this coordinate represents"

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

    def construct_search_space(self, source_code: Iterable[str], task: str, n_train: int)\
            -> Tuple[list[np.ndarray], tinygp.kernels.Kernel]:
        """
        :param source_code: iterable[str] containing source code for all models in the ensemble candidate set
        :param task: str detailing the task eg "image classification"
        :param n_train: number of training samples, guess if unknown "
        :return: tuple containing list of numpy arrays representing coordinates of all models in constructed search
        space and corresponding Gaussian process kernel
        """

        source_code_prompt = "".join([f"<source index={i}> {source} </source>" for i, source in enumerate(source_code)])

        prompt = f"{HUMAN_PROMPT} Please analyse the following source code: <models> {source_code_prompt} </models>\
                   These models will be used for {task} and have been trained on {n_train} samples \
                   After generation, check your results are valid referencing the requirements in the <requirements>\
                   </requirements> XML tags and if they are invalid stop and think then try again. {AI_PROMPT}"

        complete_prompt = "".join((self.starter_prompt, prompt))
        response = self.completions.create(model=self.model, max_tokens_to_sample=self.max_tokens,
                                           prompt=complete_prompt)
        print(response)


if __name__ == "__main__":
    source_code = [open('../mnistEnsembleExample/' + file).read()
                   for file in os.listdir('../mnistEnsembleExample/')
                   if file.endswith(".py") and "Train" not in file]
    task = "image classification"
    n_train = 60000
    search_space = AnthropicSearchSpace(use_default_examples=True)
    search_space.construct_search_space(source_code[:3], task, n_train)

