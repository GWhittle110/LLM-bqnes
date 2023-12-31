"""
Testing potential interface between rest of NAS/NES algorithm and Anthropic API search space
This test has Claude analyse source code for some already trained models, identify
a common coordinate space and suggest a covariance function.
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
                               model architecture whose source code was given. You should also suggest a covariance \
                               function defined on this coordinate system which most accurately models the behaviour \
                               of the architecture likelihood surface. Here are some VERY IMPORTANT requirements for \
                               the task: <requirements> <requirement index=0> The common coordinate system should be \
                               as low dimensional as possible while still being mathematically rich enough to \
                               meaningfully cluster similar model architectures closer together and different model \
                               architectures further apart. You can think about this task as if you are performing the \
                               role of an autoencoder, compressing a naive representation of the model architecture \
                               (the source code) into a low dimensional space. How you do this is up to you, and need \
                               not be human interpretable. </requirement> <requirement index=1> The number of axes in\
                               the coordinate system MUST be equal across all models. </requirement> \
                               <requirement index=2> You will make no \
                               assumptions about the dataset the model architectures are trained on, only that they \
                               are all trained on some random subset of the same training dataset. In other words, \
                               the common coordinate system and kernel function must be invariant of dataset and \
                               depend only on the source code of the model architectures themselves. </requirement> \
                               <requirement index=3> You are to make no assumptions about how the model architectures \
                               are trained. For example, you cannot assume that they are all trained using the same \
                               loss function, with the same regularization terms or with the same stopping criteria. \
                               </requirement> <requirement index=4> The kernel functions you produce MUST be valid \
                               Mercer kernels. That is, they must be symmetric with respect to both input coordinates \
                               and must be positive definite functions with respect to the search space you define. \
                               </requirement> </requirements> This is your input specification: You will receive source \
                               code inside <models></models> XML tags. Within these tags each model architecture's \
                               source code will be inside <source index=i></source> XML tags where i is an integer \
                               indexing each model uniquely. This is your output specification: You should give the \
                               coordinates in <models></models> XML tags. Within these tags the coordinates for each \
                               model architecture should be given in <coordinate index=j></coordinate> XML tags, where \
                               it is VERY IMPORTANT that the index j of each tag corresponds to the same index i of \
                               the source code for the model architecture this coordinate represents. The kernel \
                               function should be given in <kernel></kernel> XML tags, and should specifically be \
                               given as python code for use with the tinygp library."

        self.default_examples = f""

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

    def construct_search_space(self, source_code: Iterable[str]) -> Tuple[list[np.ndarray], tinygp.kernels.Kernel]:
        """
        :param source_code: iterable[str] containing source code for all models in the ensemble candidate set
        :return: tuple containing list of numpy arrays representing coordinates of all models in constructed search
        space and corresponding Gaussian process kernel
        """

        source_code_prompt = "".join([f"<source index={i}> {source} </source>" for i, source in enumerate(source_code)])

        prompt = f"{HUMAN_PROMPT} Please analyse the following source code: <models> {source_code_prompt} </models>\
                   After generation, check your results are valid referencing the requirements in the <requirements>\
                   </requirements> XML tags and if they are invalid stop and think then try again. {AI_PROMPT}"

        complete_prompt = "".join((self.starter_prompt, prompt))
        response = self.completions.create(model=self.model, max_tokens_to_sample=self.max_tokens,
                                           prompt=complete_prompt)
        print(response)


if __name__ == "__main__":
    source_code = [open('testSourceCode/'+file).read() for file in os.listdir('../../misc/testSourceCode/')]
    search_space = AnthropicSearchSpace(use_default_examples=False)
    search_space.construct_search_space(source_code)

