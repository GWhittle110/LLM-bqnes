"""
Testing potential interface between rest of NAS/NES algorithm and Anthropic API search space
This test has Claude set up a search space and sample code for models from it.
"""

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import dotenv_values


class AnthropicSearchSpace(Anthropic):
    """
    Responsible for constructing and maintaining access to the underlying latent search space
    """

    def __init__(self, inputs: int = 1,
                 outputs: int = 1,
                 task: str = "classification",
                 max_tokens: int = 1000,
                 model: str = "claude-2",
                 *args, **kwargs):

        """
        Constructs search space object according to specifications
        :param inputs: int, number of input variables
        :param outputs: int, number of output variables
        :param task: str, machine learning task to perform, e.g. regression or classification
        :param max_tokens: int, maximum number of tokens to sample
        :param model: str, Anthropic LLM model used
        """

        super().__init__(api_key=dotenv_values()['ANTHROPIC_API_KEY'], *args, **kwargs)

        self.inputs = inputs
        self.outputs = outputs
        self.task = task
        self.max_tokens = max_tokens
        self.model = model

        init_prompt = f"{HUMAN_PROMPT} I want to define a search space for neural architecture search that is capable \
        of expressing a wide range of different neural network architectures. The training details of each model will  \
        not matter at all, only the architecture of the neural network. The space should have three dimensions. The \
        first dimension should represent network type, with values close to 0 corresponding to feedforward neural \
        networks, values close to 100 corresponding to purely convolutional neural networks, and values close to -100 \
        representing pure transformer encoder networks. Values in this dimension will range from -100 to 100, and the \
        makeup of each model should smoothly interpolate over this range. The second dimension should represent \
        network depth, with values close to 0 having only 1 hidden layer and values close to 100 having 10 hidden \
        layers. Values in this second dimension will range from 0 to 100, and model depth should smoothly interpolate \
        over this range. Finally the third dimension should represent model width, with values close to 0 having at \
        most 1 node per hidden layer and values close to 100 having up to 100 nodes per hidden layer. Values in this \
        third dimension will range from 0 to 100, and maximum network width should interpolate smoothly over this \
        range. All networks will have {inputs} inputs and {outputs} outputs, and will be performing a {task} task. \
        I want you to construct and keep track of this search space so that I can give you three coordinates and you \
        can sample a network from this space at those coordinates. You can assume the architecture is the only \
        thing that matters in this search space, and that the dataset and training will be up to me. You can also \
        assume typical hyperparameters for each network. Regardless of dataset, networks at different points in this \
        search space should have expected output covariance roughly following the RBF kernel used in Gaussian \
        processes. Please describe your understanding of this latent space in detail so that you can consistently \
        and accurately sample from it later. {AI_PROMPT}"

        setup = self.completions.create(model=self.model, max_tokens_to_sample=self.max_tokens, prompt=init_prompt)
        self.starter_prompt = init_prompt + setup.completion

    def query(self, coords: list) -> str:
        """
        Queries search space at specified coordinates
        :param coords: list[float], coordinates of latent search space to query
        :return: str, description of model at coordinates
        """

        query_prompt = self.starter_prompt + f"{HUMAN_PROMPT} Please generate code in pytorch specifying the network \
                                               at the coordinates ({coords[0]}, {coords[1]}, {coords[2]}). {AI_PROMPT}"

        output = self.completions.create(model=self.model, max_tokens_to_sample=self.max_tokens, prompt=query_prompt)
        return output.completion


search_space = AnthropicSearchSpace()
print(search_space.query([0, 10, 10]))
print(search_space.query([0, 10, 10]))
print(search_space.query([0, 10.5, 10.5]))
print(search_space.query([50, 10, 10]))
print(search_space.query([-50, 10, 10]))
