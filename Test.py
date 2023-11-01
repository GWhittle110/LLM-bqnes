""" Testing for gitHub integration """
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import dotenv_values

anthropic = Anthropic(
    api_key=dotenv_values()['ANTHROPIC_API_KEY']
)


setup = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=1000,
    prompt=f"{HUMAN_PROMPT} I want to sample models from an arbitrary latent space that is capable of expressing a \
     wide array of different machine learning models. The training details of each model will not matter at all, only \
     the architecture of the model itself. I want to do this so I can use this as a search space for neural architecture \
     search, the details of which you do not need to be concerned about, nor the details of hyperparameter optimization. \
     I want this latent space to be made up of 3 axes. The first axis should represent model class: coordinates close \
     to 0 here should represent feed forward neural networks, coordinates close to 1 should represent gradient boosted \
     trees, and coordinates close to -1 should represent SVMs. The second axis should represent model depth, with the number of layers or \
     equivalent for other models should increase with an increase in coordinate value. Finally the third dimension should \
     represent model width. Similar models should be clustered close together. Points inbetween should smoothly interpolate between model classes. \
     I want you to guess what model is represented at the following point: (1, 1, 1) and generate me example code for \
     this point with 5 inputs and 1 output. Just give code for the model specification itself, nothing else. {AI_PROMPT}",
)
print(setup.completion)
