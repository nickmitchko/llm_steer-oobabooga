"""
An example of extension. It does nothing, but you can add transformations
before the return statements to customize the webui behavior.

Starting from history_modifier and ending in output_modifier, the
functions are declared in the same order that they are called at
generation time.
"""

import lm_eval.models
import lm_eval.tasks
import gradio as gr
import torch
from transformers import LogitsProcessor
from llm_steer import Steer
# Import lm_eval, an evaluator for language models we will optimzie using our steering vectors
import lm_eval
# Import pyswarms, a particle swarming library we can optimize vector positioning for
import numpy as np
import pyswarms as ps

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

params = {
    "display_name": "LLM Steer",
    "is_tab": True,
}

class MyLogits(LogitsProcessor):
    """
    Manipulates the probabilities for the next token before it gets sampled.
    Used in the logits_processor_modifier function below.
    """
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores

def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history

def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    return state

def chat_input_modifier(text, visible_text, state):
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text

def input_modifier(string, state, is_chat=False):
    """
    In default/notebook modes, modifies the whole prompt.

    In chat mode, it is the same as chat_input_modifier but only applied
    to "text", here called "string", and not to "visible_text".
    """
    return string

def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifies the input ids and embeds.
    Used by the multimodal extension to put image embeddings in the prompt.
    Only used by loaders that use the transformers library for sampling.
    """
    return prompt, input_ids, input_embeds

def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    return string

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """

    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result

def custom_css():
    """
    Returns a CSS string that gets appended to the CSS for the webui.
    """
    return ''

def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ''

def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    shared.steered_model = None
    pass

def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """
    with gr.Row():
        with gr.Column():
            layer_idx = gr.Number(label="Layer Index", value=20)
            coeff = gr.Number(label="Coefficient", value=0.4)
            offset = gr.Number(label="Slice Size (0-1)", value=0)
            text = gr.Textbox(label="Steering Text", value="logical")
            add_button = gr.Button("Add Steering Vector")
            add_output = gr.Textbox(label="Add Status")
        with gr.Column():
            reset_button = gr.Button("Reset Steering Vectors")
            get_button = gr.Button("Get Steering Vectors")
            steering_vectors_output = gr.Textbox(label="Steering Vectors")

    def add_steering_vector(layer_idx, coeff, text, offset):
        if shared.steered_model is None:
            shared.steered_model = Steer(shared.model, shared.tokenizer)
        shared.steered_model.add(layer_idx=int(layer_idx), coeff=float(coeff), text=text, try_keep_nr=float(offset))
        shared.model = shared.steered_model.model
        return f"Steering vector added: Layer {layer_idx}, Coefficient {coeff}, Text '{text}'"

    def reset_steering_vectors():
        if shared.steered_model is not None:
            shared.steered_model.reset_all()
            shared.steered_model = None

    def get_steering_vectors():
        if shared.steered_model is not None:
            steering_vectors = shared.steered_model.get_all()
            return str(steering_vectors)
        else:
            return "No steering vectors found."
        
        
    def evaluate_task(tasks: list):
        hf_model = lm_eval.models.huggingface.HFLM(model=shared.model, batch_size=1)
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=hf_model,
            tasks=tasks,
            num_fewshot=0,
        )
        return results
        # ... I don't know what the results dictionary contains....

    def __swarm_fitness(x):
        # TODO: change this to make it the size of the number of particles
        results = np.ndarray(x.shape[0])
        i = 0
        for particle in x:
            steering_vectors = shared.steered_model.get_all()
            # reset steering
            reset_steering_vectors()
            # add steering with parameteres in x[]
            for vector in steering_vectors:
                # Bounds, important to leave the bounding between -1 > 1 for weight
                # weight bounds : [-1,1] 
                # layer bounds  : [0, MAX_LAYER]
                # try_leep_nr   : [0, 1]
                # Scale layers to an integer between 1 and max layer number
                layer_idx = int(particle[0] * (shared.model.config.num_hidden_layers - 1))
                # Here we have the particle coeff, already properly scaled to [0,1] but we need [-1,1]
                coeff = float((2 * particle[1]) - 1)
                # In our layer offset, we are properly scaled 0, 1 and don't need any adjustments
                offset = float(particle[2])
                # Set the steering vectors, keep the original text
                add_steering_vector(layer_idx, coeff, vector.text, offset)            
            # Now let's run the evaluation
            eval = evaluate_task(['medqa'])['results']
            core_metric = eval['medqa']['agg']
            results[i] = core_metric
            i = i + 1
        return results
            
        
        
    # Method to swarm optimize a set of vectors added into the steered model against
    # a known lm_eval benchmark
    def optimize_steering_to_eval():
        if shared.steered_model is not None:
            # steering_vectors = shared.steered_model.get_all()
            
            # Swarm Loop:
            # 1. Add Steering Vectors
            # 2. Loop through the fitness evolution
            # 3. At each step, when the particles are set, for each fitness function
            #    set the steering vectors to the weights of the particles and run lm_eval
            # 4. Reset Steering Vectors
            # 5. After enough iterations, return the optimal vectors
            
            # SWARM OPTIONS -- don't know what this does
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
            # Bounds, important to leave the bounding between -1 > 1 for weight
            # weight bounds : [-1,1] 
            # layer bounds  : [0, MAX_LAYER]
            # try_leep_nr   : [0, 1]
            # bounds = ?
            # https://hf.co/chat/r/mz1tRP0
            NUM_DIMENSIONS = 3
            X_MAX = 1
            X_MIN = 0
            
            x_max = X_MAX * np.ones(NUM_DIMENSIONS)
            x_min = X_MIN * np.ones(NUM_DIMENSIONS)
            
            optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=NUM_DIMENSIONS, options=options, bounds=(x_min, x_max))
            
            cost, pos = optimizer.optimize(__swarm_fitness, iters=5)
            print(cost)
            print(pos)
            return str(pos)
        else:
            return "Please add some steering vectors for optimization"
    

    add_button.click(add_steering_vector, inputs=[layer_idx, coeff, text, offset], outputs=[add_output])
    reset_button.click(reset_steering_vectors)
    get_button.click(get_steering_vectors, outputs=[steering_vectors_output])
    pass
