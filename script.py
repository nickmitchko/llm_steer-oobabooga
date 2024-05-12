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
    
    task_manager = None
    # TODO: Add in a setting for the batch size
    # TODO: Add in a setting and handling for the other parameters
    # TODO: Add in a setting for the definion of the forward functions
    with gr.Row():
        with gr.Column():
            # with gr.Row():
            layer_idx = gr.Number(label="Layer Index", value=20)
            coeff = gr.Number(label="Coefficient", value=0.4)
            offset = gr.Number(label="Slice Size (0-1)", value=0)
            text = gr.Textbox(label="Steering Text", value="logical")
            add_button = gr.Button("Add Steering Vector")
            # with gr.Row():
            add_output = gr.Textbox(label="Add Status")
        with gr.Column():
            # with gr.Row():
            optimize_button = gr.Button("Optimize Vectors")
            optimize_particles = gr.Slider(label="Number of Particles", value=3, min=1, max=10, step=1)
            optimize_iterations = gr.Slider(label="Number of Iterations", value=5, min=1, max=10, step=1)
            # with gr.Row():
            reset_button = gr.Button("Reset Steering Vectors")
            run_benchmark = gr.Button("Run Benchmark")
            benchmark_name = gr.Textbox(label="Benchmark Name")
            # TODO: make benchmarks runnable via single button
            # TODO: make benchmarks selectable via dropdown
            # TODO: make output of benchmarks output in a nice table
            get_button = gr.Button("Get Steering Vectors")
            steering_vectors_output = gr.Textbox(label="Steering Vectors")
    # with gr.Row():
    #     with gr.Column():
    #         gr.Label("Steering Vectors")
            

    def add_steering_vector(layer_idx, coeff, text, offset):
        if shared.steered_model is None:
            shared.steered_model = Steer(shared.model, shared.tokenizer)
        shared.steered_model.add(layer_idx=int(layer_idx), coeff=float(coeff), text=text, try_keep_nr=int(offset))
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
        hf_model = lm_eval.models.huggingface.HFLM(pretrained=shared.model, batch_size=2)
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=hf_model,
            tasks=tasks,
            num_fewshot=0,
            task_manager=task_manager
        )
        return results
        # ... I don't know what the results dictionary contains....
        
        
    def __scale_layeridx(value):
        # Scale layers to an integer between 1 and max layer number
        layer_idx = int(value * (shared.model.config.num_hidden_layers - 1))
        return layer_idx
    
    def __scale_coeff(value):
        # Here we have the particle coeff, already properly scaled to [0,1] but we need [-1,1]
        coeff = float((2 * value) - 1)
        return coeff
    
    def __scale_particle(particle):
        scaled_particle = particle
        for i in range(0, particle.shape[0], 3):
            scaled_particle[0] = __scale_layeridx(particle[0 + n])
            scaled_particle[1] = __scale_coeff(particle[1 + n])
            scaled_particle[2] = float(particle[2 + n])
        return scaled_particle

    def __swarm_fitness(x):
        task_manager = lm_eval.tasks.TaskManager(include_path="/media/nmitchko/NVME/text-generation-webui/venv/lib/python3.11/site-packages/lm_eval/tasks/medqa/")
        # TODO: change this to make it the size of the number of particles
        results = np.ndarray(x.shape[0])
        i = 0
        print(x)
        # bad logic, we are setting the same coefficents for all particles...
        for particle in x:
            steering_vectors = shared.steered_model.get_all()
            print(steering_vectors)
            # reset steering
            reset_steering_vectors()
            # add steering with parameteres in x[]
            n = 0
            for vector in steering_vectors:
                # Bounds, important to leave the bounding between -1 > 1 for weight
                # weight bounds : [-1,1] 
                # layer bounds  : [0, MAX_LAYER]
                # try_leep_nr   : [0, 1]
                # Scale layers to an integer between 1 and max layer number
                layer_idx_inner = __scale_layeridx(particle[0 + n * 3])
                # Here we have the particle coeff, already properly scaled to [0,1] but we need [-1,1]
                coeff_inner = __scale_coeff(particle[1 + n * 3])
                # In our layer offset, we are properly scaled 0, 1 and don't need any adjustments
                # offset_inner = float(particle[2 + n * 3])
                offset_inner = int(0)
                # Set the steering vectors, keep the original text
                # print(vector)
                # print(offset)
                # print(coeff)
                # print(layer_idx)
                add_steering_vector(layer_idx_inner, coeff_inner, vector['text'], offset_inner)        
                n = n + 1    
            # Now let's run the evaluation
            # TODO: Let this be a user parameter and evaluation metric chooser
            # TODO: Add progress bar tracking https://www.gradio.app/guides/key-features#progress-bars
            evaluation = evaluate_task(['pubmedqa'])
            # print(evaluation['results']['pubmedqa'])
            core_metric = evaluation['results']['pubmedqa']
            # import json
            # with open('results.json', 'w') as outfile:
            #     json.dump(core_metric, outfile)
            # Since this is our cost function we need it to show the error (1 - score) since score is 0-1 ( ie, .93 would have an error of 0.07)
            results[i] = (1 - core_metric['acc,none'])
            i = i + 1
        return results
            
        
        
    # Method to swarm optimize a set of vectors added into the steered model against
    # a known lm_eval benchmark
    def optimize_steering_to_eval(optimize_particles=3, optimize_iterations=5, progress=gr.Progress(track_tqdm=True)):
        if shared.steered_model is not None:
            steering_vectors_num = len(shared.steered_model.get_all())
            
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
            NUM_PARTICLES = optimize_particles
            NUM_DIMENSIONS = 3 * steering_vectors_num
            X_MAX = 1
            X_MIN = 0
            
            x_max = X_MAX * np.ones(NUM_DIMENSIONS)
            x_min = X_MIN * np.ones(NUM_DIMENSIONS)
            
            # Edit this to add a number of dimensions that equals steering_vectors_num * 3
            # We need this because each particle should be multi-dimensional based on the number of steering vectors...
            # So the optimization gets huge when you add lots of steering vectors
            # and each particle triggers an LM_EVAL
            # so that makes this process very long.
            # and possibly even longer than just running a finetune or ablation
            # But maybe there is a better was to do this
            # Think.....
            # Maybe limit the number of particles to 3 and that gives us 9 dimensions? and then 9*5 lm_evals? Wow...
            # lots to think about, maybe pick a smaller set of questions?
            # Or let the lm_eval have a limit?
            optimizer = ps.single.GlobalBestPSO(n_particles=NUM_PARTICLES, dimensions=NUM_DIMENSIONS, options=options, bounds=(x_min, x_max))
            
            cost, pos = optimizer.optimize(__swarm_fitness, iters=optimize_iterations)
            # print out the scaled particle
            scaled_particle = str(__scale_particle(pos))
            # Build explanation of optimization
            steering_vectors = shared.steered_model.get_all()
            # print(steering_vectors)
            # reset steering
            reset_steering_vectors()
            # add steering with parameteres in x[]
            particle_explanation = f"Benchmark Optimum Found: {1 - cost} \n"
            n = 0
            for vector in steering_vectors:
                layer = scaled_particle[n * 3 + 0]
                coeff = scaled_particle[n * 3 + 1]
                offset_inner = int(0)
                # Build Explanation of optimization
                particle_explanation += f"Layer: {layer} \t Coeff: {coeff} \t text: {vector['text']} \n"
                # reset the vectors to the best optimization
                add_steering_vector(layer, coeff, vector['text'], offset_inner)
                # increment the counter
                n = n + 1
                
            return particle_explanation
        else:
            return "Please add some steering vectors for optimization"
    

    add_button.click(add_steering_vector, inputs=[layer_idx, coeff, text, offset], outputs=[add_output])
    optimize_button.click(optimize_steering_to_eval, inputs=[optimize_particles, optimize_iterations], outputs=[add_output])
    reset_button.click(reset_steering_vectors)
    get_button.click(get_steering_vectors, outputs=[steering_vectors_output])
    pass
