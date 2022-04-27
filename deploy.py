from argparse import ArgumentParser

import numpy as np
import pandas as pd
from core.static_funcs import load_model, intialize_model, random_prediction,deploy_relation_prediction,deploy_triple_prediction,deploy_tail_entity_prediction,deploy_head_entity_prediction
import json
from collections import namedtuple
import torch
import gradio as gr
import random
from core import KGE




def launch_kge(args, pre_trained_kge: KGE):
    """
    Launch server
    :param args:
    :param pre_trained_kge:
    :return:
    """

    def predict(str_subject: str, str_predicate: str, str_object: str, random_examples: bool):
        if random_examples:
            return random_prediction(pre_trained_kge)
        else:
            if pre_trained_kge.is_seen(entity=str_subject) and pre_trained_kge.is_seen(
                    relation=str_predicate) and pre_trained_kge.is_seen(entity=str_object):
                """ Triple Prediction """
                return deploy_triple_prediction(pre_trained_kge, str_subject, str_predicate, str_object)

            elif pre_trained_kge.is_seen(entity=str_subject) and pre_trained_kge.is_seen(
                    relation=str_predicate):
                """ Tail Entity Prediction """
                return deploy_tail_entity_prediction(pre_trained_kge, str_subject, str_predicate, args['top_k'])
            elif pre_trained_kge.is_seen(entity=str_object) and pre_trained_kge.is_seen(
                    relation=str_predicate):
                """ Head Entity Prediction """
                return deploy_head_entity_prediction(pre_trained_kge, str_object, str_predicate, args['top_k'])
            elif pre_trained_kge.is_seen(entity=str_subject) and pre_trained_kge.is_seen(entity=str_object):
                """ Relation Prediction """
                return deploy_relation_prediction(pre_trained_kge, str_subject, str_object, args['top_k'])
            else:
                KeyError('Uncovered scenario')
        # If user simply select submit
        return random_prediction(pre_trained_kge)

    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=1, placeholder=None, label='Subject'),
                gr.inputs.Textbox(lines=1, placeholder=None, label='Predicate'),
                gr.inputs.Textbox(lines=1, placeholder=None, label='Object'), "checkbox"],
        outputs=[gr.outputs.Textbox(label='Input Triple'),
                 gr.outputs.Dataframe(label='Outputs')],
        title=f'{pre_trained_kge.name} Deployment',
        description='1. Enter a triple to compute its score,\n'
                    '2. Enter a subject and predicate pair to obtain most likely top ten entities or\n'
                    '3. Checked the random examples box and click submit').launch(share=args['share'])


def run(args: dict):
    # (1) Train a knowledge graph embedding model
    # (2) Give the path of serialized (1)
    pre_trained_kge = KGE(path_of_pretrained_model_dir=args['path_of_experiment_folder'])
    print(f'Done!\n')
    launch_kge(args, pre_trained_kge)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_of_experiment_folder", type=str, default='Experiments/2022-04-27 20:53:49.861173')
    parser.add_argument('--share', default=True, type=eval, choices=[True, False])
    parser.add_argument('--top_k', default=10, type=int)
    run(vars(parser.parse_args()))
