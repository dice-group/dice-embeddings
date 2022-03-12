from argparse import ArgumentParser

import numpy as np
import pandas as pd
from core.static_funcs import load_model, select_model
import json
from collections import namedtuple
import torch
import gradio as gr
import random
from core import KGE


def update_arguments_with_training_configuration(args: dict) -> dict:
    """ namedTuple from input argument """
    settings = dict()
    with open(args['path_of_experiment_folder'] + '/configuration.json', 'r') as r:
        settings.update(json.load(r))
    settings.update(args)
    return settings


def launch_service(config, pretrained_model, entity_idx, predicate_idx):
    """
    Launch a web service for the deployment of a pretrained model.
    :param config:
    :param pretrained_model:
    :param entity_idx:
    :param predicate_idx:
    :return:
    """
    # WN18RR strings are ints. Later read it as string.
    entity_idx = {str(k): v for k, v in entity_idx.items()}
    idx_to_entity = {v: k for k, v in entity_idx.items()}

    def predict(str_subject: str, str_predicate: str, str_object: str, random_examples: bool):
        if random_examples:
            str_subject = random.sample(list(entity_idx.keys()), 1)[0]
            str_predicate = random.sample(list(predicate_idx.keys()), 1)[0]

            idx_subject_idx_predicate = torch.LongTensor(
                [entity_idx[str_subject], predicate_idx[str_predicate]]).reshape(1, 2)
            # Normalize logits via sigmoid
            pred_scores = torch.sigmoid(pretrained_model.forward_k_vs_all(idx_subject_idx_predicate))
            sort_val, sort_idxs = torch.sort(pred_scores, dim=1, descending=True)
            top_10_entity, top_10_score = [idx_to_entity[i] for i in sort_idxs[0][:config['top_k']].tolist()], sort_val[
                                                                                                                   0][
                                                                                                               :config[
                                                                                                                   'top_k']].numpy()
            return f'( {str_subject},{str_predicate}, ? )', pd.DataFrame(
                {'Entity': top_10_entity, 'Score': top_10_score})

        else:
            try:
                idx_subject = torch.LongTensor([entity_idx[str_subject]])
            except KeyError:
                print(f'index of subject **{str_subject}** of length {len(str_subject)} is not found.')
                return 'Failed at mapping the subject', pd.DataFrame()
            try:
                idx_predicate = torch.LongTensor([predicate_idx[str_predicate]])
            except KeyError:
                print(f'index of predicate **{str_predicate}** of length {len(str_predicate)} is not found.')
                return 'Failed at mapping the predicate', pd.DataFrame()

            if len(str_object) == 0:

                pred_scores = torch.sigmoid(
                    pretrained_model.forward_k_vs_all(torch.cat([idx_subject, idx_predicate]).reshape(1, 2)))
                sort_val, sort_idxs = torch.sort(pred_scores, dim=1, descending=True)
                top_10_entity, top_10_score = [idx_to_entity[i] for i in sort_idxs[0][:config['top_k']].tolist()], \
                                              sort_val[0][
                                              :config['top_k']].numpy()
                return f'( {str_subject},{str_predicate}, ? ) ', pd.DataFrame(
                    {'Entity': top_10_entity, 'Score': np.around(top_10_score, 3)})
            else:
                try:
                    idx_object = torch.LongTensor([entity_idx[str_object]])
                except KeyError:
                    print(f'index of object **{str_object}** of length {len(str_object)} is not found.')
                    return 'Failed at mapping the object', pd.DataFrame()
                pred_score = \
                    torch.sigmoid(
                        pretrained_model.forward_k_vs_all(torch.cat([idx_subject, idx_predicate]).reshape(1, 2)))[
                        0, idx_object]
                return f'( {str_subject}, {str_predicate}, {str_object} )', pd.DataFrame(
                    {'Entity': str_object, 'Score': pred_score})

    gr.Interface(
        fn=predict,
        inputs=[gr.inputs.Textbox(lines=1, placeholder=None, label='Subject'),
                gr.inputs.Textbox(lines=1, placeholder=None, label='Predicate'),
                gr.inputs.Textbox(lines=1, placeholder=None, label='Object'), "checkbox"],
        outputs=[gr.outputs.Textbox(label='Input Triple'),
                 gr.outputs.Dataframe(label='Outputs')],
        title=f'{pretrained_model.name} Deployment',
        description='1. Enter a triple to compute its score,\n'
                    '2. Enter a subject and predicate pair to obtain most likely top ten entities or\n'
                    '3. Checked the random examples box and click submit').launch(share=config['share'])


def launch_kge(args, pre_trained_kge: KGE):
    def predict(str_subject: str, str_predicate: str, str_object: str, random_examples: bool):
        if random_examples:
            head_entity: List[str]
            relation: List[str]
            tail_entity: List[str]
            head_entity = pre_trained_kge.sample_entity(1)
            relation = pre_trained_kge.sample_relation(1)
            tail_entity = pre_trained_kge.sample_entity(1)
            triple_score = pre_trained_kge.predict_topk(head_entity=head_entity,
                                                        relation=relation,
                                                        tail_entity=tail_entity)
            return f'( {head_entity[0]},{relation[0]}, {tail_entity[0]} )', pd.DataFrame({'Score': triple_score})
        else:

            if pre_trained_kge.is_seen(entity=str_subject) and pre_trained_kge.is_seen(
                    relation=str_predicate) and pre_trained_kge.is_seen(entity=str_object):
                """ Triple Prediction """
                triple_score = pre_trained_kge.predict_topk(head_entity=str_subject,
                                                            relation=str_predicate,
                                                            tail_entity=str_object)
                return f'( {str_subject},{str_predicate}, {str_object} )', pd.DataFrame({'Score': triple_score})

            elif pre_trained_kge.is_seen(entity=str_subject) and pre_trained_kge.is_seen(
                    relation=str_predicate):
                """ Tail Entity Prediction """

                triple_score = pre_trained_kge.predict_topk(head_entity=[str_subject],
                                                            relation=[str_predicate])

                return f'( {str_subject},{str_predicate}, {str_object} )', pd.DataFrame({'Score': triple_score})


            elif pre_trained_kge.is_seen(relation=str_predicate) and pre_trained_kge.is_seen(entity=str_object):
                """ Head Entity Prediction """
            elif pre_trained_kge.is_seen(entity=str_subject) and pre_trained_kge.is_seen(entity=str_object):
                """ Relation Prediction"""
            else:
                KeyError('Uncovered scenario')

            try:
                idx_subject = torch.LongTensor([entity_idx[str_subject]])
            except KeyError:
                print(f'index of subject **{str_subject}** of length {len(str_subject)} is not found.')
                return 'Failed at mapping the subject', pd.DataFrame()
            try:
                idx_predicate = torch.LongTensor([predicate_idx[str_predicate]])
            except KeyError:
                print(f'index of predicate **{str_predicate}** of length {len(str_predicate)} is not found.')
                return 'Failed at mapping the predicate', pd.DataFrame()

            if len(str_object) == 0:

                pred_scores = torch.sigmoid(
                    pretrained_model.forward_k_vs_all(torch.cat([idx_subject, idx_predicate]).reshape(1, 2)))
                sort_val, sort_idxs = torch.sort(pred_scores, dim=1, descending=True)
                top_10_entity, top_10_score = [idx_to_entity[i] for i in sort_idxs[0][:config['top_k']].tolist()], \
                                              sort_val[0][
                                              :config['top_k']].numpy()
                return f'( {str_subject},{str_predicate}, ? ) ', pd.DataFrame(
                    {'Entity': top_10_entity, 'Score': np.around(top_10_score, 3)})
            else:
                try:
                    idx_object = torch.LongTensor([entity_idx[str_object]])
                except KeyError:
                    print(f'index of object **{str_object}** of length {len(str_object)} is not found.')
                    return 'Failed at mapping the object', pd.DataFrame()
                pred_score = \
                    torch.sigmoid(
                        pretrained_model.forward_k_vs_all(torch.cat([idx_subject, idx_predicate]).reshape(1, 2)))[
                        0, idx_object]
                return f'( {str_subject}, {str_predicate}, {str_object} )', pd.DataFrame(
                    {'Entity': str_object, 'Score': pred_score})

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

    pass


def run(args: dict):
    print('Loading Model...')

    # (1) Train a knowledge graph embedding model
    # (2) Give the path of serialized (1)
    pre_trained_kge = KGE(path_of_pretrained_model_dir=args['path_of_experiment_folder'])
    print(f'Done!\n')
    launch_kge(args, pre_trained_kge)
    exit(1)
    """
    config = update_arguments_with_training_configuration(args)
    
    with open(args['path_of_experiment_folder'] + '/report.json', 'r') as r:
        config.update(json.load(r))
    pretrained_model, _ = select_model(config)
    weights = torch.load(args['path_of_experiment_folder'] + '/model.pt', torch.device('cpu'))
    pretrained_model.load_state_dict(weights)
    for parameter in pretrained_model.parameters():
        parameter.requires_grad = False
    pretrained_model.eval()
    """

    entity_to_idx = pd.read_parquet(args['path_of_experiment_folder'] + '/entity_to_idx.gzip').to_dict()['entity']
    relation_to_idx = pd.read_parquet(args['path_of_experiment_folder'] + '/relation_to_idx.gzip').to_dict()['relation']
    print(f'Done!\n')
    launch_service(config, pretrained_model, entity_to_idx, relation_to_idx)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_of_experiment_folder", type=str, default='Experiments/2022-03-11 08:30:52.896174')
    parser.add_argument('--share', default=False, type=eval, choices=[True, False])
    parser.add_argument('--top_k', default=25, type=int)
    run(vars(parser.parse_args()))
