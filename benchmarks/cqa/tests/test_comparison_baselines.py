"""Missing baseline integration, resume-safe heuristic and author training parity."""

import copy
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarks.cqa import cli
from benchmarks.cqa.evaluate import evaluate_manifest
from benchmarks.cqa.manifests import REPO, prepare_manifest, read_manifest, validate_entry
from benchmarks.cqa.oracles import verify_predictions
from benchmarks.cqa.study import freeze, run_job
from benchmarks.cqa.tests.test_ultraquery_launcher import PUBLIC, fixture
from benchmarks.cqa.train import verify_v2
from dicee.query_answering import QueryContext, load_benchmark
from dicee.query_answering._checkpoint import checksum
from dicee.query_answering.methods import load_method
from dicee.query_answering.methods.heuristic import IncomingRelationHeuristic

EXPORTER = REPO / 'benchmarks/cqa/verification/export_reference.py'


def test_comparison_recipes_cover_published_families():
    m = prepare_manifest([PUBLIC / 'comparisons.json', PUBLIC / 'trained_baselines.json'],
                         suite='ultraquery', answer_filter='released')
    counts = {}
    for e in m['entries']:
        validate_entry(e)
        counts[e['method']] = counts.get(e['method'], 0) + 1
    assert counts == {'ultraquery-lp': 46, 'incoming-relation': 11, 'qto': 3, 'inductive-gnnqe': 9}
    thresholds = {e['dataset']: e['options']['threshold'] for e in m['entries']
                  if e['method']=='ultraquery-lp' and '-no-threshold-' not in e['id']}
    assert thresholds['NELL995LogicalQuery'] == .97
    assert all(value == (.97 if name=='NELL995LogicalQuery' else .8) for name,value in thresholds.items())
    assert all('2404.07198v2' in e['reference']['settings'] for e in m['entries'] if e['method']=='ultraquery-lp')


def test_heuristic_ignores_heads_and_random_order_is_subset_batch_resume_stable():
    context = QueryContext([(0, 0, 1), (2, 0, 3), (1, 1, 2)], 5, 2)
    m = IncomingRelationHeuristic(context, seed=19)
    a, b = (0, (0,)), (4, (0,))
    classes = m.membership_batch([('project', 0, ('anchor', 0)), ('project', 0, ('anchor', 4))])
    assert torch.equal(classes[0], classes[1])
    assert classes[0].tolist() == [0., 1., 0., 1., 0.]
    scores = m.predict_batch([a, b])
    assert torch.equal(scores[0], m.predict(a)) and torch.equal(scores[1], m.predict(b))
    torch.rand(100)
    assert torch.equal(scores, m.predict_batch([a, b]))
    assert torch.equal(scores.flip(0), m.predict_batch([b, a]))
    assert (scores[:, [1, 3]].min() > scores[:, [0, 2, 4]].max())
    assert not torch.equal(scores, IncomingRelationHeuristic(context, seed=20).predict_batch([a, b]))


def test_v2_gate_rejects_unknown_or_missing_training_inputs(tmp_path):
    with pytest.raises(ValueError, match='Missing v2'):
        verify_v2(tmp_path, '106')
    (tmp_path / 'train_graph.txt').write_text('0 0 1\n')
    with pytest.raises(ValueError):
        verify_v2(tmp_path, '106')


def test_train_selection_cpu_and_no_writes_in_dry_run(tmp_path, capsys):
    cli.main(['ultraquery', 'train', '--methods', 'inductive-gnnqe', '--datasets', 'InductiveFB15k237Query:106',
              '--output', str(tmp_path / 'models'), '--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    assert plan['device']=='cpu' and len(plan['jobs'])==1
    assert not (tmp_path / 'models').exists()
    for suite, selection in (('ultraquery', ['--methods', 'qto', '--datasets', 'WikiTopicsQuery:art']), ('plus_h', ['--methods', 'qto'])):
        with pytest.raises(SystemExit):
            cli.main([suite, 'train', *selection, '--output', str(tmp_path), '--dry-run'])


def test_docker_train_defaults_to_cpu(tmp_path, capsys):
    args = ['ultraquery', 'train', '--image', 'test', '--input-root', str(tmp_path), '--output', str(tmp_path / 'models'),
            '--methods', 'qto', '--datasets', 'FB15kLogicalQuery', '--dry-run']
    cli.main(args)
    out = capsys.readouterr().out
    assert '--gpus' not in out and 'nvidia.com/gpu' not in out
    assert 'test ultraquery train --input-root /inputs --output /results' in out
    assert '--manifest-output-root /inputs/models' in out
    with pytest.raises(SystemExit):
        cli.main([*args, '--device', 'cuda'])


def trained_fixture(root):
    manifest=fixture(root,'FB15kLogicalQuery')
    entry=copy.deepcopy(next(e for e in read_manifest(PUBLIC/'trained_baselines.json')['entries']
                             if e['method']=='qto'))
    entry.update(checkpoint='model.pt',training='training.json')
    manifest['entries']=[entry]
    folder=root/'data/FB15k-betae'
    (folder/'valid.txt').write_text('0\t0\t3\n')
    (folder/'test.txt').write_text('1\t2\t3\n')
    torch.save({'embeddings.0.weight':torch.zeros(8,16),'embeddings.1.weight':torch.zeros(4,16)},root/'model.pt')
    record=dict(version=1,status='complete',method='qto',dataset=entry['dataset'],profile='smoke',dataset_release='fixture',
                checkpoint_sha256=checksum(root/'model.pt'),
                inputs={p.name:checksum(p) for p in folder.iterdir() if p.is_file()})
    (root/'training.json').write_text(json.dumps(record))
    return manifest,record,folder


@pytest.mark.parametrize('profile,release',[('smoke','fixture'),('fixture','fixture'),('custom','fixture')])
def test_shipped_manifest_cannot_promote_smoke_or_fixture_weights(tmp_path,profile,release):
    manifest,record,_=trained_fixture(tmp_path)
    record.update(profile=profile,dataset_release=release)
    (tmp_path/'training.json').write_text(json.dumps(record))
    assert manifest['entries'][0]['blockers']==[]
    freeze(manifest,tmp_path/'bundle',tmp_path)
    with pytest.raises(ValueError,match='smoke/fixture training checkpoint'):
        run_job(tmp_path/'bundle',manifest['entries'][0]['id'],tmp_path,tmp_path/'final',phase='test')
    pilot=run_job(tmp_path/'bundle',manifest['entries'][0]['id'],tmp_path,tmp_path/'pilot',phase='pilot')
    assert pilot['queries']==14


@pytest.mark.parametrize('changed_file',['id2ent.pkl','id2rel.pkl','train.txt','valid.txt','test.txt','test-hard-answers.pkl'])
def test_training_input_changes_rejected_by_freeze_and_direct_evaluation(tmp_path,changed_file):
    manifest,_,folder=trained_fixture(tmp_path)
    path=folder/changed_file
    if changed_file.startswith('id2'):
        mapping=pickle.loads(path.read_bytes())
        mapping[0],mapping[1]=mapping[1],mapping[0]
        path.write_bytes(pickle.dumps(mapping))
    else:
        path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Training input differs'):
        freeze(manifest,tmp_path/'bundle',tmp_path)
    with pytest.raises(ValueError,match='Training input differs'):
        evaluate_manifest(manifest,input_root=tmp_path,output=tmp_path/'direct')
    assert not (tmp_path/'direct').exists()


@pytest.mark.parametrize('missing',['training','input','profile'])
def test_required_training_bindings_cannot_be_omitted(tmp_path,missing):
    manifest,record,_=trained_fixture(tmp_path)
    if missing=='training':
        del manifest['entries'][0]['training']
    elif missing=='input':
        del record['inputs']['id2ent.pkl']
    else:
        del record['profile']
    (tmp_path/'training.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Training provenance'):
        freeze(manifest,tmp_path/'bundle',tmp_path)
    with pytest.raises(ValueError,match='Training provenance'):
        evaluate_manifest(manifest,input_root=tmp_path,output=tmp_path/'direct')


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('DICEE_INDUCTIVE_REFERENCE_ROOT'), reason='Author dependencies and clean checkout required')
@pytest.mark.parametrize('method', ['incoming-relation', 'inductive-gnnqe', 'ultraquery-lp', 'qto'])
def test_author_training_export_parity_and_verified_execution(tmp_path, method):
    name = ('WikiTopicsQuery:art' if method=='incoming-relation' else 'InductiveFB15k237Query:106'
            if method=='inductive-gnnqe' else 'FB15kLogicalQuery')
    manifest = fixture(tmp_path, name)
    e = manifest['entries'][0]
    e['method'], e['id'] = method, method+'-fixture'
    if method=='incoming-relation':
        (tmp_path / 'heuristic.json').write_bytes((PUBLIC / 'heuristic.json').read_bytes())
        e.update(checkpoint='heuristic.json', options={})
    elif method=='ultraquery-lp':
        # The actual released LP weights, rather than a synthetic query checkpoint.
        (tmp_path/'lp.pt').write_bytes((REPO / 'Experiments/query-baselines/upstream/ultra/ckpts/ultra_3g.pth').read_bytes())
        e['checkpoint']='lp.pt'
        e['options']['threshold']=.8
    else:
        e.update(copy.deepcopy(next(row for row in read_manifest(PUBLIC/'trained_baselines.json')['entries']
                                    if row['method']==method and row['dataset']==name)))
        folder = tmp_path / 'data' / ('106' if method=='inductive-gnnqe' else 'FB15k-betae')
        if method=='inductive-gnnqe':
            queries=pickle.loads((folder/'valid_queries.pkl').read_bytes())
            (folder/'train_queries.pkl').write_bytes(pickle.dumps(queries))
            answers={s:{q:{1} for q in group} for s,group in queries.items()}
            (folder/'train_answers_hard.pkl').write_bytes(pickle.dumps(answers))
        else:
            (folder/'valid.txt').write_text('0\t0\t3\n')
            (folder/'test.txt').write_text('1\t2\t3\n')
        source = Path(os.environ['DICEE_INDUCTIVE_REFERENCE_ROOT']) if method=='inductive-gnnqe' else REPO/'Experiments/query-baselines/upstream/qto'
        checkout = tmp_path/'Experiments/query-baselines/upstream'/('inductiveqe' if method=='inductive-gnnqe' else 'qto')
        checkout.parent.mkdir(parents=True)
        checkout.symlink_to(source, target_is_directory=True)
        subprocess.run([sys.executable,'-m','benchmarks.cqa','ultraquery','train','--methods',method,
                        '--datasets',name,'--input-root',str(tmp_path),'--data-root','data',
                        '--output',str(tmp_path/'trained'),'--smoke'],check=True)
        training=tmp_path/'trained'/f'{method}-{name}'
        e['checkpoint']=str(training/'checkpoint.pt')
        metadata=json.loads((training/'training.json').read_text())
        assert metadata['status']=='complete' and metadata['profile']=='smoke' and metadata['updates']==1
        assert metadata['checkpoint_selection']=='validation MRR only'
        generated=json.loads((training/'manifest.json').read_text())
        assert generated['data_root']=='data' and generated['threads']==2
        assert generated['entries'][0]['blockers']
        assert (tmp_path/generated['entries'][0]['checkpoint']).is_file()
        assert (tmp_path/generated['entries'][0]['training']).is_file()
        e['training']=generated['entries'][0]['training']
        freeze(generated,tmp_path/'smoke-bundle',tmp_path)
        with pytest.raises(ValueError,match='not ready for final testing'):
            run_job(tmp_path/'smoke-bundle',generated['entries'][0]['id'],tmp_path,tmp_path/'blocked',phase='test')
        changed=dict(metadata,checkpoint_sha256='0'*64)
        (training/'training.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='Training provenance'):
            freeze(generated,tmp_path/'invalid-training',tmp_path)
        (training/'training.json').write_text(json.dumps(metadata))
        evaluate_manifest(generated,input_root=tmp_path,output=tmp_path/'direct',split='valid',limit=1)
        result=json.loads((tmp_path/'direct'/generated['entries'][0]['id']/'result.json').read_text())
        assert result['inference']['paper_protocol']['training']['profile']=='smoke'
        e['options']={'logic':'product'} if method=='inductive-gnnqe' else {
            'threshold':.001,'negation_scale':6,'cache_bytes':2**20,'row_batch_size':100}
        context=load_benchmark(tmp_path/'data',name,split='valid').context
        native=load_method(method,e['checkpoint'],context,**e['options'])
        if method=='inductive-gnnqe':
            assert native.provenance['checkpoint_graph_buffers']['train_graph']['training_facts_verified']
            from benchmarks.cqa.train import compatibility
            compatibility()
            from torchdrug.data import Graph
            with torch.serialization.safe_globals([Graph]):
                state=torch.load(e['checkpoint'],weights_only=True)
            del state['model']['train_graph']
            torch.save(state,tmp_path/'incomplete.pt')
            with pytest.raises(ValueError,match='Missing inductive checkpoint graph'):
                load_method(method,tmp_path/'incomplete.pt',context,**e['options'])
    freeze(manifest,tmp_path/'bundle',tmp_path)
    source = (Path(os.environ['DICEE_INDUCTIVE_REFERENCE_ROOT']) if method in ('incoming-relation','inductive-gnnqe')
              else REPO/'Experiments/query-baselines/upstream'/('qto' if method=='qto' else 'ultra'))
    subprocess.run([sys.executable,str(EXPORTER),'--bundle',str(tmp_path/'bundle'),
                    '--input-root',str(tmp_path),'--entry',e['id'],'--upstream',str(source),
                    '--device','cpu','--output',str(tmp_path/'reference.pt')],check=True)
    evidence=verify_predictions(tmp_path/'bundle',e['id'],tmp_path,tmp_path/'reference.pt',tmp_path/'evidence.json')
    assert evidence['passed'] and len(evidence['queries'])==14, json.dumps(evidence)
    e['verification']='evidence.json'
    freeze(manifest,tmp_path/'verified',tmp_path)
    if method in ('qto','inductive-gnnqe'):
        # Even successful independent parity cannot promote genuine one-update
        # smoke weights through the shipped manifest's empty blocker list.
        assert e['blockers']==[]
        with pytest.raises(ValueError,match='smoke/fixture training checkpoint'):
            run_job(tmp_path/'verified',e['id'],tmp_path,tmp_path/'test',phase='test')
        report=run_job(tmp_path/'verified',e['id'],tmp_path,tmp_path/'pilot',phase='pilot')
    else:
        report=run_job(tmp_path/'verified',e['id'],tmp_path,tmp_path/'test',phase='test')
    assert report['coverage']['complete_benchmark_types']


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get('DICEE_INDUCTIVE_REFERENCE_ROOT'),reason='Author runtime required')
def test_qto_training_consumes_exact_epochs(tmp_path):
    fixture(tmp_path,'FB15kLogicalQuery')
    folder=tmp_path/'data/FB15k-betae'
    (folder/'valid.txt').write_text('0\t0\t3\n')
    (folder/'test.txt').write_text('1\t2\t3\n')
    checkout=tmp_path/'Experiments/query-baselines/upstream/qto'
    checkout.parent.mkdir(parents=True)
    checkout.symlink_to(REPO/'Experiments/query-baselines/upstream/qto',target_is_directory=True)
    subprocess.run([sys.executable,'-m','benchmarks.cqa','ultraquery','train',
                    '--methods','qto','--datasets','FB15kLogicalQuery','--input-root',str(tmp_path),
                    '--data-root','data','--output',str(tmp_path/'trained'),
                    '--epochs','2','--batch-size','2'],check=True)
    record=json.loads((tmp_path/'trained/qto-FB15kLogicalQuery/training.json').read_text())
    assert record['updates']==6  # Six triples, three batches per epoch, two epochs.
    assert record['best_epoch']==1  # Retain the authors' e % valid == 0 schedule.
    assert record['profile']=='custom' and record['status']=='complete'
    manifest=json.loads((tmp_path/'trained/qto-FB15kLogicalQuery/manifest.json').read_text())
    entry=manifest['entries'][0]
    freeze(manifest,tmp_path/'bundle',tmp_path)
    subprocess.run([sys.executable,str(EXPORTER),'--bundle',str(tmp_path/'bundle'),
                    '--input-root',str(tmp_path),'--entry',entry['id'],
                    '--upstream',str(REPO/'Experiments/query-baselines/upstream/qto'),
                    '--device','cpu','--output',str(tmp_path/'reference.pt')],check=True)
    evidence=verify_predictions(tmp_path/'bundle',entry['id'],tmp_path,tmp_path/'reference.pt',tmp_path/'evidence.json')
    assert evidence['passed']
    entry['verification']='evidence.json'
    freeze(manifest,tmp_path/'verified',tmp_path)
    report=run_job(tmp_path/'verified',entry['id'],tmp_path,tmp_path/'test',phase='test')
    assert report['protocol']['full_split']
