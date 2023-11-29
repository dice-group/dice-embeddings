.. DICE Embeddings documentation master file, created by
   sphinx-quickstart on Mon Aug 14 13:07:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DICE Embeddings!
===========================================

`DICE Embeddings <https://github.com/dice-group/dice-embeddings>`_: Hardware-agnostic Framework for Large-scale Knowledge Graph Embeddings:
=======



.. warning::

   Train embedding models in multi-node, multi-GPUs, distributed data parallel or model parallel without expert knowledge!


   .. code-block:: bash

      // 1 CPU
      (dicee) $ dicee --dataset_dir KGs/UMLS
      // 10 CPU
      (dicee) $ dicee --dataset_dir KGs/UMLS --num_core 10
      // Distributed Data Parallel (DDP) with all GPUs
      (dicee) $ dicee --trainer PL --accelerator gpu --strategy ddp --dataset_dir KGs/UMLS
      // Model Parallel with all GPUs and low precision
      (dicee) $ dicee --trainer PL --accelerator gpu --strategy deepspeed_stage_3 --dataset_dir KGs/UMLS --precision 16
      // DDP with all GPUs on two nodes (felis and nebula):
      (dicee) cdemir@felis  $ torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 0 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula -m dicee.run --trainer torchDDP --dataset_dir KGs/UMLS
      (dicee) cdemir@nebula $ torchrun --nnodes 2 --nproc_per_node=gpu  --node_rank 1 --rdzv_id 455 --rdzv_backend c10d --rdzv_endpoint=nebula -m dicee.run --trainer torchDDP --dataset_dir KGs/UMLS

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Usage
-------

.. code-block:: console

   $ git clone https://github.com/dice-group/dice-embeddings.git

   $ conda create -n dice python=3.9.18 --no-default-packages && conda activate dice

   (dice) $ pip3 install -r requirements.txt

or

.. code-block:: console

   (dice) $ pip install dicee




Indices and tables
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
