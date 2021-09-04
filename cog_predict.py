import json
import os
import tempfile
from util import set_seed
import cog
import numpy as np
import torch

import models
from predict import get_all_splits, prepare_data, get_args, to_iter

class Predictor(cog.Predictor):
    def setup(self):
        self.args = get_args([
            "--path", "mqan_decanlp_better_sampling_cove_cpu",
            "--evaluate", "valid",
            "--checkpoint_name", "iteration_560000.pth",
            "--embeddings", "/src/.embeddings"
           ])

        print(f'Loading from {self.args.best_checkpoint}')
        save_dict = torch.load(self.args.best_checkpoint)
        field = save_dict['field']
        print(f'Initializing Model')
        Model = getattr(models, self.args.model) 
        self.model = Model(field, self.args)
        model_dict = save_dict['model_state_dict']
        backwards_compatible_cove_dict = {}
        for k, v in model_dict.items():
            if 'cove.rnn.' in k:
                k = k.replace('cove.rnn.', 'cove.rnn1.')
            backwards_compatible_cove_dict[k] = v
        model_dict = backwards_compatible_cove_dict
        self.model.load_state_dict(model_dict)

        print("Preparing embeddings...")
        # Running prepare_data with no tasks just fetches embeddings 
        self.args.tasks = []
        self.field, _ = prepare_data(self.args, field)

    @cog.input("context", type=str)
    @cog.input("question", type=str)
    def predict(self, context, question):
        # Input is on disk
        with tempfile.TemporaryDirectory() as data_dir:
            self.args.data = data_dir

            print("Getting splits...")
            os.makedirs(os.path.join(data_dir, "custom_task"))
            with open(os.path.join(data_dir, "custom_task/val.jsonl"), "w") as fh:
                json.dump({"context": context, "question": question, "answer": ""}, fh)
            self.args.tasks = ["custom_task"]

            splits = get_all_splits(self.args, self.field)
            self.model.set_embeddings(self.field.vocab.vectors)

            print("Running prediction...")
            device = set_seed(self.args)
            self.model.to(device)
            self.model.eval()

            it = to_iter(splits[0], 1, device)
            batch = list(it)[0]

            with torch.no_grad():
                _, p = self.model(batch)
                p = self.field.reverse(p)

                return p[0]



