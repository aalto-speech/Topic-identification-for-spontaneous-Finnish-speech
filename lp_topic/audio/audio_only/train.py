import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

from transformers import BertTokenizer, BertModel
import sentencepiece as spm
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from hyperpyyaml import load_hyperpyyaml
import os
import sys
import numpy as np
import tqdm
from jiwer import wer, cer


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        #"Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalizer(feats, lens)

        encoder_out = self.modules.encoder(feats.detach())
        encoder_out_avg = torch.mean(encoder_out, dim=1)
         
        # Output layer for topic prediction
        topic_logits = self.modules.topic_lin(encoder_out_avg)
        predictions = {"topic_logprobs": self.hparams.log_softmax(topic_logits)}

        return predictions, lens


    def compute_objectives(self, predictions, batch, stage):
        # Compute NLL loss
        predictions, lens = predictions
        topics, topic_lens = batch.topics_encoded


        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["topic_logprobs"],
            targets=topics.squeeze(-1),
            label_smoothing=self.hparams.label_smoothing,
        )

                
        if stage != sb.Stage.TRAIN:
            # Monitor word error rate and character error rated at valid and test time.
            self.accuracy_metric.append(predictions["topic_logprobs"].unsqueeze(1), topics, topic_lens)
 
        return loss

    
    def on_stage_start(self, stage, epoch=None):
       # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.accuracy_metric = self.hparams.accuracy_computer()



    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Summarize the statistics from the stage for record-keeping.
        else:
            stage_stats["ACC"] = self.accuracy_metric.summarize()


        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["ACC"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            ) 
            
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"]}, max_keys=["ACC"],
                num_to_keep=1
            )


        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            with open(self.hparams.decode_text_file, "w") as fo:
                for utt_details in self.wer_metric.scores:
                    print(utt_details["key"], " ".join(utt_details["hyp_tokens"]), file=fo)
    
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        super().on_evaluate_start(max_key=max_key, min_key=min_key)
        
        ckpts = self.checkpointer.find_checkpoints(
                max_key=max_key,
                min_key=min_key,
        )
        model_state_dict = sb.utils.checkpoints.average_checkpoints(
                ckpts, "model" 
        )
        self.hparams.model.load_state_dict(model_state_dict)


    def is_ctc_active(self, stage):
        """Check if CTC is currently active.
        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current


        return current_epoch <= self.hparams.number_of_ctc_epochs
    

    def transcribe_dataset(
            self,
            dataset, # Must be obtained from the dataio_function
            max_key, # We load the model with the highest ACC
            loader_kwargs, # opts for the dataloading
            tokenizer
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.checkpointer.recover_if_possible(max_key=max_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            true_topics = []
            pred_topics = []
            #for batch in tqdm(dataset, dynamic_ncols=True):
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 
                out = self.compute_forward(batch, stage=sb.Stage.TEST) 
                predictions, wav_lens = out


                # Topic prediction
                topi, topk = predictions["topic_logprobs"].topk(1)
                topk = topk.squeeze()

                topics, topic_lens = batch.topics_encoded
                topics = topics.squeeze()

                topk = topk.cpu().detach().numpy()
                topics = topics.cpu().detach().numpy()
                true_topics.append(topics)
                pred_topics.append(topk)
                
                
                #with open("output/predictions_5_dev.txt", "a") as f:
                #    for i in range(topi.size(0)):
                #        if topk[i] == 1:
                #            topk[i] = 2
                #        elif topk[i] == 2:
                #            topk[i] = 1
                #        f.write(str(topk[i]) + " " + str(torch.exp(topi[i]).item()) + "\n")


        true_topics = np.concatenate(true_topics)
        pred_topics = np.concatenate(pred_topics)

        #np.save("output/true_topics.npy", true_topics)
        #np.save("output/pred_topics.npy", pred_topics)
        print('Accuracy: ', accuracy_score(true_topics, pred_topics))
        print('F1: ', f1_score(true_topics, pred_topics, average="micro"))
        print('UAR: ', balanced_accuracy_score(true_topics, pred_topics))



    def extract_bert_embeddings(self, model, tokenizer, data):
        sentence = []
        words = []
        sent = '[CLS] ' + data + ' [SEP]'
    
        # tokenize it
        tokenized_sent = tokenizer.tokenize(sent)
        if len(tokenized_sent) > 512:
            tokenized_sent = tokenized_sent[:511]
            tokenized_sent.append('[SEP]')
        sent_idx = tokenizer.convert_tokens_to_ids(tokenized_sent)
    
        # add segment ID
        segments_ids = [1] * len(sent_idx)
    
        # convert data to tensors
        sent_idx = torch.tensor([sent_idx]).to(self.device)
        segments_ids = torch.tensor([segments_ids]).to(self.device)
        
        #get embeddings
        with torch.no_grad():
            outputs = model(sent_idx, segments_ids)
            hidden_states = outputs[2]
            hidden_states = torch.stack(hidden_states, dim=0)
            embeddings = torch.sum(hidden_states[-4:], dim=0)
            embeddings = embeddings.squeeze()
            sentence.append(embeddings)
        
        return sentence[0]


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev.json"), replacements={"data_root": data_folder})

    datasets = [train_data, valid_data, test_data]
    

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = sb.dataio.dataio.read_audio(file_path)
        if len(sig.size()) == 2:
            sig = torch.mean(sig, dim=-1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("topic")
    @sb.utils.data_pipeline.provides("topics_encoded")
    def text_pipeline(topic):
        topics_encoded = hparams["topic_encoder"].encode_sequence_torch([topic])
        yield topics_encoded

    
    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    hparams["topic_encoder"].update_from_didataset(train_data, output_key="topic")

    # save the encoder
    #hparams["topic_encoder"].save(hparams["topic_encoder_file"])
    
    # load the encoder
    hparams["topic_encoder"].load_if_possible(hparams["topic_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "topics_encoded"])
    
    train_data = train_data.filtered_sorted(sort_key="length", reverse=False)
    
    return train_data, valid_data, test_data




def main(device="cuda"):
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    sb.utils.distributed.ddp_init_group(run_opts) 
    
    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

 
    # Dataset creation
    train_data, valid_data, test_data = data_prep("data", hparams)
   
    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        ###asr_brain.checkpointer.delete_checkpoints(num_to_keep=0)
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=hparams["dataloader_options"],
        )
    
    else:
        # evaluate
        print("Evaluating")
        asr_brain.transcribe_dataset(test_data, "ACC", hparams["test_dataloader_options"], hparams["tokenizer"])


if __name__ == "__main__":
    main()
