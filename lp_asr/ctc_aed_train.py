import torch
from torch.utils.data import DataLoader

import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.nnet.losses import nll_loss
from speechbrain.utils.checkpoints import Checkpointer

import sentencepiece as spm
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

        tokens_bos, _ = batch.tokens_bos
        
        encoder_out = self.modules.encoder(feats.detach())
        
        #ctc_logits = self.modules.ctc_lin(encoder_out)
        #predictions = {"ctc_logprobs": self.hparams.log_softmax(ctc_logits)}


        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outputs, _ = self.modules.decoder(embedded_tokens, encoder_out, lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        #predictions["seq_logprobs"] = self.hparams.log_softmax(logits)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}


        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoder_out)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
        elif stage == sb.Stage.VALID:
            predictions["tokens"], _ = self.hparams.greedy_search(encoder_out, lens)
        elif stage == sb.Stage.TEST:
            predictions["tokens"], _ = self.hparams.test_search(encoder_out, lens)
        
        return predictions, lens


    def compute_objectives(self, predictions, batch, stage):
        # Compute NLL loss
        predictions, lens = predictions
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        
        #tokens, token_lens = batch.tokens
        #loss_ctc = self.hparams.ctc_cost(predictions["ctc_logprobs"], tokens, lens, token_lens)

        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )


        # Add ctc loss if necessary. The total cost is a weighted sum of ctc loss + seq2seq loss
        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            tokens, token_lens = batch.tokens
            loss_ctc = self.hparams.ctc_cost(predictions["ctc_logprobs"], tokens, lens, token_lens)
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc

        

        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            predicted_words = [self.hparams.tokenizer.decode_ids(prediction).split(" ") for prediction in predictions["tokens"]]
            #predicted_words = [self.hparams.tokenizer.decode_ids(prediction) for prediction in predictions["tokens"]]

            #predicted_words = sb.decoders.ctc_greedy_decode(predictions["ctc_logprobs"], lens, blank_id=self.hparams.blank_index)
            #predicted_words = [self.hparams.tokenizer.decode_ids(pred) for pred in predicted_words]
            
            
            target_words = [words.split(" ") for words in batch.words]
            #target_words = batch.words
            
            # Monitor word error rate and character error rated at valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)
                                                                    
        return loss

    
    def on_stage_start(self, stage, epoch=None):
       # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer() 
            self.cer_metric = self.hparams.cer_computer() 


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
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")


        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            ) 
            
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                num_to_keep=1
            )


        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

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
            min_key, # We load the model with the lowest WER
            loader_kwargs, # opts for the dataloading
            tokenizer
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )


        #self.on_evaluate_start(min_key=min_key) # We call the on_evaluate_start that will load the best model
        self.checkpointer.recover_if_possible(min_key=min_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            true_labels = []
            pred_labels = []
            #for batch in tqdm(dataset, dynamic_ncols=True):
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 
                out = self.compute_forward(batch, stage=sb.Stage.TEST) 
                predictions, wav_lens = out

                pred_batch = []
                predicted_words = [self.hparams.tokenizer.decode_ids(prediction).split(" ") for prediction in predictions["tokens"]]
                for sent in predicted_words:
                    sent = filter_repetitions(sent, 3)
                    sent = " ".join(sent)
                    pred_batch.append(sent)
                
                

                pred_labels.append(pred_batch)
                true_labels.append(batch.words)
                
                #with open("data/transcribed_test.txt", "a") as f:
                #    for i in range(len(pred_batch)):
                #        f.write(batch.id[i] + " " + pred_batch[i] + "\n")

                #with open("data/true_test.txt", "a") as f:
                #    for i in range(len(pred_batch)):
                #        #print("New")
                #        #print(pred_batch[i])
                #        #print(batch.words[i])
                #        f.write(batch.words[i] + "\n")
        
        
        print("Filtered")
        true_labels = [item for sublist in true_labels for item in sublist]
        pred_labels = [item for sublist in pred_labels for item in sublist]

        print('WER: ', wer(true_labels, pred_labels) * 100)
        print('CER: ', cer(true_labels, pred_labels) * 100)


def filter_repetitions(seq, max_repetition_length):
    seq = list(seq)
    output = []
    max_n = len(seq) // 2
    for n in range(max_n, 0, -1):
        max_repetitions = max(max_repetition_length // n, 1)
        # Don't need to iterate over impossible n values:
        # len(seq) can change a lot during iteration
        if (len(seq) <= n*2) or (len(seq) <= max_repetition_length):
            continue
        iterator = enumerate(seq)
        # Fill first buffers:
        buffers = [[next(iterator)[1]] for _ in range(n)]
        for seq_index, token in iterator:
            current_buffer = seq_index % n
            if token != buffers[current_buffer][-1]:
                # No repeat, we can flush some tokens
                buf_len = sum(map(len, buffers))
                flush_start = (current_buffer-buf_len) % n
                # Keep n-1 tokens, but possibly mark some for removal
                for flush_index in range(buf_len - buf_len%n):
                    if (buf_len - flush_index) > n-1:
                        to_flush = buffers[(flush_index + flush_start) % n].pop(0)
                    else:
                        to_flush = None
                    # Here, repetitions get removed:
                    if (flush_index // n < max_repetitions) and to_flush is not None:
                        output.append(to_flush)
                    elif (flush_index // n >= max_repetitions) and to_flush is None:
                        output.append(to_flush)
            buffers[current_buffer].append(token)
        # At the end, final flush
        current_buffer += 1
        buf_len = sum(map(len, buffers))
        flush_start = (current_buffer-buf_len) % n
        for flush_index in range(buf_len):
            to_flush = buffers[(flush_index + flush_start) % n].pop(0)
            # Here, repetitions just get removed:
            if flush_index // n < max_repetitions:
                output.append(to_flush)
        seq = []
        to_delete = 0
        for token in output:
            if token is None:
                to_delete += 1
            elif to_delete > 0:
                to_delete -= 1
            else:
                seq.append(token)
        output = []
    return seq


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev.json"), replacements={"data_root": data_folder})
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "dev.json"), replacements={"data_root": data_folder})
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(data_folder, "test.json"), replacements={"data_root": data_folder})

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
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(text):
        text = text.replace("  ", " ")
        if text[0] == " ":
            text = text[1:]
        yield text
        tokens_list = hparams["tokenizer"].encode_as_ids(text)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
    
    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "words",  "tokens_bos", "tokens_eos", "tokens"])
    
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

    
    # Dataset creation
    train_data, valid_data, test_data = data_prep('data', hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

    
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
        print("Evaluating...")
        asr_brain.transcribe_dataset(test_data, "WER", hparams["test_dataloader_options"], hparams["tokenizer"])


if __name__ == "__main__":
    main()
