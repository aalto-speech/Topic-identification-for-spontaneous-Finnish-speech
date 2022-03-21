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
from sklearn.metrics import accuracy_score
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
        embedded_words, embedded_words_lens = batch.embedded_words

        encoder_out_avg = torch.mean(encoder_out, dim=1)
        embedded_words_avg = torch.mean(embedded_words, dim=1)
         
        # Embed tokens and pass tokens & encoded signal to decoder
        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outputs, _ = self.modules.decoder(embedded_tokens, encoder_out, lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}
        
        # Output layer for topic prediction
        topic_logits = self.modules.topic_lin(torch.cat((encoder_out_avg, embedded_words_avg), dim=-1))
        predictions["topic_logprobs"] = self.hparams.log_softmax(topic_logits)


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
        topics, topic_lens = batch.topics_encoded

        #tokens, token_lens = batch.tokens
        #loss_ctc = self.hparams.ctc_cost(predictions["ctc_logprobs"], tokens, lens, token_lens)

        loss_asr = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )
        

        loss_topic = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["topic_logprobs"],
            targets=topics.squeeze(-1),
            label_smoothing=self.hparams.label_smoothing,
        )


        # Add ctc loss if necessary. The total cost is a weighted sum of ctc loss + seq2seq loss
        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            tokens, token_lens = batch.tokens
            loss_ctc = self.hparams.ctc_cost(predictions["ctc_logprobs"], tokens, lens, token_lens)
            loss_asr *= 1 - self.hparams.ctc_weight
            loss_asr += self.hparams.ctc_weight * loss_ctc
        

        # Combine ASR and Topic loss
        loss = loss_topic + loss_asr
        
        if stage != sb.Stage.TRAIN:
            # Converted predicted tokens from indexes to words
            predicted_words = [self.hparams.tokenizer.decode_ids(prediction).split(" ") for prediction in predictions["tokens"]]
            target_words = [words.split(" ") for words in batch.words]
            
            # Monitor word error rate and character error rated at valid and test time.
            self.wer_metric.append(batch.id, predicted_words, target_words)
            self.cer_metric.append(batch.id, predicted_words, target_words)
            self.accuracy_metric.append(predictions["topic_logprobs"].unsqueeze(1), topics, topic_lens)
 
        return loss

    
    def on_stage_start(self, stage, epoch=None):
       # Set up statistics trackers for this stage
        # In this case, we would like to keep track of the word error rate (wer)
        # and the character error rate (cer)
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer() 
            self.cer_metric = self.hparams.cer_computer()
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
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
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
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                num_to_keep=1
            )
            
            #self.checkpointer.save_and_keep_only(
            #    meta={"ACC": stage_stats["ACC"]}, max_keys=["ACC"],
            #    num_to_keep=1
            #)

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

                #for i in range(len(true_topics)):
                #    print("New")
                #    print(pred_topics[i])
                #    print(true_topics[i])
                
                # ASR prediction
                pred_batch = []
                predicted_words = [self.hparams.tokenizer.decode_ids(prediction).split(" ") for prediction in predictions["tokens"]]

                for sent in predicted_words:
                    #sent = " ".join(sent)
                    #pred_batch.append(sent)
                    sent = filter_repetitions(sent, 3)
                    sent = " ".join(sent)
                    pred_batch.append(sent)


                #pred_labels.append(pred_batch)
                #true_labels.append(batch.words)
                
                for i in range(len(pred_batch)):
                    pred_labels.append(pred_batch[i])
                    true_labels.append(batch.words[i])
                    #print("New")
                    #print(pred_batch[i])
                    #print(batch.words[i])

                #with open("data/untranscribed_predictions.txt", "a") as f:
                #    for i in range(len(pred_batch)):
                #        f.write(batch.id[i] + " " + pred_batch[i])
                #        f.write("\n")

        

        true_topics = np.concatenate(true_topics)
        pred_topics = np.concatenate(pred_topics)
        print('Accuracy: ', accuracy_score(true_topics, pred_topics))
        print('WER: ', wer(true_labels, pred_labels) * 100)
        print('CER: ', cer(true_labels, pred_labels) * 100)



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


def data_prep(data_folder, bert_model, bert_tokenizer, asr_brain, hparams):
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
    @sb.utils.data_pipeline.takes("words", "topic")
    @sb.utils.data_pipeline.provides("words", "embedded_words", "tokens_bos", "tokens_eos", "tokens", "topics_encoded")
    def text_pipeline(text, topic):
        text = text.replace("  ", " ")
        if text[0] == " ":
            text = text[1:]
        yield text
        embedded_words = asr_brain.extract_bert_embeddings(bert_model, bert_tokenizer, text)
        yield embedded_words
        tokens_list = hparams["tokenizer"].encode_as_ids(text)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        topics_encoded = hparams["topic_encoder"].encode_sequence_torch([topic])
        yield topics_encoded

    
    
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)
    
    hparams["topic_encoder"].update_from_didataset(train_data, output_key="topic")

    # save the encoder
    #hparams["topic_encoder"].save(hparams["topic_encoder_file"])
    
    # load the encoder
    hparams["topic_encoder"].load_if_possible(hparams["topic_encoder_file"])

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "words", "embedded_words","tokens_bos", "tokens_eos", "tokens", "topics_encoded"])
    
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
    
    
    # Initialize BERT model
    bert_model = BertModel.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1", output_hidden_states=True).to(device)
    bert_model.eval()
    bert_tokenizer = BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1")
 

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        )

 
    # Dataset creation
    train_data, valid_data, test_data = data_prep("data", bert_model, bert_tokenizer, asr_brain, hparams)
   
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
