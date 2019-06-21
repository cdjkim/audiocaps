import argparse
import json
import os
from collections import OrderedDict, defaultdict
import random
import pprint

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import colorlog
import colorful
import shutil

from models.pyramid_lstm import PyramidLSTM

from configs import add_arguments, create_or_load_hparams
from helpers import input_helper
from helpers.etc_helper import create_or_load_model, dict_to_summary, load_pretrained_word_vector
from utils import vocab_utils
from utils.evaluator import Evaluator
from utils.etc_utils import set_logger, dict_to_matrix, _save_prediction_answer, _trim_after_eos


pp = pprint.PrettyPrinter()


def create_graph(cfg, graph, mode):
    colorlog.info("Build %s graph" % mode)
    with graph.as_default(), tf.container(mode):
        # XXX : if you implement new input pipeline, add here.
        if cfg.data_name == "audiocaps":
            prefetch_data_fn = input_helper.prefetch_dataset
            vocab_fname = os.path.join('data/audiocaps/features/auxiliary', '{}.vocab'.format(cfg.vocab_size))
        else:
            raise NotImplementedError()

        # Read vocab
        _, index_to_string = vocab_utils.create_vocab_tables(vocab_fname, cfg.vocab_size)

        # Read dataset
        num_data, iterator_init, iterators = prefetch_data_fn(
            cfg.batch_size, cfg.bucket_width, cfg.buffer_size, cfg.random_seed, cfg.num_gpus,
            cfg.num_epochs, mode, cfg.feature_name,
        )

        iters_in_data = int(num_data / cfg.batch_size / cfg.num_gpus)


        # Build model
        # XXX : add here if you implement new model
        model_args = cfg, index_to_string, iterators, iters_in_data, mode == "train"
        if cfg.model_name == "PyramidLSTM":
            model = PyramidLSTM(*model_args)
        else:
            raise NotImplementedError()

    return model, iters_in_data, iterator_init



def run_evaluation(model, session, epoch, num_steps, cfg, global_step):
    evaluator = Evaluator()
    results_dict = None

    # Run evaluation steps
    for current_step in tqdm(range(num_steps), ncols=40, desc="Epoch {} (test)".format(epoch)):
        step_result = model.test(session)

        if results_dict is None:
            results_dict = {key: [] for key in step_result.keys()}

        for key, value in step_result.items():
            results_dict[key].append(value)

    # Logging
    log_dict = OrderedDict()


    if "perplexity" in results_dict:
        log_dict["test/perplexity"] = np.exp(np.mean(results_dict["perplexity"]))

    if "prediction" in results_dict and "answer" in results_dict and "access_id" in results_dict:
        tqdm.write(colorful.bold_green("Show samples").styled_string)

        flattend_prediction = _trim_after_eos(np.concatenate(results_dict["prediction"], axis=0).astype(str))
        flattend_answer = _trim_after_eos(np.concatenate(results_dict["answer"], axis=0).astype(str))
        flattend_access_id = np.concatenate(results_dict["access_id"], axis=0).astype(str)

        # Save results as file
        _save_prediction_answer(flattend_prediction, flattend_answer, flattend_access_id,
                                cfg.checkpoint_dir, int(global_step))

        # Map pred-answer by access_id
        pred_answers_dict = defaultdict(lambda: {'prediction': None, "answers": []})
        for prediction, answer, access_id in zip(flattend_prediction, flattend_answer, flattend_access_id):
            pred_answers_dict[access_id]['prediction'] = prediction
            pred_answers_dict[access_id]['answers'].append(answer)

        # Show examples
        show_keys = random.sample(list(pred_answers_dict.keys()), 10)
        examples = []
        for show_key in show_keys:
            prediction = pred_answers_dict[show_key]['prediction']
            answers = pred_answers_dict[show_key]['answers']
            tqdm.write("{}.".format(show_key))
            for idx, answer in enumerate(answers):
                tqdm.write(f'(gt-{idx}) {answer}')
            tqdm.write(f'(pred) {prediction}')
            tqdm.write(' ')
            examples.append([answers[0], prediction])

        coco_result = evaluator.evaluation_with_dict(pred_answers_dict, method="coco")

        for k, v in coco_result.items():
            log_dict["coco/{}".format(k)] = v

    log_string = []
    for key, value in log_dict.items():
        log_string.append("{}: {:.6}".format(key, value))
    log_string = colorful.bold_red(pp.pformat(" ".join(log_string))).styled_string
    tqdm.write(log_string)

    # Add result to tensorboard
    summary = dict_to_summary(log_dict)

    return summary, log_dict, examples


def main(args):
    cfg = create_or_load_hparams(args)
    set_logger()

    # Build graph
    train_graph = tf.Graph()
    test_graph = tf.Graph()
    train_model, iters_in_train, train_iterator_init = create_graph(cfg, train_graph, "train")
    test_model, iters_in_test, test_iterator_init = create_graph(cfg, test_graph, "test")

    # Build session
    config_proto = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.95),
        allow_soft_placement=True,
        log_device_placement=False)
    train_session = tf.Session(config=config_proto, graph=train_graph)
    test_session = tf.Session(config=config_proto, graph=test_graph)

    # Load pretrained word vectors
    pretrained_word_vector = load_pretrained_word_vector(cfg)

    # Initialize models/vocab/iterator
    colorlog.info("Initialize models, vocab, iterator")
    with train_graph.as_default():
        loaded_train_model, is_fresh = create_or_load_model(train_model,
                                    cfg.checkpoint_dir, train_session, "train")
    if is_fresh and pretrained_word_vector is not None:
        _ = loaded_train_model.apply_word_vector(train_graph, train_session,
                                                 pretrained_word_vector)
    train_session.run(train_iterator_init)

    # Summary writer
    summary_writer = tf.summary.FileWriter(cfg.checkpoint_dir, train_graph)

    # Print Hparams
    cfg_dict = json.loads(cfg.to_json())
    for key, value in cfg_dict.items():
        print(key, ":", value)

    # Add Hparams to Tensorboard
    with train_graph.as_default():
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(dict_to_matrix(cfg_dict)))
    summary_writer.add_summary(config_summary.eval(session=train_session))

    # Build graph in order to add examples to tensorboard
    with test_graph.as_default():
        examples_placeholder = tf.placeholder(tf.string, [None, 2])
        examples_summary = tf.summary.text("GT and Pred", examples_placeholder)

    # Build auxiliary checkpoint saving directory
    os.makedirs(os.path.join(cfg.checkpoint_dir, 'auxiliary'))

    colorlog.info("Start training")
    current_epoch = 0
    while True:
        current_epoch += 1
        # Train
        tqdm.write(cfg.other_info)
        for current_step in tqdm(range(iters_in_train),
                desc="Epoch {} (train), {}, {}".format(current_epoch, cfg.model_name, cfg.data_name), ncols=50):

            step_result = loaded_train_model.train(train_session)
            global_step = step_result['global_step']
            summaries = step_result['summaries']

            # Logging
            if (global_step + 1) % 20 == 0:
                for summary in summaries:
                    summary_writer.add_summary(summary, global_step)

            # Test
            if (global_step + 1) % int(iters_in_train * cfg.evaluation_epoch) == 0:
                # Save checkpoint
                loaded_train_model.saver.save(
                    train_session,
                    os.path.join(cfg.checkpoint_dir, "model.ckpt"),
                    global_step=loaded_train_model.global_step)

                # Load checkpoint
                with test_graph.as_default():
                    loaded_test_model, _ = create_or_load_model(
                        test_model, cfg.checkpoint_dir, test_session, "test")

                # Run test
                test_session.run(test_iterator_init)
                test_steps = min(cfg.num_test_steps, iters_in_test)


                test_summary, log_dict, examples = run_evaluation(
                    loaded_test_model, test_session, current_epoch, test_steps, cfg, global_step)
                summary_writer.add_summary(test_summary, global_step)

                # Add examples to tensorboard
                examples_summary_str = test_session.run(examples_summary, feed_dict={examples_placeholder: examples})
                summary_writer.add_summary(examples_summary_str, global_step)

                tqdm.write(cfg.other_info)

        if current_epoch == cfg.end_epoch:
            break


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    add_arguments(argparser)
    args = argparser.parse_args()
    main(args)

