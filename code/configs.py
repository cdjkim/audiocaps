import os
import json

import tensorflow as tf

def pyramid_lstm_arguments(parser):
    # encoder
    parser.add_argument("--encoder_dropout", type=float, default=0.0, help="0.0 means no dropout is applied")
    parser.add_argument("--encoder_unit_type", type=str, default="layer_norm_lstm")
    parser.add_argument("--L0_encoder_type", type=str, default="simple_brnn")
    parser.add_argument("--L0_encoder_num_units", type=int, default=256)
    parser.add_argument("--L1_encoder_type", type=str, default="simple_brnn")
    parser.add_argument("--L1_encoder_num_units", type=int, default=256)
    parser.add_argument("--L2_encoder_type", type=str, default="simple_brnn")
    parser.add_argument("--L2_encoder_num_units", type=int, default=256)
    parser.add_argument("--L3_encoder_type", type=str, default="simple_brnn")
    parser.add_argument("--L3_encoder_num_units", type=int, default=256)

    parser.add_argument("--pass_hidden_state", type="bool", default=False)
    parser.add_argument("--method", type=str, default="pyramid_semal_attention")
    parser.add_argument("--combine", type=str, default="add")

    parser.add_argument("--context_encoder_dropout", type=float, default=0.2)
    parser.add_argument("--context_encoder_num_units", type=int, default=256)
    parser.add_argument("--word_encoder_dropout", type=float, default=0.2)
    parser.add_argument("--word_encoder_num_units", type=int, default=256)
    parser.add_argument("--context_concat", type="bool", default=False)
    parser.add_argument("--use_context_lstm", type="bool", default=False)
    parser.add_argument("--attention_flow_type", type=str, default="full")


def add_model_arguments(parser, model_name):
    if model_name == "PyramidLSTM":
        pyramid_lstm_arguments(parser)
    else:
        raise NotImplementedError()


def add_arguments(parser):
    # Defaults arguments comes here
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=12345)
    parser.add_argument("--model_name", type=str, default="PyramidLSTM")
    parser.add_argument("--other_info", type=str, default="")
    parser.add_argument("--override_hparams", type="bool", default=False)

    # pretrained word vector
    parser.add_argument("--pretrained_word", type=str, default="cached")
    parser.add_argument("--fasttext_fname", type=str, default='data/pretrained_word_vectors/fastText.commoncrawl/crawl-300d-2M.vec')
    parser.add_argument("--cached_emb_fname", type=str, default='data/pretrained_word_vectors/fastText.commoncrawl/audiocaps_cached.pkl')

    parser.add_argument("--finetuning_word", type="bool", default=True)

    # data
    parser.add_argument("--data_name", type=str, default="audiocaps")
    parser.add_argument("--feature_name", type=str, default="fc3c4_pyramid")
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=51)  # Include EOS
    parser.add_argument("--bucket_width", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=30000)

    parser.add_argument("--modal", type=str, default="audio")

    # vocab size / embeddings
    parser.add_argument("--vocab_size", type=int, default=4000)
    parser.add_argument("--word_embed_size", type=int, default=300)

    # logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/default")
    parser.add_argument("--ckpt", type=str, default="model.ckpt")
    parser.add_argument("--logging_step", type=int, default=20)
    parser.add_argument("--evaluation_epoch", type=float, default=0.3)

    # training
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--min_lr", type=float, default=0.0001)
    parser.add_argument("--max_grad_norm", type=float, default=0.4)
    parser.add_argument("--num_epochs_per_decay", type=float, default=15)
    parser.add_argument("--lr_decay_factor", type=float, default=1.0)
    parser.add_argument("--max_to_keep", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # decoder
    parser.add_argument("--decoder_dropout", type=float, default=0.0, help="0.0 means no dropout is applied")
    parser.add_argument("--decoder_unit_type", type=str, default="layer_norm_lstm")
    parser.add_argument("--decoder_type", type=str, default="rnn",
                        help="rnn or attention_rnn")
    parser.add_argument("--decoder_num_units", type=int, default=256)
    parser.add_argument("--decoder_attention_type", type=str, default="luong")
    parser.add_argument("--beam_width", type=int, default="1")

    # test
    parser.add_argument("--num_test_steps", type=int, default=99999)
    parser.add_argument("--num_test_examples_to_visualize", type=int, default=20)
    parser.add_argument("--visualize", type="bool", default=False)
    parser.add_argument("--end_epoch", type=int, default=99999)


    args, _ = parser.parse_known_args()
    model_name = args.model_name
    add_model_arguments(parser, model_name)


def create_or_load_hparams(args):
    hparams = tf.contrib.training.HParams()
    for key, value in vars(args).items():
        hparams.add_hparam(key, value)

    # Some additional hparams

    # Save/load hparams
    if not os.path.exists(hparams.checkpoint_dir):
        os.makedirs(hparams.checkpoint_dir)

    hparam_fname = os.path.join(hparams.checkpoint_dir, '%s_hparams.json' % args.model_name)

    if os.path.exists(hparam_fname) and not hparams.override_hparams:
        with open(hparam_fname, 'r') as f:
            loaded_json = json.load(f)
        hparams.parse_json(loaded_json)

    else:
        with open(hparam_fname, 'w') as f:
            json.dump(hparams.to_json(), f, indent=2)
    return hparams



