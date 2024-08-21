# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import copy
import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.data.audio_to_text_lhotse_lang_id import LhotseSpeechToTextBpeDatasetLangID
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.metrics.bleu import BLEU
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.utils import logging, model_utils
from nemo.core.neural_types import AudioSignal, LengthsType, NeuralType, SpectrogramType, LabelsType

class EncDecHybridRNNTCTCBPEModelAST(EncDecHybridRNNTCTCModel, ASRBPEMixin):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Tokenizer is necessary for this model
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        # setup auxiliary CTC decoder
        if 'aux_ctc' not in cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )

        with open_dict(cfg):
            if self.tokenizer_type == "agg":
                cfg.aux_ctc.decoder.vocabulary = ListConfig(vocabulary)
            else:
                cfg.aux_ctc.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        if cfg.aux_ctc.decoder["num_classes"] < 1:
            logging.info(
                "\nReplacing placholder number of classes ({}) with actual number of classes - {}".format(
                    cfg.aux_ctc.decoder["num_classes"], len(vocabulary)
                )
            )
            cfg.aux_ctc.decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)
        
        if cfg.get("initialize_lang_embeddings", False):
            self.initialize_embeddings()

    def initialize_embeddings(self):
        self.insertion_method = "concat"
        self.insertion_location = "encoder"
        print("embedding has been initalized")
        if self.tokenizer_type == "agg":
            with open_dict(self.cfg):
                self.cfg.lang_labels = self.tokenizer.langs        
        lang_embed_dim = self.cfg.get("lang_embed_dim", self.cfg.preprocessor["features"])     
        self.lang_embedding = torch.nn.Embedding(
            num_embeddings=len(self.cfg.get("lang_labels")), 
            embedding_dim=lang_embed_dim,
        )

        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
        )

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self.cfg.get('use_cer', False),
            log_prediction=self.cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Setup bleu object
        self.bleu = BLEU(
            decoding=self.decoding,
            tokenize=self.cfg.get('bleu_tokenizer', "13a"),
            log_prediction=True
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Setup CTC decoding
        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg
        self.ctc_decoding = CTCBPEDecoding(self.cfg.aux_ctc.decoding, tokenizer=self.tokenizer)

        # Setup CTC WER
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if self.tokenizer_type == "agg":
            with open_dict(self.cfg):
                self.cfg.lang_labels = self.tokenizer.langs
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='lang_labels')

        if config.get("use_lhotse"):
            if config.get('lang_labels'):
                dataset = LhotseSpeechToTextBpeDatasetLangID(tokenizer=self.tokenizer, lang_labels=config.get('lang_labels'))
            else:
                dataset = LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer)
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=dataset,
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """

        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer (if the tokenizer type is `agg`)
            new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for auxiliary CTC decoding, which is optional and can be used to change the decoding type.

        Returns: None

        """
        if isinstance(new_tokenizer_dir, DictConfig):
            if new_tokenizer_type == 'agg':
                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        joint_config = self.joint.to_config_dict()
        new_joint_config = copy.deepcopy(joint_config)
        if self.tokenizer_type == "agg":
            new_joint_config["vocabulary"] = ListConfig(vocabulary)
        else:
            new_joint_config["vocabulary"] = ListConfig(list(vocabulary.keys()))

        new_joint_config['num_classes'] = len(vocabulary)
        del self.joint
        self.joint = EncDecHybridRNNTCTCBPEModelAST.from_config_dict(new_joint_config)

        decoder_config = self.decoder.to_config_dict()
        new_decoder_config = copy.deepcopy(decoder_config)
        new_decoder_config.vocab_size = len(vocabulary)
        del self.decoder
        self.decoder = EncDecHybridRNNTCTCBPEModelAST.from_config_dict(new_decoder_config)

        del self.loss
        self.loss = RNNTLoss(num_classes=self.joint.num_classes_with_blank - 1)

        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.decoding = RNNTBPEDecoding(
            decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
        )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Update config
        with open_dict(self.cfg.joint):
            self.cfg.joint = new_joint_config

        with open_dict(self.cfg.decoder):
            self.cfg.decoder = new_decoder_config

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed tokenizer of the RNNT decoder to {self.joint.vocabulary} vocabulary.")

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            ctc_decoder_config = copy.deepcopy(self.ctc_decoder.to_config_dict())
            # sidestepping the potential overlapping tokens issue in aggregate tokenizers
            if self.tokenizer_type == "agg":
                ctc_decoder_config.vocabulary = ListConfig(vocabulary)
            else:
                ctc_decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

            decoder_num_classes = ctc_decoder_config['num_classes']
            # Override number of classes if placeholder provided
            logging.info(
                "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                    decoder_num_classes, len(vocabulary)
                )
            )
            ctc_decoder_config['num_classes'] = len(vocabulary)

            del self.ctc_decoder
            self.ctc_decoder = EncDecHybridRNNTCTCBPEModelAST.from_config_dict(ctc_decoder_config)
            del self.ctc_loss
            self.ctc_loss = CTCLoss(
                num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
            )

            if ctc_decoding_cfg is None:
                # Assume same decoding config as before
                ctc_decoding_cfg = self.cfg.aux_ctc.decoding

            # Assert the decoding config with all hyper parameters
            ctc_decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
            ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

            self.ctc_decoding = CTCBPEDecoding(decoding_cfg=ctc_decoding_cfg, tokenizer=self.tokenizer)

            self.ctc_wer = WER(
                decoding=self.ctc_decoding,
                use_cer=self.cfg.aux_ctc.get('use_cer', False),
                log_prediction=self.cfg.get("log_prediction", False),
                dist_sync_on_step=True,
            )

            # Update config
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoder = ctc_decoder_config

            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

            logging.info(f"Changed tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None):
        """
        Changes decoding strategy used during RNNT decoding process.
        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having both RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.decoding = RNNTBPEDecoding(
                decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
            )

            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Setup fused Joint step
            if self.joint.fuse_loss_wer or (
                self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
            ):
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

            self.joint.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            self.cur_decoder = "rnnt"
            logging.info(f"Changed decoding strategy of the RNNT decoder to \n{OmegaConf.to_yaml(self.cfg.decoding)}")

        elif decoder_type == 'ctc':
            if not hasattr(self, 'ctc_decoding'):
                raise ValueError("The model does not have the ctc_decoding module and does not support ctc decoding.")
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.aux_ctc.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            self.ctc_decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer)

            self.ctc_wer = WER(
                decoding=self.ctc_decoding,
                use_cer=self.ctc_wer.use_cer,
                log_prediction=self.ctc_wer.log_prediction,
                dist_sync_on_step=True,
            )

            self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.aux_ctc.decoding):
                self.cfg.aux_ctc.decoding = decoding_cfg

            self.cur_decoder = "ctc"
            logging.info(
                f"Changed decoding strategy of the CTC decoder to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}"
            )
        else:
            raise ValueError(f"decoder_type={decoder_type} is not supported. Supported values: [ctc,rnnt]")
    
    @typecheck()
    def forward(
        self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None, lang_id=None
    ):
        """
        Forward pass of the model. Note that for RNNT Models, the forward pass of the model is a 3 step process,
        and this method only performs the first step - forward of the acoustic model.

        Please refer to the `training_step` in order to see the full `forward` step for training - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the loss and possibly compute the detokenized text via the `decoding` step.

        Please refer to the `validation_step` in order to see the full `forward` step for inference - which
        performs the forward of the acoustic model, the prediction network and then the joint network.
        Finally, it computes the decoded tokens via the `decoding` step and possibly compute the batch metrics.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 2 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        lang_vect = self.lang_embedding(lang_id)

        if self.insertion_location == "encoder":
            processed_signal, processed_signal_length = self._add_embedding(processed_signal, processed_signal_length, lang_vect)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            input_signal_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            input_signal_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), input_signal_eltype, optional=True),
            "input_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType(), optional=True),
            "processed_signal_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "lang_id": NeuralType(tuple('B'), LabelsType(), optional=True),
            "sample_id": NeuralType(tuple('B'), LengthsType(), optional=True)
        }

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, lang_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, lang_id=lang_id)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # TODO: add support for CTC decoding
        signal, signal_len, transcript, transcript_len, sample_id, lang_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, lang_id=lang_id)
        del signal

        best_hyp_text, all_hyp_text = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=False
        )

        sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp_text))

    def validation_pass(self, batch, batch_idx, dataloader_idx):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True)

        signal, signal_len, transcript, transcript_len, lang_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len, lang_id=lang_id)
        del signal

        tensorboard_logs = {}
        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )
            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value

        # CTC WER calculation
        self.ctc_wer.update(
            predictions=log_probs, targets=transcript, targets_lengths=transcript_len, predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        # BLEU score calculation
        self.bleu.update(
            predictions=encoded,
            predictions_lengths=encoded_len,
            targets=transcript,
            targets_lengths=transcript_len
        )
        bleu_metrics = self.bleu.compute(return_all_metrics=True, prefix="val_")
        tensorboard_logs.update({
            'val_bleu_num': bleu_metrics['val_bleu_num'],
            'val_bleu_denom': bleu_metrics['val_bleu_denom'],
            'val_bleu_pred_len': bleu_metrics['val_bleu_pred_len'],
            'val_bleu_target_len': bleu_metrics['val_bleu_target_len'],
            'val_bleu': bleu_metrics['val_bleu']
        })
        self.bleu.reset()

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        # Inter-CTC losses and additional logging
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)

        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        # Calculate validation loss if required
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}

        # Calculate WER
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}

        # Calculate CTC WER if applicable
        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        # Calculate BLEU score
        bleu_num = torch.stack([x['val_bleu_num'] for x in outputs]).sum(dim=0)
        bleu_denom = torch.stack([x['val_bleu_denom'] for x in outputs]).sum(dim=0)
        bleu_pred_len = torch.stack([x['val_bleu_pred_len'] for x in outputs]).sum(dim=0)
        bleu_target_len = torch.stack([x['val_bleu_target_len'] for x in outputs]).sum(dim=0)

        val_bleu = self.bleu._compute_bleu(bleu_pred_len, bleu_target_len, bleu_num, bleu_denom)
        tensorboard_logs['val_bleu'] = val_bleu

        # Finalize and log metrics
        metrics = {**val_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        
        return metrics


    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_en_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_de_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_fastconformer_hybrid_large_pc/versions/1.20.0/files/stt_it_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_es_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_hr_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_hr_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hr_fastconformer_hybrid_large_pc/versions/1.21.0/files/FastConformer-Hybrid-Transducer-CTC-BPE-v256-averaged.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ua_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ua_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ua_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_ua_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_pl_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_pl_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_pl_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_by_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_by_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_by_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_by_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_ru_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_fr_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu/versions/1.21.0/files/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_multilingual_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_multilingual_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_multilingual_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_multilingual_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_80ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_80ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_80ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_480ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_480ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_480ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_480ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_1040ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_1040ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_1040ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_1040ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_multi",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_multi",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_multi/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
        )
        results.append(model)

        return results
    def _add_embedding(self, audio, audio_len, lang):
        # Assumes audio is BxDxT
        input, input_len, lang = audio, audio_len, lang.unsqueeze(-1)

        if self.insertion_method == "concat":
            input = torch.concat([lang, audio], dim=2)
            input_len += 1
        else:
            raise NameError
        return input, input_len
        
    @property
    def bleu(self):
        return self._bleu

    @bleu.setter
    def bleu(self, bleu):
        self._bleu = bleu