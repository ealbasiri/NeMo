# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data.audio_to_text_lhotse_speaker import LhotseSpeechToTextSpkBpeDataset

import nemo.collections.asr.models as asr_models
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.parts.mixins.asr_adapter_mixins import ASRAdapterModelMixin
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.data.audio_to_text_dali import AudioToCharDALIDataset, DALIOutputs
from nemo.collections.asr.parts.utils.asr_multispeaker_utils import apply_spk_mapping

from nemo.collections.asr.parts.mixins import (
    ASRModuleMixin,
    ASRTranscriptionMixin,
    TranscribeConfig,
    TranscriptionReturnType,
)

from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.eesd_models import SortformerEncLabelModel
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo

from nemo.utils import logging

class EncDecRNNTSpkBPEModel(EncDecRNNTBPEModel):
    """Base class for encoder decoder RNNT-based models with subword tokenization."""

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        return results

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)
        
        if 'diar_model_path' in self.cfg:
            self.diar = True
            # Initialize the speaker branch
            self._init_diar_model()

            self.num_speakers = cfg.model_defaults.get('num_speakers', 4)
            
            # layer normalization, ln, l2, or None
            self.norm = cfg.get('norm', None)

            if cfg.norm == 'ln':
                self.asr_norm = torch.nn.LayerNorm(cfg.model_defaults.enc_hidden)
                self.diar_norm = torch.nn.LayerNorm(4)

            self.kernel_norm = cfg.get('kernel_norm',None)

            # projection layer
            self.diar_kernel_type = cfg.get('diar_kernel_type', None)

            proj_in_size = self.num_speakers + cfg.model_defaults.enc_hidden
            proj_out_size = cfg.model_defaults.enc_hidden
            self.joint_proj = torch.nn.Sequential(
                torch.nn.Linear(proj_in_size, proj_out_size*2),
                torch.nn.ReLU(),
                torch.nn.Linear(proj_out_size*2, proj_out_size)
            )
            self.diar_kernal = self.joint_proj

            if self.diar_kernel_type == 'sinusoidal':
                self.diar_kernel = self.get_sinusoid_position_encoding(self.num_speakers, cfg.model_defaults.enc_hidden)
            elif self.diar_kernel_type == 'metacat':
                # projection layer
                proj_in_size = self.num_speakers * cfg.model_defaults.enc_hidden
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj
            elif self.diar_kernel_type == 'metacat_residule' or self.diar_kernel_type == 'metacat_residule_early':
                # projection layer
                proj_in_size = cfg.model_defaults.enc_hidden
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj
            elif self.diar_kernel_type == 'metacat_residule_projection':
                proj_in_size = cfg.model_defaults.enc_hidden + self.num_speakers
                proj_out_size = cfg.model_defaults.enc_hidden
                self.joint_proj = torch.nn.Sequential(
                    torch.nn.Linear(proj_in_size, proj_out_size*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(proj_out_size*2, proj_out_size)
                )
                self.diar_kernel = self.joint_proj                

            #binarize diar_pred
            self.binarize_diar_preds_threshold = cfg.get('binarize_diar_preds_threshold', None)

        else:
            self.diar = False

    def _init_diar_model(self):
        """
        Initialize the speaker model.
        """

        model_path = self.cfg.diar_model_path
        # model_path = '/home/jinhanw/workdir/scripts/dataloader/pipeline/checkpoints/sortformer/im303a-ft7_epoch6-19.nemo'

        if model_path.endswith('.nemo'):
            pretrained_diar_model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        elif model_path.endswith('.ckpt'):
            pretrained_diar_model = SortformerEncLabelModel.load_from_checkpoint(model_path, map_location="cpu")
            logging.info("Diarization Model restored locally from {}".format(model_path))
        else:
            pretrained_diar_model = None
            logging.info("Model path incorrect")

        self.diarization_model = pretrained_diar_model

        if self.cfg.freeze_diar:
           self.diarization_model.eval()



    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextSpkBpeDataset(cfg = config, tokenizer=self.tokenizer,),
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
            'use_lhotse': True,
            'use_bucketing': False,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
            'num_speakers': self.cfg.test_ds.get('num_speakers',4),
            'spk_tar_all_zero': self.cfg.test_ds.get('spk_tar_all_zero',False),
            'num_sample_per_mel_frame': self.cfg.test_ds.get('num_sample_per_mel_frame',160),
            'num_mel_frame_per_asr_frame': self.cfg.test_ds.get('num_mel_frame_per_asr_frame',8),
            'shuffle_spk_mapping': self.cfg.test_ds.get('shuffle_spk_mapping',False),
            'inference_mode': self.cfg.test_ds.get('inference_mode', True)
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer
    
    def forward_diar(
        self,
        input_signal=None,
        input_signal_length=None,
        is_raw_waveform_input=True,
    ):
        preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.diarization_model.forward(audio_signal=input_signal, audio_signal_length=input_signal_length, is_raw_waveform_input=is_raw_waveform_input)

        return preds, _preds, attn_score_stack, total_memory_list, encoder_states_list

    def fix_diar_output(
        self,
        diar_pred,
        asr_frame_count
    ):
        """
        Duct-tapeß solution for extending the speaker predictions 
        """
        # Extract the first and last embeddings along the second dimension
        # first_emb = diar_pred[:, 0, :].unsqueeze(1)
        if diar_pred.shape[1] < asr_frame_count:
            last_emb = diar_pred[:, -1, :].unsqueeze(1)

            #number of repeatitions needed
            additional_frames = asr_frame_count - diar_pred.shape[1]

            # Create tensors of repeated first and last embeddings
            # first_repeats = first_emb.repeat(1, additional_frames // 2, 1)
            # last_repeats = last_emb.repeat(1, (additional_frames + 1) // 2, 1)
            last_repeats = last_emb.repeat(1, additional_frames, 1)

            # Concatenate the repeated tensors with the original embeddings
            # extended_diar_preds = torch.cat((first_repeats, diar_pred, last_repeats), dim=1)
            extended_diar_preds = torch.cat((diar_pred, last_repeats), dim=1)

            return extended_diar_preds
        else:
            # temporary solution if diar_pred longer than encoded
            return diar_pred[:, :asr_frame_count, :]

    
    def _get_probablistic_mix(self, diar_preds, spk_targets, rttm_mix_prob:float=0.0):
        """ 
        Sample a probablistic mixture of speaker labels for each time step then apply it to the diarization predictions and the speaker targets.
        
        Args:
            diar_preds (Tensor): Tensor of shape [B, T, D] representing the diarization predictions.
            spk_targets (Tensor): Tensor of shape [B, T, D] representing the speaker targets.
            
        Returns:
            mix_prob (float): Tensor of shape [B, T, D] representing the probablistic mixture of speaker labels for each time step.
        """
        batch_probs_raw = torch.distributions.categorical.Categorical(probs=torch.tensor([(1-rttm_mix_prob), rttm_mix_prob]).repeat(diar_preds.shape[0],1)).sample()
        batch_probs = (batch_probs_raw.view(diar_preds.shape[0], 1, 1).repeat(1, diar_preds.shape[1], diar_preds.shape[2])).to(diar_preds.device)
        batch_diar_preds = (1 - batch_probs) * diar_preds + batch_probs * spk_targets
        return batch_diar_preds 

    def get_sinusoid_position_encoding(self, max_position, embedding_dim):
        """
        Generates a sinusoid position encoding matrix.
        
        Args:
        - max_position (int): The maximum position to generate encodings for.
        - embedding_dim (int): The dimension of the embeddings.
        
        Returns:
        - torch.Tensor: A tensor of shape (max_position, embedding_dim) containing the sinusoid position encodings.
        """
        position = np.arange(max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
        
        position_encoding = np.zeros((max_position, embedding_dim))
        position_encoding[:, 0::2] = np.sin(position * div_term)
        position_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Convert the numpy array to a PyTorch tensor
        position_encoding_tensor = torch.tensor(position_encoding, dtype=torch.float32)
        
        return position_encoding_tensor


    def train_val_forward(self, batch, batch_nb):

        signal, signal_len, transcript, transcript_len, spk_targets, spk_mappings = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)

        encoded = torch.transpose(encoded, 1, 2) # B * D * T -> B * T * D

        if self.diar == True:
            if self.cfg.spk_supervision_strategy == 'rttm':
                if spk_targets is not None:
                    diar_preds = spk_targets 
                else:
                    raise ValueError("`spk_targets` is required for speaker supervision strategy 'rttm'")
            elif self.cfg.spk_supervision_strategy == 'diar':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(signal, signal_len)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device).half()
                if diar_preds is None:
                    raise ValueError("`diar_pred`is required for speaker supervision strategy 'diar'")
            elif self.cfg.spk_supervision_strategy == 'mix':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(signal, signal_len)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device)
                diar_preds = self._get_probablistic_mix(diar_preds=diar_preds, spk_targets=spk_targets, rttm_mix_prob=float(self.cfg.rttm_mix_prob))
            else:
                raise ValueError(f"Invalid RTTM strategy {self.cfg.spk_supervision_strategy} is not supported.")
        
            # Speaker mapping shuffling to equalize the speaker label's distributions
            if self.cfg.shuffle_spk_mapping:
                diar_preds = apply_spk_mapping(diar_preds, spk_mappings)

            if(diar_preds.shape[1]!=encoded.shape[1]):
            # KD duct-tape solution for extending the speaker predictions 
                asr_frame_count = encoded.shape[1]
                diar_preds = self.fix_diar_output(diar_preds, asr_frame_count)

            # Normalize the features
            if self.norm == 'ln':
                diar_preds = self.diar_norm(diar_preds)
                encoded = self.asr_norm(encoded)
            elif self.norm == 'l2':
                diar_preds = torch.nn.functional.normalize(diar_preds, p=2, dim=-1)
                encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)
            
            if diar_preds.shape[1] > encoded.shape[1]:
                diar_preds = diar_preds[:, :encoded.shape[1], :]

            if self.diar_kernel_type == 'sinusoidal':
                speaker_infusion_asr = torch.matmul(diar_preds, self.diar_kernel.to(diar_preds.device))
                if self.kernel_norm == 'l2':
                    speaker_infusion_asr = torch.nn.functional.normalize(speaker_infusion_asr, p=2, dim=-1)
                encoded = speaker_infusion_asr + encoded
            elif self.diar_kernel_type == 'metacat':
                concat_enc_states = encoded.unsqueeze(2) * diar_preds.unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(concat_enc_states)
            elif self.diar_kernel_type == 'metacat_residule':
                #only pick speaker 0
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = encoded + self.joint_proj(concat_enc_states)    
            elif self.diar_kernel_type == 'metacat_residule_early':
                #only pick speaker 0
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(encoded + concat_enc_states)
            elif self.diar_kernel_type == 'metacat_residule_projection':
                #only pick speaker 0 and add diar_preds
                concat_enc_states = encoded.unsqueeze(2) * diar_preds[:,:,:1].unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = encoded + concat_enc_states
                concat_enc_states = torch.cat([encoded, diar_preds], dim = -1)
                encoded = self.joint_proj(concat_enc_states)
            else:
                concat_enc_states = torch.cat([encoded, diar_preds], dim=-1)
                encoded = self.joint_proj(concat_enc_states)
        else:
            encoded = encoded
        
        encoded = torch.transpose(encoded, 1, 2) # B * T * D -> B * D * T
        return encoded, encoded_len, transcript, transcript_len


    # training_step include speaker information
    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        encoded, encoded_len, transcript, transcript_len = self.train_val_forward(batch, batch_nb)

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if (sample_id + 1) % log_every_n_steps == 0:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:
            # If experimental fused Joint-Loss-WER is used
            if (sample_id + 1) % log_every_n_steps == 0:
                compute_wer = True
            else:
                compute_wer = False

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

            # Reset access registry
            if AccessMixin.is_access_enabled(self.model_guid):
                AccessMixin.reset_registry(self)

            tensorboard_logs = {
                'train_loss': loss_value,
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}
    

    def validation_pass(self, batch, batch_idx, dataloader_idx=0):

        encoded, encoded_len, transcript, transcript_len = self.train_val_forward(batch, batch_idx)

        tensorboard_logs = {}

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

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        return tensorboard_logs

    def conformer_stream_step(
        self,
        audio_signal: torch.Tensor,
        processed_signal: torch.Tensor,
        spk_targets: torch.Tensor,
        global_mapping: dict,
        text_idx: int,
        token_idx: int,
        raw_texts: str,
        prev_spk: str,
        audio_signal_lengths: torch.Tensor = None,
        processed_signal_length: torch.Tensor = None,
        cache_last_channel: torch.Tensor = None,
        cache_last_time: torch.Tensor = None,
        cache_last_channel_len: torch.Tensor = None,
        keep_all_outputs: bool = True,
        previous_hypotheses: List[Hypothesis] = None,
        previous_pred_out: torch.Tensor = None,
        drop_extra_pre_encoded: int = None,
        return_transcription: bool = True,
        return_log_probs: bool = False,
    ):
        """
        For detailed usage, please refer to ASRModuleMixin.conformer_stream_step
        """
        if not isinstance(self, asr_models.EncDecRNNTModel) and not isinstance(self, asr_models.EncDecCTCModel):
            raise NotImplementedError(f"stream_step does not support {type(self)}!")

        if not isinstance(self.encoder, StreamingEncoder):
            raise NotImplementedError(f"Encoder of this model does not support streaming!")

        if isinstance(self, asr_models.EncDecRNNTModel) and return_transcription is False:
            logging.info(
                "return_transcription can not be False for Transducer models as decoder returns the transcriptions too."
            )

        if not isinstance(self, asr_models.EncDecCTCModel) and return_log_probs is True:
            logging.info("return_log_probs can only be True for CTC models.")
        (
            encoded,
            encoded_len,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
        ) = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=keep_all_outputs,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
        )
        encoded = torch.transpose(encoded, 1, 2) # B * D * T -> B * T * D

        if self.diar == True:
            if self.cfg.spk_supervision_strategy == 'rttm':
                if spk_targets is not None:
                    diar_preds = spk_targets 
                else:
                    raise ValueError("`spk_targets` is required for speaker supervision strategy 'rttm'")
            elif self.cfg.spk_supervision_strategy == 'diar':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(audio_signal, audio_signal_lengths)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device)
                    if diar_preds is None:
                        raise ValueError("`diar_pred`is required for speaker supervision strategy 'diar'")
            elif self.cfg.spk_supervision_strategy == 'mix':
                with torch.set_grad_enabled(not self.cfg.freeze_diar):
                    diar_preds, _preds, attn_score_stack, total_memory_list, encoder_states_list = self.forward_diar(audio_signal, audio_signal_lengths)
                    if self.binarize_diar_preds_threshold:
                        diar_preds = torch.where(diar_preds > self.binarize_diar_preds_threshold, torch.tensor(1), torch.tensor(0)).to(encoded.device)
                    diar_preds = self.fix_diar_output(diar_preds, spk_targets.shape[1])
                diar_preds = self._get_probablistic_mix(diar_preds=diar_preds, spk_targets=spk_targets, rttm_mix_prob=float(self.cfg.rttm_mix_prob))
            else:
                raise ValueError(f"Invalid RTTM strategy {self.cfg.spk_supervision_strategy} is not supported.")
        
            # Speaker mapping shuffling to equalize the speaker label's distributions
            if self.cfg.shuffle_spk_mapping:
                diar_preds = apply_spk_mapping(diar_preds, spk_mappings)

            if(diar_preds.shape[1]!=encoded.shape[1]):
            # KD duct-tape solution for extending the speaker predictions 
                asr_frame_count = encoded.shape[1]
                diar_preds = self.fix_diar_output(diar_preds, asr_frame_count)

            # Normalize the features
            if self.norm == 'ln':
                diar_preds = self.diar_norm(diar_preds)
                encoded = self.asr_norm(encoded)
            elif self.norm == 'l2':
                diar_preds = torch.nn.functional.normalize(diar_preds, p=2, dim=-1)
                encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)
            
            if diar_preds.shape[1] > encoded.shape[1]:
                diar_preds = diar_preds[:, :encoded.shape[1], :]

            if self.diar_kernel_type == 'sinusoidal':
                speaker_infusion_asr = torch.matmul(diar_preds, self.diar_kernel.to(diar_preds.device))
                if self.kernel_norm == 'l2':
                    speaker_infusion_asr = torch.nn.functional.normalize(speaker_infusion_asr, p=2, dim=-1)
                encoded = speaker_infusion_asr + encoded
            elif self.diar_kernel_type == 'metacat':
                concat_enc_states = encoded.unsqueeze(2) * diar_preds.unsqueeze(3)
                concat_enc_states = concat_enc_states.flatten(2,3)
                encoded = self.joint_proj(concat_enc_states)
            else:
                concat_enc_states = torch.cat([encoded, diar_preds], dim=-1)
                encoded = self.joint_proj(concat_enc_states)
        else:
            encoded = encoded
        
        encoded = torch.transpose(encoded, 1, 2) # B * T * 
        if isinstance(self, asr_models.EncDecCTCModel) or (
            isinstance(self, asr_models.EncDecHybridRNNTCTCModel) and self.cur_decoder == "ctc"
        ):
            if hasattr(self, "ctc_decoder"):
                decoding = self.ctc_decoding
                decoder = self.ctc_decoder
            else:
                decoding = self.decoding
                decoder = self.decoder
            log_probs = decoder(encoder_output=encoded)
            predictions_tensor = log_probs.argmax(dim=-1, keepdim=False)

            # Concatenate the previous predictions with the current one to have the full predictions.
            # We drop the extra predictions for each sample by using the lengths returned by the encoder (encoded_len)
            # Then create a list of the predictions for the batch. The predictions can have different lengths because of the paddings.
            greedy_predictions = []
            if return_transcription:
                all_hyp_or_transcribed_texts = []
            else:
                all_hyp_or_transcribed_texts = None
            for preds_idx, preds in enumerate(predictions_tensor):
                if encoded_len is None:
                    preds_cur = predictions_tensor[preds_idx]
                else:
                    preds_cur = predictions_tensor[preds_idx, : encoded_len[preds_idx]]
                if previous_pred_out is not None:
                    greedy_predictions_concat = torch.cat((previous_pred_out[preds_idx], preds_cur), dim=-1)
                    encoded_len[preds_idx] += len(previous_pred_out[preds_idx])
                else:
                    greedy_predictions_concat = preds_cur
                greedy_predictions.append(greedy_predictions_concat)

                # TODO: make decoding more efficient by avoiding the decoding process from the beginning
                if return_transcription:
                    decoded_out = decoding.ctc_decoder_predictions_tensor(
                        decoder_outputs=greedy_predictions_concat.unsqueeze(0),
                        decoder_lengths=encoded_len[preds_idx : preds_idx + 1],
                        return_hypotheses=False,
                    )
                    all_hyp_or_transcribed_texts.append(decoded_out[0][0])
            best_hyp = None
        else:
            best_hyp, all_hyp_or_transcribed_texts = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_len,
                return_hypotheses=True,
                partial_hypotheses=previous_hypotheses,
                # partial_hypotheses=None,
            )
            greedy_predictions = [hyp.y_sequence for hyp in best_hyp]
            if all_hyp_or_transcribed_texts is None:
                all_hyp_or_transcribed_texts = best_hyp

        '''
        spk mapping recovery from global rttm mapping
        '''

        # processing_text = all_hyp_or_transcribed_texts[0].text[text_idx:]
        # elements = processing_text.split()
        # output = []
        # for element in elements:
        #     if element == '<|spltoken0|>' and 0 in global_mapping.keys():
        #         element = f'<|spltoken{global_mapping[0]}|>'
        #     elif element == '<|spltoken1|>' and 1 in global_mapping.keys():
        #         element = f'<|spltoken{global_mapping[1]}|>'
        #     elif element == '<|spltoken2|>' and 2 in global_mapping.keys():
        #         element = f'<|spltoken{global_mapping[2]}|>'
        #     elif element == '<|spltoken3|>' and 3 in global_mapping.keys():
        #         element = f'<|spltoken{global_mapping[3]}|>'
        #     output.append(element)
        # if not raw_texts:
        #     raw_texts = ' '.join(output)
        # else:
        #     raw_texts = raw_texts +  ' ' + ' '.join(output)

        # merged_output = []
        # prev_spk = 'kkkk'
        # for i in range(len(output)):
        #     element = output[i]
        #     if element.find('spltoken') != -1:
        #         if element == prev_spk:
        #             continue
        #         else:
        #             prev_spk = element
        #     merged_output.append(element)
        # print(' '.join(merged_output))

        # import pdb; pdb.set_trace()
        # kkkkk = raw_texts.split()
        # kkkkk_sub = []
        # prev_spk = 'kkkkk'
        # for i in range(len(kkkkk)):
        #     element = kkkkk[i]
        #     if element.find('spltoken') != -1:
        #         if element == prev_spk:
        #             continue
        #         else:
        #             prev_spk = element
        #     kkkkk_sub.append(element)
        # print(' '.join(kkkkk_sub))
        # import pdb; pdb.set_trace()             

        '''
        spk mapping recovery from alignment
        '''

        # assert len(all_hyp_or_transcribed_texts[0].y_sequence[token_idx:]) == len(all_hyp_or_transcribed_texts[0].timestep)

        # #get all speaker token in y_sequence
        # spk_token_idx = []
        # for i in range(len(all_hyp_or_transcribed_texts[0].y_sequence[token_idx:])):
        #     if all_hyp_or_transcribed_texts[0].y_sequence[token_idx + i] in [1, 2, 3, 4]:
        #         spk_token_idx.append(i)

        # processing_text = all_hyp_or_transcribed_texts[0].text[text_idx:]
        # processing_time_step = all_hyp_or_transcribed_texts[0].timestep

        # spk_token_elements = []

        # for element in processing_text.split():
        #     if element.find('<|spltoken') != -1:
        #         spk_token_elements.append(element)
        # assert len(spk_token_idx) == len(spk_token_elements)
        # new_spk_token = []
        # for i in range(len(spk_token_idx)):
        #     spk_time_step = processing_time_step[spk_token_idx[i]]
        #     spk_tensor_hidden = spk_targets[0,spk_time_step, :]
        #     local_spk_token = torch.argmax(spk_tensor_hidden)# 0, 1, 2, 3
        #     try:
        #         new_spk_token.append(f'<|spltoken{global_mapping[int(local_spk_token.cpu().numpy())]}|>')
        #     except:
        #         new_spk_token.append(f'<|spltoken0|>')

        # output = []
        # j = 0
        # for element in processing_text.split():
        #     if element.find('<|spltoken') != -1:
        #         output.append(new_spk_token[j])
        #         j += 1
        #     else:
        #         output.append(element)
        # if not raw_texts:
        #     raw_texts = ' '.join(output)
        # else:
        #     raw_texts = raw_texts + ' ' + ' '.join(output)

        '''
        spk mapping recovery from alignment oracle
        '''
        assert len(all_hyp_or_transcribed_texts[0].y_sequence[token_idx:]) == len(all_hyp_or_transcribed_texts[0].timestep)

        #get all speaker token in y_sequence
        spk_token_idx = []
        for i in range(len(all_hyp_or_transcribed_texts[0].y_sequence[token_idx:])):
            if all_hyp_or_transcribed_texts[0].y_sequence[token_idx + i] in [1, 2, 3, 4]:
                spk_token_idx.append(i)

        processing_text = all_hyp_or_transcribed_texts[0].text[text_idx:]
        processing_time_step = all_hyp_or_transcribed_texts[0].timestep
        new_spk_token = []
        for i in range(len(spk_token_idx)):
            spk_time_step = processing_time_step[spk_token_idx[i]]
            spk_time_step_next = processing_time_step[spk_token_idx[i+1]] if i < len(spk_token_idx) - 1 else spk_targets.shape[1]
            spk_tensor_hidden = spk_targets[0,spk_time_step:spk_time_step_next+1,:]
            #get the most probable spk_token


            # #not considering overlap, always local spk0
            # local_spk_token = torch.argmax(torch.sum(spk_tensor_hidden, dim = 0))# 0, 1, 2, 3
            # try:
            #     new_spk_token.append(f'<|spltoken{global_mapping[int(local_spk_token.cpu().numpy())]}|>')
            # except:
            #     #because too long rttm segments
            #     import pdb; pdb.set_trace()
            #     new_spk_token.append(f'<|spltoken0|>')

            #if overlapped spk targets, use preceeding speaker
            spk_tensor_hidden_sum = torch.sum(spk_tensor_hidden, dim = 0)
            max_value = torch.max(spk_tensor_hidden_sum)
            local_spk_token = (spk_tensor_hidden_sum == max_value).nonzero(as_tuple=True)[0]
            assign_new_speaker = False

            if len(local_spk_token) > 1:
                k = processing_time_step[spk_token_idx[i]] - 1
                assign_new_speaker = True
                # import pdb; pdb.set_trace()
                while k >= 0 and len(local_spk_token)>1:
                    max_value = torch.max(spk_targets[0,k,:])
                    local_spk_token = (spk_targets[0,k,:] == max_value).nonzero(as_tuple=True)[0]
                    k -= 1
                if k>=0:
                    assign_new_speaker = False
                else:
                    assigned_spk = prev_spk
            if not assign_new_speaker:
                new_spk_token.append(f'<|spltoken{global_mapping[int(local_spk_token.cpu().numpy())]}|>')
            else:
                # import pdb; pdb.set_trace()
                new_spk_token.append(assigned_spk)
            prev_spk = new_spk_token[-1]

        output = []
        j = 0
        for element in processing_text.split():
            if element.find('<|spltoken') != -1:
                output.append(new_spk_token[j])
                j += 1
            else:
                output.append(element)
        if not raw_texts:
            raw_texts = ' '.join(output)
        else:
            raw_texts = raw_texts + ' ' + ' '.join(output)
            
        if len(new_spk_token) == 0:
            new_spk_token.append(prev_spk)
        # import pdb; pdb.set_trace()
        result = [
            greedy_predictions,
            all_hyp_or_transcribed_texts,
            raw_texts,
            cache_last_channel_next,
            cache_last_time_next,
            cache_last_channel_next_len,
            best_hyp,
            len(all_hyp_or_transcribed_texts[0].text),
            len(all_hyp_or_transcribed_texts[0].y_sequence),
            new_spk_token[-1],
        ]
        if return_log_probs:
            result.append(log_probs)
            result.append(encoded_len)

        return tuple(result)

    def _transcribe_forward(self, batch, trcfg:TranscribeConfig):
        #modify config 
        self.cfg.spk_supervision_strategy = 'diar'
        #forward pass
        encoded, encoded_len, _, _ = self.train_val_forward(batch, 0)
        output = dict(encoded=encoded, encoded_len=encoded_len)
        return output
    
    def _transcribe_output_processing(self, outputs,trcfg:TranscribeConfig):
        encoded = outputs.pop('encoded')
        encoded_len = outputs.pop('encoded_len')
        best_hyp, all_hyp = self.decoding.rnnt_decoder_predictions_tensor(
            encoded,
            encoded_len,
            return_hypotheses=True
        )
        # cleanup memory
        del encoded, encoded_len

        hypotheses = []
        all_hypotheses = []

        hypotheses += best_hyp
        if all_hyp is not None:
            all_hypotheses += all_hyp
        else:
            all_hypotheses += best_hyp

        return (hypotheses, all_hypotheses)

