from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
from layers.WaveletBlock import MultiScaleWaveletBlock
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.num_classes = getattr(configs, 'num_classes', 2)
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.use_wavelet = bool(getattr(configs, 'use_wavelet', False))
        self.wavelet_levels = int(getattr(configs, 'wavelet_level', 2))
        self.num_wave_bands = self.wavelet_levels + 1
        self.llm_model_id = getattr(configs, 'llm_model_id', None)
        self.llm_local_files_only = bool(getattr(configs, 'llm_local_files_only', False))

        if self.llm_model_id:
            self.auto_config = AutoConfig.from_pretrained(
                self.llm_model_id,
                trust_remote_code=True,
                local_files_only=self.llm_local_files_only
            )
            self.auto_config.output_attentions = True
            self.auto_config.output_hidden_states = True
            if hasattr(self.auto_config, 'num_hidden_layers'):
                self.auto_config.num_hidden_layers = configs.llm_layers
            try:
                self.llm_model = AutoModel.from_pretrained(
                    self.llm_model_id,
                    trust_remote_code=True,
                    local_files_only=self.llm_local_files_only,
                    config=self.auto_config
                )
            except EnvironmentError:
                self.llm_model = AutoModel.from_pretrained(
                    self.llm_model_id,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.auto_config
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_model_id,
                    trust_remote_code=True,
                    local_files_only=self.llm_local_files_only
                )
            except EnvironmentError:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_model_id,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.wavelet_block = MultiScaleWaveletBlock(
            levels=self.wavelet_levels, dropout=configs.dropout
        )
        self.task_to_id = {
            'long_term_forecast': 0,
            'short_term_forecast': 0,
            'forecast': 0,
            'anomaly_detection': 1,
            'classification': 2,
            'imputation': 3,
        }
        self.task_embedding = nn.Embedding(4, configs.d_model)
        self.task_band_logits = nn.Embedding(4, self.num_wave_bands)
        self.context_band_logits = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, self.num_wave_bands)
        )
        self.wavelet_gate = nn.Sequential(
            nn.Linear(configs.d_model * 3, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Sigmoid()
        )

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.classification_projection = nn.Linear(self.d_ff, self.num_classes)
        elif self.task_name == 'imputation':
            self.imputation_projection = nn.Linear(self.d_ff, 1)
        elif self.task_name == 'anomaly_detection':
            self.anomaly_projection = nn.Linear(self.d_ff, 1)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self._llm_backbone(
            x_enc,
            "forecast the next {} steps given the previous {} steps information".format(self.pred_len, self.seq_len),
            "forecast"
        )

        dec_out = torch.reshape(
            dec_out, (-1, x_enc.shape[-1], dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def classification(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self._llm_backbone(
            x_enc,
            "classify the category of this multivariate time series pattern",
            "classification"
        )
        dec_out = torch.reshape(
            dec_out, (-1, x_enc.shape[-1], dec_out.shape[-2], dec_out.shape[-1]))
        pooled = dec_out.mean(dim=1).mean(dim=1)  # [B, d_ff]
        logits = self.classification_projection(pooled)
        return logits

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self._llm_backbone(
            x_enc,
            "impute the missing values in this multivariate time series",
            "imputation"
        )
        dec_out = torch.reshape(
            dec_out, (-1, x_enc.shape[-1], dec_out.shape[-2], dec_out.shape[-1]))
        per_t = dec_out.mean(dim=1)  # [B, L, d_ff]
        per_t = per_t[:, -self.patch_nums:, :]
        per_t = per_t.transpose(1, 2)  # [B, d_ff, P]
        per_t = F.interpolate(per_t, size=self.seq_len, mode='linear', align_corners=False)
        per_t = per_t.transpose(1, 2)  # [B, seq_len, d_ff]
        imputed = self.imputation_projection(per_t)  # [B, seq_len, 1]
        if x_enc.shape[-1] > 1:
            imputed = imputed.repeat(1, 1, x_enc.shape[-1])  # [B, seq_len, N]
        imputed = self.normalize_layers(imputed, 'denorm')
        return imputed

    def anomaly_detection(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self._llm_backbone(
            x_enc,
            "detect whether each time point is anomalous in this multivariate time series",
            "anomaly_detection"
        )
        dec_out = torch.reshape(
            dec_out, (-1, x_enc.shape[-1], dec_out.shape[-2], dec_out.shape[-1]))
        per_t = dec_out.mean(dim=1)  # [B, L, d_ff]
        per_t = per_t[:, -self.patch_nums:, :]  # keep patch tokens
        per_t = per_t.transpose(1, 2)  # [B, d_ff, P]
        per_t = F.interpolate(per_t, size=self.seq_len, mode='linear', align_corners=False)
        per_t = per_t.transpose(1, 2)  # [B, seq_len, d_ff]
        logits = self.anomaly_projection(per_t)  # [B, seq_len, 1]
        return logits

    def _llm_backbone(self, x_enc, task_prompt, task_name):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc_1d = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc_1d, dim=1)[0]
        max_values = torch.max(x_enc_1d, dim=1)[0]
        medians = torch.median(x_enc_1d, dim=1).values
        lags = self.calcute_lags(x_enc_1d)
        trends = x_enc_1d.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc_1d.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: {task_prompt}; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        patch_input = x_enc.permute(0, 2, 1).contiguous()
        patch_dtype = torch.bfloat16 if patch_input.is_cuda else torch.float32
        enc_out, _ = self.patch_embedding(patch_input.to(patch_dtype))
        if self.use_wavelet:
            # True Haar-DWT bands + task-conditioned band weights.
            band_tensor = self.wavelet_block(x_enc)  # [B,T,N,num_bands]
            b, _, n, _ = band_tensor.shape
            task_id = self.task_to_id.get(task_name, 0)
            task_ids = torch.full((b,), task_id, device=x_enc.device, dtype=torch.long)
            base_logits = self.task_band_logits(task_ids)  # [B,num_bands]
            enc_group = enc_out.reshape(b, n, enc_out.shape[1], enc_out.shape[2])  # [B,N,P,d]
            context = enc_group.mean(dim=(1, 2)).float()  # [B,d_model]
            dynamic_logits = self.context_band_logits(context)  # [B,num_bands]
            band_weights = torch.softmax(base_logits + dynamic_logits, dim=-1)  # [B,num_bands]
            wave_fused = torch.einsum("btcn,bn->btc", band_tensor, band_weights)  # [B,T,N]
            wave_patch_input = wave_fused.permute(0, 2, 1).contiguous()  # [B,N,T]
            wave_patch, _ = self.patch_embedding(wave_patch_input.to(patch_dtype))
            task_embed = self.task_embedding(task_ids).repeat_interleave(n, dim=0).unsqueeze(1).expand(-1, enc_out.shape[1], -1)
            fusion_in = torch.cat([enc_out.float(), wave_patch.float(), task_embed], dim=-1)
            gate = self.wavelet_gate(fusion_in.float()).to(enc_out.dtype)
            enc_out = gate * enc_out + (1.0 - gate) * wave_patch
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
