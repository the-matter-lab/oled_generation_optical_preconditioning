import torch
from lightning import pytorch as pl
# from transformers import XLNetLMHeadModel, XLNetConfig
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Config
from weighted_methods import NashMTL

class OledModel(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.save_hyperparameters()
        # self.xlcfg = XLNetConfig(**cfg.xlnet_config)
        # self.model = XLNetLMHeadModel(self.xlcfg)
        self.gptcfg = GPT2Config(**cfg.gpt2_config)
        self.model = GPT2LMHeadModel(self.gptcfg)
        self.weight_method = NashMTL(5)

    def training_step(self, batch, batch_idx):
        if self.cfg.use_nash and self.training:
            losses = torch.empty(len(batch), device=self.device)
            for i, b in enumerate(batch.values()):
                model_outputs = self.model(**b)
                losses[i] = model_outputs.loss.mean()
            loss, _ = self.weight_method.backward(
                losses=losses,
                shared_parameters=list(self.model.parameters()),
            )
        else:
            loss = self.model(**batch['oled'],has_properties=True).loss
            loss += self.model(**batch['oled_unconditional']).loss
            loss += self.model(**batch['comp'],has_properties=True).loss
            loss += self.model(**batch['comp_unconditional']).loss
            loss += self.model(**batch['chembl']).loss
            # loss = self.model(**batch['chembl']).loss
            loss /= 5
        self.log("train_loss", loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("val_loss", loss, on_step=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.cfg.warmup_steps, self.cfg.warmup_steps*5)
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
# add to the end of OLEDModel class
    def to_lora(self):
        # FREEZE WEIGHTS
        for param in self.model.parameters():
            param.requires_grad = False

        # LoRa
        config = LoraConfig(
            r=128,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['c_fc', 'c_proj', 'c_attn']
        )
        self.model = get_peft_model(self.model, config)
