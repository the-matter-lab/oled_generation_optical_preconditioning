import numpy as np
from ml_collections import ConfigDict

NUM_BINS = dict(
    strength=4,
    absorption=4,
    splitting=4, # actually gap
)

BINS = dict(
    strength=np.array([0.00000000e+00, 5.52315881e-04, 4.80000000e-03, 3.05000000e-02, 3.21600000e-01, 5.79904384e+00]),
    absorption=np.array([ 0.049 ,  2.7575,  3.0675,  3.3489,  3.671 , 12.796 ]),
    splitting=np.array([ 1.14075568,  6.20877618,  6.56544554,  6.92625786,  7.35424152,  42.863594  ]),
)


def get_config():
    cfg = ConfigDict()
    cfg.name = "mixed_gpt2_smiles_gas_nash"
    cfg.use_nash = True
    cfg.has_properties = True # flag for collator
    cfg.seed = 42
    cfg.num_bins = NUM_BINS
    cfg.n_aug = 2
    cfg.batch_size = 1 # this is per class (e.g., 7+4+7+7=25 classes)
    cfg.lr = 1e-5
    cfg.warmup_steps = 50000
    cfg.max_epochs = 20
    cfg.clip_norm_value = 1.0
    cfg.num_workers = 12
    cfg.debugging = False
    cfg.load_model = ''
    
    # for plm
    cfg.selfies_index = 5 # = 1 bos + 4 ppty tokens 
    cfg.plm_probability = 0.2# 0.4
    cfg.max_span_length = 4#7
    
    cfg.val_bins = BINS
    
    # cfg.xlnet_config = {
    #     "architectures": [
    #         "XLNetLMHeadModel"
    #     ],
    #     "d_model": 256,
    #     "d_head": 16,
    #     "d_inner": 1024,
    #     "n_head": 16,
    #     "n_layer": 32,
    #     "mem_len": None,
    #     "dropout": 0.2,
    #     "vocab_size": -1,  # to be set in main.py
    #     "bos_token_id": -1, # to be set in main.py
    #     "eos_token_id": -1, # to be set in main.py
    #     "pad_token_id": -1,  # to be set in main.py
    #     "task_specific_params": {
    #         "text-generation": {
    #         "do_sample": True,
    #         "max_length": 250
    #         }
    #     }
    # }
    
    cfg.gpt2_config = {
        "vocab_size": -1,  # to be set in main.py
        "bos_token_id": -1, # to be set in main.py
        "eos_token_id": -1, # to be set in main.py
        "pad_token_id": -1,  # to be set in main.py
        "n_positions": 1024,
        "n_layer": 12, #16
        "n_head": 12, #6,
        "n_embd": 768,
        "add_cross_attention": False,
        "max_length": 250,
    }
    
    cfg.trainer = trainer = ConfigDict()
    trainer.accelerator="auto"
    trainer.max_epochs=cfg.max_epochs
    # trainer.max_steps=1000000
    trainer.val_check_interval=2000
    # trainer.check_val_every_n_epoch=1
    trainer.gradient_clip_val=cfg.clip_norm_value
    trainer.accumulate_grad_batches=8
    trainer.limit_val_batches=50
    # trainer.limit_train_batches=1000
    trainer.log_every_n_steps=100
    trainer.enable_progress_bar=False
    # trainer.devices=4
    
    return cfg
