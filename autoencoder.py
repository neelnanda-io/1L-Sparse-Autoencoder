# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
os.environ["DATASETS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *
# %%
import argparse
def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments, convert these to command line arguments, look at what was passed in, and return an updated dictionary.

    If in Ipython, just returns with no changes
    """
    if get_ipython() is not None:
        # Is in IPython
        print("In IPython - skipped argparse")
        return default_cfg
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) == bool:
            # argparse for Booleans is broken rip. Now you put in a flag to change the default --{flag} to set True, --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")

        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg
default_cfg = {
    "seed": 47,
    "batch_size": 1024,
    "model_batch_size": 128,
    "lr": 1e-4,
    "num_tokens": int(1e7),
    "l1_coeff": 3e-3,
    "wd": 1e-2,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 8,
    "seq_len": 128,
}
default_cfg["model_batch_size"] = default_cfg["batch_size"] // default_cfg["seq_len"] * 16
cfg = arg_parse_update_cfg(default_cfg)
pprint.pprint(cfg)
# %%

SEED = cfg["seed"]
GENERATOR = torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(True)

model = HookedTransformer.from_pretrained("gelu-1l")

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
@torch.no_grad()
def get_mlp_acts(tokens, batch_size=1024):
    _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)]
    mlp_acts = mlp_acts.reshape(-1, d_mlp)
    subsample = torch.randperm(mlp_acts.shape[0], generator=GENERATOR)[:batch_size]
    subsampled_mlp_acts = mlp_acts[subsample, :]
    return subsampled_mlp_acts, mlp_acts
sub, acts = get_mlp_acts(torch.arange(20).reshape(2, 10), batch_size=3)
sub.shape, acts.shape
# %%
class AutoEncoder(nn.Module):
    def __init__(self, d_hidden, l1_coeff, dtype=torch.float32, seed=47):
        super().__init__()
        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct - x).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj



l1_coeff = 0.01
encoder = AutoEncoder(d_mlp*cfg["dict_mult"], l1_coeff=cfg['l1_coeff']).cuda()
loss, x_reconstruct, acts, l2_loss, l1_loss = encoder(sub)
print(loss, l2_loss, l1_loss)

loss.backward()
print(encoder.W_dec.grad.norm())
encoder.remove_parallel_component_of_grads()
print(encoder.W_dec.grad.norm())
print((encoder.W_dec.grad * encoder.W_dec).sum(-1))
# %%
import wandb
# wandb.init(project="autoencoder", entity="neelnanda-io")
# # %%
# c4_urls = [f"https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.{i:0>5}-of-01024.json.gz" for i in range(901, 950)]

# dataset = load_dataset("json", data_files=c4_urls, split="train")

# dataset_name="c4"
# dataset.save_to_disk(f"/workspace/data/{dataset_name}_text.hf")
# # %%
# print(dataset)

# from transformer_lens.utils import tokenize_and_concatenate

# tokenizer = model.tokenizer

# tokens = tokenize_and_concatenate(dataset, tokenizer, streaming=False, num_proc=20, max_length=128)
# tokens.save_to_disk(f"/workspace/data/{dataset_name}_tokens.hf")
# %%
loading_data_first_time = False
if loading_data_first_time:
    data = load_dataset("NeelNanda/c4-code-tokenized-2b", split="train")
    data.save_to_disk("/workspace/data/c4_code_tokenized_2b.hf")
    data.set_format(type="torch", columns=["tokens"])
    all_tokens = data["tokens"]
    all_tokens.shape


    all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
    all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
    all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]
    torch.save(all_tokens_reshaped, "/workspace/data/c4_code_2b_tokens_reshaped.pt")
else:
    # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
    all_tokens = torch.load("/workspace/data/c4_code_2b_tokens_reshaped.pt")
# %%
num_batches = cfg["num_tokens"] // cfg["batch_size"]
model_num_batches = cfg["model_batch_size"] * num_batches
encoder_optim = torch.optim.AdamW(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]), weight_decay=cfg["wd"])
# wandb.init(project="autoencoder", entity="neelnanda-io")
for i in tqdm.trange(0, model_num_batches, cfg["model_batch_size"]):
    i = i % all_tokens.shape[0]
    tokens = all_tokens[i:i+cfg["model_batch_size"]]
    acts = get_mlp_acts(tokens, batch_size=cfg["batch_size"])[0].detach()
    loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
    loss.backward()
    encoder.remove_parallel_component_of_grads()
    encoder_optim.step()
    encoder_optim.zero_grad()
    loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
    del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, acts, tokens
    # wandb.log(loss_dict)
    if (i // cfg["model_batch_size"]) % 50 == 0:
        print(loss_dict)




# %%
acts = get_mlp_acts(all_tokens[i:i+cfg["model_batch_size"]], batch_size=cfg["batch_size"])[0].detach()
loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
# %%
acts.shape, x_reconstruct.shape
acts.norm(dim=-1).mean(), x_reconstruct.norm(dim=-1).mean(), (acts - x_reconstruct).norm(dim=-1).mean()
# %%
line(encoder.W_enc[:, :20].T)
line(encoder.W_dec[:20])
# %%
line(mid_acts.mean(0))
# %%
