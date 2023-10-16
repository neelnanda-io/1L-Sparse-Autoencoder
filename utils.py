# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
os.environ["DATASETS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *
import wandb
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
    "seed": 49,
    "batch_size": 4096,
    "buffer_mult": 384,
    "lr": 1e-4,
    "num_tokens": int(2e9),
    "l1_coeff": 3e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "dict_mult": 8,
    "seq_len": 128,
    "d_mlp": 2048,
    "enc_dtype":"fp32",
    "remove_rare_dir": False,
}
cfg = arg_parse_update_cfg(default_cfg)
cfg["model_batch_size"] = cfg["batch_size"] // cfg["seq_len"] * 16
cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]
pprint.pprint(cfg)
# %%

SEED = cfg["seed"]
GENERATOR = torch.manual_seed(SEED)
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(True)

model = HookedTransformer.from_pretrained("gelu-1l").to(DTYPES[cfg["enc_dtype"]])

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
# sub, acts = get_mlp_acts(torch.arange(20).reshape(2, 10), batch_size=3)
# sub.shape, acts.shape
# %%
SAVE_DIR = Path("/workspace/1L-Sparse-Autoencoder/checkpoints")
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to("cuda")
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
    
    def get_version(self):
        return 1+max([int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)])

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
        with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
        return self

    @classmethod
    def load_from_hf(cls, version):
        """
        Loads the saved autoencoder from HuggingFace. 
        
        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47
        
        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self

# %%



# %%
def shuffle_data(all_tokens):
    print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]

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
    all_tokens = shuffle_data(all_tokens)

# %%
class Buffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty. 
    """
    def __init__(self, cfg):
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["d_mlp"]), dtype=torch.bfloat16, requires_grad=False).cuda()
        self.cfg = cfg
        self.token_pointer = 0
        self.first = True
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.cfg["buffer_batches"]
            else:
                num_batches = self.cfg["buffer_batches"]//2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = all_tokens[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
                _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
                mlp_acts = cache[utils.get_act_name("post", 0)].reshape(-1, self.cfg["d_mlp"])
                # print(tokens.shape, mlp_acts.shape, self.pointer, self.token_pointer)
                self.buffer[self.pointer: self.pointer+mlp_acts.shape[0]] = mlp_acts
                self.pointer += mlp_acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]
                # if self.token_pointer > all_tokens.shape[0] - self.cfg["model_batch_size"]:
                #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).cuda()]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out

# buffer.refresh()
 # %%

# %%
def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

@torch.no_grad()
def get_recons_loss(num_batches=5, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replacement_hook, encoder=local_encoder))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss
# print(get_recons_loss())

# %%
# Frequency
@torch.no_grad()
def get_freqs(num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).cuda()
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        
        _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)]
        mlp_acts = mlp_acts.reshape(-1, d_mlp)

        hidden = local_encoder(mlp_acts)[2]
        
        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores
# %%
@torch.no_grad()
def re_init(indices, encoder):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(encoder.W_dec)))
    new_b_enc = (torch.zeros_like(encoder.b_enc))
    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    encoder.W_enc.data[:, indices] = new_W_enc[:, indices]
    encoder.W_dec.data[indices, :] = new_W_dec[indices, :]
    encoder.b_enc.data[indices] = new_b_enc[indices]