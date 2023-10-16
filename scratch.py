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
    "remove_rare_dir": True
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
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
    
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
# %%



# %%


# l1_coeff = 0.01
# encoder = AutoEncoder(d_mlp*cfg["dict_mult"], l1_coeff=cfg['l1_coeff']).cuda()
# loss, x_reconstruct, acts, l2_loss, l1_loss = encoder(sub)
# print(loss, l2_loss, l1_loss)

# loss.backward()
# print(encoder.W_dec.grad.norm())
# encoder.remove_parallel_component_of_grads()
# print(encoder.W_dec.grad.norm())
# print((encoder.W_dec.grad * encoder.W_dec).sum(-1))
# %%
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
# %%
encoder = AutoEncoder(cfg)
buffer = Buffer(cfg)
evil_dir = torch.load("/workspace/1L-Sparse-Autoencoder/evil_dir.pt")
evil_dir.requires_grad = False
# %%
try:
    wandb.init(project="autoencoder", entity="neelnanda-io")
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    # model_num_batches = cfg["model_batch_size"] * num_batches
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    recons_scores = []
    act_freq_scores_list = []
    for i in tqdm.trange(num_batches):
        i = i % all_tokens.shape[0]
        # tokens = all_tokens[i:i+cfg["model_batch_size"]]
        # acts = get_mlp_acts(tokens, batch_size=cfg["batch_size"])[0].detach()
        acts = buffer.next()
        loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
        loss.backward()
        encoder.remove_parallel_component_of_grads()
        if cfg["remove_rare_dir"]:
            with torch.no_grad():
                encoder.W_enc.grad -= (evil_dir @ encoder.W_enc.grad)[None, :] * evil_dir[:, None]
        encoder_optim.step()
        encoder_optim.zero_grad()
        if cfg["remove_rare_dir"]:
            with torch.no_grad():
                encoder.W_enc -= (evil_dir @ encoder.W_enc)[None, :] * evil_dir[:, None]
        loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}
        del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, acts
        if (i) % 100 == 0:
            wandb.log(loss_dict)
            print(loss_dict)
        if (i) % 1000 == 0:
            x = (get_recons_loss())
            print("Reconstruction:", x)
            recons_scores.append(x[0])
            freqs = get_freqs(5)
            act_freq_scores_list.append(freqs)
            # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
            wandb.log({
                "recons_score": x[0],
                "dead": (freqs==0).float().mean().item(),
                "below_1e-6": (freqs<1e-6).float().mean().item(),
                "below_1e-5": (freqs<1e-5).float().mean().item(),
            })
        if (i+1) % 30000 == 0:
            encoder.save()
            wandb.log({"reset_neurons": 0.0})
            freqs = get_freqs(50)
            to_be_reset = (freqs<10**(-5.5))
            print("Resetting neurons!", to_be_reset.sum())
            re_init(to_be_reset, encoder)
finally:
    encoder.save()

# %%

# # %%
# acts = get_mlp_acts(all_tokens[i:i+cfg["model_batch_size"]], batch_size=cfg["batch_size"])[0].detach()
# loss, x_reconstruct, mid_acts, l2_loss, l1_loss = encoder(acts)
# # %%
# acts.shape, x_reconstruct.shape
# acts.norm(dim=-1).mean(), x_reconstruct.norm(dim=-1).mean(), (acts - x_reconstruct).norm(dim=-1).mean()
# # %%
# line(encoder.W_enc[:, :20].T)
# line(encoder.W_dec[:20])
# # %%
# line(mid_acts.mean(0))
# # %%
freqs = get_freqs(50)
histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
freqs_5 = get_freqs(5)
scatter(x=freqs_5.log10(), y=freqs.log10())
# %%

# %%
(freqs<10**(-5.5)).float().mean()
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
freqs = get_freqs(50)
to_be_reset = (freqs<10**(-5.5))
re_init(to_be_reset, encoder)
# %%
x = (get_recons_loss())
print("Reconstruction:", x)
# recons_scores.append(x[0])
freqs = get_freqs(5)
# act_freq_scores_list.append(freqs)
histogram((freqs+10**(-6.5)).log10(), marginal="box", histnorm="percent", title="Frequencies")

# %%

# %%
enc2 = AutoEncoder.load(5)
tokens = all_tokens[:32]
acts = get_mlp_acts(tokens, batch_size=1)[1].detach()
# acts = buffer.next()
loss, x_reconstruct, mid_acts, l2_loss, l1_loss = enc2(acts)
print(loss, l2_loss, l1_loss)
# %%
freqs = (mid_acts>0).float().mean(0)
feature_df = pd.DataFrame({"freqs": to_numpy(freqs), "log_freq":to_numpy((freqs).log10())})
feature_df[feature_df["log_freq"]>-5]
# %%
f_id = 18
token_df = nutils.make_token_df(tokens)
token_df["act"] = to_numpy(mid_acts[:, f_id])
token_df["active"] = to_numpy(mid_acts[:, f_id]>0)
nutils.show_df(token_df.sort_values("act", ascending=False).head(100))

# %%
line(token_df.groupby("batch")["act"].mean())
line(token_df.groupby("batch")["active"].mean())
# %%1
act_freq_scores_list = []
encoders = {}
checkpoints = [25, 22, 21, 18, 15, 12, 9]
for i in checkpoints:
    print(i)
    encoders[i] = AutoEncoder.load(i)
    freqs = get_freqs(20, encoders[i])
    act_freq_scores_list.append(freqs)
    histogram((freqs+10**(-6.5)).log10(), marginal="box", histnorm="percent", title=f"Frequencies for checkpoint {i}")
# %%
def num_tokens_per_checkpoint(c):
    if c==25:
        # return "2000M"
        return 2000
    else:
        return int(((c - 8) * 30000 * 4096)/1e6)
line(x=list(range(26)), y=[num_tokens_per_checkpoint(c) for c in range(26)], title="Number of tokens per checkpoint")
freqs = torch.stack(act_freq_scores_list).flatten()
temp_df = pd.DataFrame({"freqs": to_numpy(freqs), "log_freq":to_numpy((freqs+10**(-6.5)).log10()), 
                        "checkpoint": [c for c in checkpoints for _ in range(encoders[25].d_hidden)],
                        "million_tokens": [num_tokens_per_checkpoint(c) for c in checkpoints for _ in range(encoders[25].d_hidden)],
                        })
px.histogram(temp_df, color="million_tokens", x="log_freq", barmode="overlay", marginal="box", histnorm="percent", title="Frequencies for checkpoints")
# %%
scatter(x=temp_df.query("checkpoint==21").log_freq, y=temp_df.query("checkpoint==22").log_freq, marginal_x="box", marginal_y="box", title="Frequencies for checkpoints 21 and 22", include_diag=True, xaxis=f"{num_tokens_per_checkpoint(21)}M", yaxis=f"{num_tokens_per_checkpoint(22)}M")
scatter(x=temp_df.query("checkpoint==21").log_freq, y=temp_df.query("checkpoint==25").log_freq, marginal_x="box", marginal_y="box", title="Frequencies for checkpoints 21 and 25", include_diag=True, xaxis=f"{num_tokens_per_checkpoint(21)}M", yaxis=f"{num_tokens_per_checkpoint(25)}M")
scatter(x=temp_df.query("checkpoint==22").log_freq, y=temp_df.query("checkpoint==25").log_freq, marginal_x="box", marginal_y="box", title="Frequencies for checkpoints 22 and 25", include_diag=True, xaxis=f"{num_tokens_per_checkpoint(22)}M", yaxis=f"{num_tokens_per_checkpoint(25)}M")
# %%
torch.set_grad_enabled(False)
# %%
tokens = all_tokens[:256]
_, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
mlp_acts = cache[utils.get_act_name("post", 0)]
mlp_acts_flattened = mlp_acts.reshape(-1, cfg["d_mlp"])
encoder = AutoEncoder.load(25)
hidden_acts = F.relu((mlp_acts_flattened - encoder.b_dec) @ encoder.W_enc + encoder.b_enc)
mlp_reconstr = hidden_acts @ encoder.W_dec + encoder.b_dec
l2_loss = (mlp_acts_flattened - mlp_reconstr).pow(2).sum(-1).mean(0)
l1_loss = encoder.l1_coeff * (hidden_acts.abs().sum())
print(l2_loss, l1_loss)
# %%
freqs = get_freqs(25, encoder)
# %%
histogram((freqs+10**-6.5).log10(), histnorm="percent", title="Frequencies for Final Checkpoint", xaxis="Freq (Log10)", yaxis="Percent")

# %%
is_rare = freqs < 1e-4


# %%


# %%

def replace_mlp_post(mlp_post, hook, replacement):
    mlp_post[:] = replacement
    return mlp_post
recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post, replacement=mlp_reconstr.reshape(mlp_acts.shape)))])
zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post, replacement=torch.zeros_like(mlp_acts)))])
normal_loss = model(tokens, return_type="loss")
mean_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post, replacement=mlp_acts.mean(0, keepdim=True).mean(1, keepdim=True)))])
print(f"{recons_loss.item()=}")
print(f"{zero_abl_loss.item()=}")
print(f"{normal_loss.item()=}")
print(f"{mean_loss.item()=}")
# %%
new_losses = []
min_freq_list = []
for thresh in [-6.5, -5, -4.5, -4.4, -4.3, -4.2, -4.1, -4, -3, -2.5, -2, -1.5, -1, 0]:
    indices = freqs >= 10**thresh
    replacement = hidden_acts[:, indices] @ encoder.W_dec[indices, :] + encoder.b_dec
    new_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post, replacement=replacement.reshape(mlp_acts.shape)))])
    new_losses.append(new_loss.item())
    min_freq_list.append(thresh)
new_losses = np.array(new_losses)
line(x=min_freq_list, y=new_losses, title="Loss vs minimum frequency")
line(x=min_freq_list, y=(zero_abl_loss.item() - new_losses)/(zero_abl_loss.item() - normal_loss.item()), title="Scaled Loss vs minimum frequency final checkpoint", yaxis="% Loss Recovered", xaxis="Log freq floor")
# %%
# encoder2 = AutoEncoder.load(21)
# hidden_acts = F.relu((mlp_acts_flattened - encoder2.b_dec) @ encoder2.W_enc + encoder2.b_enc)
# mlp_reconstr = hidden_acts @ encoder2.W_dec + encoder2.b_dec
# l2_loss = (mlp_acts_flattened - mlp_reconstr).pow(2).sum(-1).mean(0)
# l1_loss = encoder2.l1_coeff * (hidden_acts.abs().sum())
# print(l2_loss, l1_loss)

# freqs = get_freqs(25, encoder2)
# histogram((freqs+10**-6.5).log10(), barmode="overlay", marginal="box", histnorm="percent", title="Frequencies for checkpoint 21")

# %%
new_losses2 = []
min_freq_list2 = []
for thresh in [-6.5, -6, -5.5, -5, -4, -3, -2.5, -2, -1.5, -1, 0]:
    indices = freqs >= 10**thresh
    replacement = hidden_acts[:, indices] @ encoder2.W_dec[indices, :] + encoder2.b_dec
    new_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replace_mlp_post, replacement=replacement.reshape(mlp_acts.shape)))])
    new_losses2.append(new_loss.item())
    min_freq_list2.append(thresh)
new_losses2 = np.array(new_losses2)
line(x=min_freq_list2, y=new_losses2, title="Loss vs minimum frequency")
line(x=min_freq_list2, y=(zero_abl_loss.item() - new_losses2)/(zero_abl_loss.item() - normal_loss.item()), title="Scaled Loss vs minimum frequency checkpoint 21", yaxis="% Loss Recovered", xaxis="Log freq floor")

# %%
fig = line(x=min_freq_list, y=(zero_abl_loss.item() - new_losses)/(zero_abl_loss.item() - normal_loss.item()), title="Scaled Reconstructed Loss vs minimum frequency checkpoint", yaxis="% Loss Recovered", xaxis="Log freq floor", return_fig=True, line_labels=["Final checkpoint"])
fig.add_trace(go.Scatter(x=min_freq_list2, y=(zero_abl_loss.item() - new_losses2)/(zero_abl_loss.item() - normal_loss.item()), name="Checkpoint 21"))
# %%
def basic_feature_vis(text, feature_index, max_val=0):
    feature_in = encoder.W_enc[:, feature_index]
    feature_bias = encoder.b_enc[feature_index]
    _, cache = model.run_with_cache(text, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)][0]
    feature_acts = F.relu((mlp_acts - encoder.b_dec) @ feature_in + feature_bias)
    if max_val==0:
        max_val = max(1e-7, feature_acts.max().item())
        # print(max_val)
    # if min_val==0:
    #     min_val = min(-1e-7, feature_acts.min().item())
    return basic_token_vis_make_str(text, feature_acts, max_val)
def basic_token_vis_make_str(strings, values, max_val=None):
    if not isinstance(strings, list):
        strings = model.to_str_tokens(strings)
    values = to_numpy(values)
    if max_val is None:
        max_val = values.max()
    # if min_val is None:
    #     min_val = values.min()
    header_string = f"<h4>Max Range <b>{values.max():.4f}</b> Min Range: <b>{values.min():.4f}</b></h4>"
    header_string += f"<h4>Set Max Range <b>{max_val:.4f}</b></h4>"
    # values[values>0] = values[values>0]/ma|x_val
    # values[values<0] = values[values<0]/abs(min_val)
    body_string = nutils.create_html(strings, values, max_value=max_val, return_string=True)
    return header_string + body_string
display(HTML(basic_token_vis_make_str(tokens[0, :10], mlp_acts[0, :10, 7], 0.1)))
display(HTML(basic_feature_vis("I really like food food calories burgers eating is great", 7)))
# %%
# The `with gr.Blocks() as demo:` syntax just creates a variable called demo containing all these components
import gradio as gr
try:
    demos[0].close()
except:
    pass
demos = [None]
def make_feature_vis_gradio(batch, pos, feature_id):
    try:
        demos[0].close()
    except:
        pass
    with gr.Blocks() as demo:
        gr.HTML(value=f"Hacky Interactive Neuroscope for gelu-1l")
        # The input elements
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", value=model.to_string(tokens[batch, 1:pos+1]))
                # Precision=0 makes it an int, otherwise it's a float
                # Value sets the initial default value
                feature_index = gr.Number(
                    label="Feature Index", value=feature_id, precision=0
                )
                # # If empty, these two map to None
                max_val = gr.Number(label="Max Value", value=None)
                # min_val = gr.Number(label="Min Value", value=None)
                inputs = [text, feature_index, max_val]
        with gr.Row():
            with gr.Column():
                # The output element
                out = gr.HTML(label="Neuron Acts", value=basic_feature_vis(model.to_string(tokens[batch, 1:pos+1]), feature_id))
        for inp in inputs:
            inp.change(basic_feature_vis, inputs, out)
    demo.launch(share=True)
    demos[0] = demo
# %%
batch = 0
feature_id = 7
pos = 28
make_feature_vis_gradio(batch, pos, feature_id)
# %%

# %%
px.scatter(x=to_numpy(encoder.b_enc), y=to_numpy((freqs+10**-5).log10()), trendline="ols", labels={"x":"b_encoder", "y":"log10 freq", "color":"Is Rare"}, color=to_numpy(freqs<10**(-3.5)), title="Encoder bias vs frequency", marginal_x="histogram", marginal_y="histogram").show()
px.scatter(x=to_numpy(encoder.W_enc.norm(dim=0)), y=to_numpy((freqs+10**-5).log10()), trendline="ols", labels={"x":"W_encoder.norm", "y":"log10 freq", "color":"Is Rare"}, color=to_numpy(freqs<10**(-3.5)), title="Encoder norm vs frequency", marginal_x="histogram", marginal_y="histogram").show()
px.scatter(x=to_numpy(encoder.W_dec.norm(dim=-1)), y=to_numpy((freqs+10**-5).log10()), trendline="ols", labels={"x":"W_decoder.norm", "y":"log10 freq", "color":"Is Rare"}, color=to_numpy(freqs<10**(-3.5)), title="Decoder norm vs frequency", marginal_x="histogram", marginal_y="histogram").show()
px.scatter(x=to_numpy(encoder.b_enc * encoder.W_dec.norm(dim=-1)), y=to_numpy((freqs+10**-5).log10()), trendline="ols", labels={"x":"b_encoder * W_dec.norm", "y":"log10 freq", "color":"Is Rare"}, color=to_numpy(freqs<10**(-3.5)), title="Weighted encoder bias vs frequency", marginal_x="histogram", marginal_y="histogram").show()
px.scatter(x=to_numpy(encoder.W_enc.norm(dim=0) * encoder.W_dec.norm(dim=-1)), y=to_numpy((freqs+10**-5).log10()), trendline="ols", labels={"x":"W_enc.norm * W_dec.norm", "y":"log10 freq", "color":"Is Rare"}, color=to_numpy(freqs<10**(-3.5)), title="Encoder norm products vs frequency", marginal_x="histogram", marginal_y="histogram").show()
# %%
import huggingface_hub
from pathlib import Path
def push_to_hub(local_dir):
    if isinstance(local_dir, huggingface_hub.Repository):
        local_dir = local_dir.local_dir
    os.system(f"git -C {local_dir} add .")
    os.system(f"git -C {local_dir} commit -m 'Auto Commit'")
    os.system(f"git -C {local_dir} push")


# move_folder_to_hub("v235_4L512W_solu_wikipedia", "NeelNanda/SoLU_4L512W_Wiki_Finetune", just_final=False)
# def move_folder_to_hub(model_name, repo_name=None, just_final=True, debug=False):
#     if repo_name is None:
#         repo_name = model_name
#     model_folder = CHECKPOINT_DIR / model_name
#     repo_folder = CHECKPOINT_DIR / (model_name + "_repo")
#     repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
#     repo = huggingface_hub.Repository(repo_folder, repo_url)

#     for file in model_folder.iterdir():
#         if not just_final or "final" in file.name or "config" in file.name:
#             if debug:
#                 print(file.name)
#             file.rename(repo_folder / file.name)
#     push_to_hub(repo.local_dir)
def upload_folder_to_hf(folder_path, repo_name=None, debug=False):
    folder_path = Path(folder_path)
    if repo_name is None:
        repo_name = folder_path.name
    repo_folder = folder_path.parent / (folder_path.name + "_repo")
    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(repo_folder, repo_url)

    for file in folder_path.iterdir():
        if debug:
            print(file.name)
        file.rename(repo_folder / file.name)
    push_to_hub(repo.local_dir)
upload_folder_to_hf("/workspace/1L-Sparse-Autoencoder/checkpoints_copy_2", "sparse_autoencoder", True)
# %%
freqs = (hidden_acts>0).float().mean(0)
feature_df = pd.DataFrame({"freqs": to_numpy(freqs), "log_freq":to_numpy((freqs).log10())})
feature_df["is_common"] = feature_df["log_freq"]>-3.5
neuron_kurts = scipy.stats.kurtosis(to_numpy(encoder.W_enc))
feature_U = (encoder.W_dec @ model.W_out[0]) @ model.W_U
vocab_kurts = scipy.stats.kurtosis(to_numpy(feature_U.T))
feature_df["vocab_kurt"] = vocab_kurts
feature_df["neuron_kurt"] = neuron_kurts
neuron_frac_max = encoder.W_enc.max(dim=0).values / encoder.W_enc.abs().sum(0)
feature_df["neuron_frac_max"] = to_numpy(neuron_frac_max)
# %%
encoder2 = AutoEncoder.load(47)
freqs2 = get_freqs(5, encoder2)
is_common2 = freqs2>10**-3.5
is_common1 = freqs>10**-3.5
cosine_sims = nutils.cos_mat(encoder.W_enc.T, encoder2.W_enc[:, is_common2].T)
max_cosine_sim = cosine_sims.max(-1).values
feature_df["max_cos"] = to_numpy(max_cosine_sim)
feature_df
# %%
px.histogram(feature_df, x="neuron_kurt", marginal="box", color="is_common", histnorm="percent", title="Neuron Kurtosis", barmode="overlay", hover_name=feature_df.index).show()
px.histogram(feature_df, x="neuron_frac_max", marginal="box", color="is_common", histnorm="percent", title="Neuron Frac Max", barmode="overlay", hover_name=feature_df.index).show()
px.histogram(feature_df, x="vocab_kurt", marginal="box", color="is_common", histnorm="percent", title="Vocab Kurtosis", barmode="overlay", hover_name=feature_df.index).show()
px.scatter(feature_df, x="neuron_kurt", y="neuron_frac_max", hover_name=feature_df.index).show()
# %%
top_features = feature_df.sort_values("neuron_kurt", ascending=False).head(20).index.tolist()
line(encoder.W_enc[:, top_features].T, line_labels=top_features, title="Top Features in Neuron Basis", xaxis="Neuron")
# # %%
# f_id = 6
# token_df = nutils.make_token_df(tokens, 8)
# token_df["act"] = to_numpy(hidden_acts[:, f_id])
# token_df["active"] = to_numpy(hidden_acts[:, f_id]>0)
# token_df = token_df.sort_values("act", ascending=False)
# nutils.show_df(token_df.head(100))

# i = 0
# make_feature_vis_gradio(token_df.batch.iloc[i], token_df.pos.iloc[i], f_id)
# # %%
# is_rare = ~feature_df.is_common.values
# U, S, Vh = torch.linalg.svd(encoder.W_enc[:, ~is_rare])
# line(S, title="Singular Values of common features")
# histogram(U[:, :5], barmode="overlay", title="MLP side singular vectors common")
# histogram(Vh[:, :5], barmode="overlay", title="Feature side singular vectors common")

# U, S, Vh = torch.linalg.svd(encoder.W_enc[:, is_rare])
# line(S, title="Singular Values of rare features")
# histogram(U[:, :5], barmode="overlay", title="MLP side singular vectors rare")
# histogram(Vh[:, :5], barmode="overlay", title="Feature side singular vectors rare")

# token_df = nutils.make_token_df(tokens, 8, 3)
# token_df["rare_ave_feature"] = to_numpy(mlp_acts_flattened @ U[:, 0] * S[0] * 0.01)
# token_df["num_rare_active"] = to_numpy((hidden_acts[:, is_rare]>0).float().mean(-1))
# token_df["num_com_active"] = to_numpy((hidden_acts[:, ~is_rare]>0).float().mean(-1))
# token_df["mlp_act_norm"] = to_numpy(mlp_acts_flattened.norm(dim=-1))
# nutils.show_df(token_df.sort_values("rare_ave_feature", ascending=False).head(50))
# nutils.show_df(token_df.sort_values("rare_ave_feature", ascending=False).tail(50))
# svd_logit_lens = U[:, 0] @ model.W_out[0] @ model.W_U
# nutils.show_df(nutils.create_vocab_df(svd_logit_lens).head(20))
# # %%
# nutils.show_df(feature_df.query("is_common").head(20))
# %%
torch.set_grad_enabled(False)
f_id = 12
print(feature_df.loc[f_id])
tokens = all_tokens[:512]

_, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
mlp_acts = cache[utils.get_act_name("post", 0)]
mlp_acts_flattened = mlp_acts.reshape(-1, cfg["d_mlp"])
hidden_acts = F.relu((mlp_acts_flattened - encoder.b_dec) @ encoder.W_enc + encoder.b_enc)

token_df = nutils.make_token_df(tokens, 8)
token_df["act"] = to_numpy(hidden_acts[:, f_id])
token_df["active"] = to_numpy(hidden_acts[:, f_id]>0)
token_df = token_df.sort_values("act", ascending=False)
nutils.show_df(token_df.head(50))
hidden = hidden_acts[:, f_id].reshape(tokens.shape)
ave_firing = (hidden>0).float().mean(-1)
ave_act = (hidden).mean(-1)
big_fire_thresh = 0.2 * token_df.act.max()
ave_act_cond = (hidden).sum(-1) / ((hidden>0).float().sum(-1)+1e-7)
line([ave_firing, ave_act, ave_act_cond], line_labels=["Freq firing", "Ave act", "Ave act if firing"], title="Per batch summary statistics")

argmax_token = tokens.flatten()[hidden.flatten().argmax(-1).cpu()]
argmax_str_token = model.to_string(argmax_token)
print(argmax_token, argmax_str_token)
pos_token_df = token_df[token_df.act>0]
frac_of_fires_are_top_token = (pos_token_df.str_tokens==argmax_str_token).sum()/len(pos_token_df)
frac_big_firing_on_top_token = (pos_token_df.query(f"act>{big_fire_thresh}").str_tokens==argmax_str_token).sum()/len(pos_token_df.query(f"act>{big_fire_thresh}"))
frac_of_top_token_are_fires = (hidden.flatten().cpu()[tokens.flatten()==argmax_token]>0).float().mean().item()
print(f"{frac_of_fires_are_top_token=:.2%}")
print(f"{frac_big_firing_on_top_token=:.2%}")
print(f"{frac_of_top_token_are_fires=:.2%}")
print(f"Sample size = {(tokens.flatten()==argmax_token).sum().item()}")

line([encoder.W_enc[:, f_id], encoder.W_dec[f_id, :]], xaxis="Neuron", title="Weights in the neuron basis", line_labels=["encoder", "decoder"])

nutils.show_df(nutils.create_vocab_df(feature_U[f_id]).head(20))
nutils.show_df(nutils.create_vocab_df(feature_U[f_id]).tail(10))
i = 0
make_feature_vis_gradio(token_df.batch.iloc[i], token_df.pos.iloc[i], f_id)
# %%
# %%
strings = [
    "and| they|",
    "and| you|",
    "and| we|",
    "and| it|",
    "and| I|",
    "and| she|",
    "but| they|",
    "but| you|",
    "but| we|",
    "but| it|",
    "but| I|",
    "but| she|",
    "or| they|",
    "or| you|",
    "or| we|",
    "or| it|",
    "or| I|",
    "or| she|",
    # "but| they|",
    # "but| you|",
    # "but| we|",
    # "but| it|",
    # "but| I|",
    # "but| she|",
    ]
token_df["and_pronoun"] = [any(x in c for x in strings) for c in token_df.context]
px.histogram(token_df, x="act", marginal="box", color="and_pronoun", histnorm="percent", title="Neuron activation for pronouns", barmode="overlay", hover_name="context").show()
# %%
