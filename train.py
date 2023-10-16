# %%
from utils import *
# %%
encoder = AutoEncoder(cfg)
buffer = Buffer(cfg)
# Code used to remove the "rare freq direction", the shared direction among the ultra low frequency features. 
# I experimented with removing it and retraining the autoencoder. 
if cfg["remove_rare_dir"]:
    rare_freq_dir = torch.load("rare_freq_dir.pt")
    rare_freq_dir.requires_grad = False

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
        encoder.make_decoder_weights_and_grad_unit_norm()
        if cfg["remove_rare_dir"]:
            with torch.no_grad():
                encoder.W_enc.grad -= (rare_freq_dir @ encoder.W_enc.grad)[None, :] * rare_freq_dir[:, None]
        encoder_optim.step()
        encoder_optim.zero_grad()
        if cfg["remove_rare_dir"]:
            with torch.no_grad():
                encoder.W_enc -= (rare_freq_dir @ encoder.W_enc)[None, :] * rare_freq_dir[:, None]
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
