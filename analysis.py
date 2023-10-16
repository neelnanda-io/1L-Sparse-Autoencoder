# %%
from utils import *
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
rare_enc = encoder.W_enc[:, is_rare]
rare_mean = rare_enc.mean(-1)

cos_with_mean = (rare_mean @ rare_enc) / rare_mean.norm() / rare_enc.norm(dim=0)
histogram(cos_with_mean, histnorm="percent", marginal="box", title="Cosine sim of rare features with mean rare direction", yaxis="percent", xaxis="Cosine Sim")
proj_onto_mean = (rare_mean @ rare_enc) / rare_mean.norm()
histogram(proj_onto_mean, histnorm="percent", marginal="box", title="Projection of rare features onto mean rare direction", yaxis="percent", xaxis="Projection")

print((cos_with_mean > 0.95).float().mean())

scatter(x=proj_onto_mean, y=(freqs[is_rare]+10**-6.5).log10())
scatter(x=cos_with_mean, y=(freqs[is_rare]+10**-6.5).log10())
# %%
encoder2 = AutoEncoder.load(47)
freqs2 = get_freqs(25, encoder2)
is_rare2 = freqs2 < 1e-4
rare_enc2 = encoder2.W_enc[:, is_rare2]
rare_mean2 = rare_enc2.mean(-1)
cos_with_mean2 = (rare_mean2 @ rare_enc2) / rare_mean2.norm() / rare_enc2.norm(dim=0)
histogram(cos_with_mean2, histnorm="percent", marginal="box", title="Cosine sim of rare features with mean rare direction", yaxis="percent", xaxis="Cosine Sim")
# %%
rare_mean2 @ rare_mean / rare_mean2.norm() / rare_mean.norm()
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

values, indices = (hidden_acts[:, is_rare]>0).float().mean(-1).sort()
# histogram((mlp_acts_flattened @ rare_mean))
token_df = nutils.make_token_df(tokens)
token_df["rare_proj"] = to_numpy(mlp_acts_flattened @ rare_mean)
token_df["frac_rare_active"] = to_numpy((hidden_acts[:, is_rare]>0).float().mean(-1))
sorted_token_df = token_df.sort_values("rare_proj", ascending=False)
for i in range(3):
    b = sorted_token_df.batch.iloc[i]
    p = sorted_token_df.pos.iloc[i]
    print(f"Frac rare features active on final token: {sorted_token_df.frac_rare_active.iloc[i]:.2%}")
    curr_tokens = tokens[b, :p+1]
    values = mlp_acts[b, :p+1] @ rare_mean
    nutils.create_html(model.to_str_tokens(curr_tokens), values)

# %%

feature_df = pd.DataFrame()
feature_kurtosis = scipy.stats.kurtosis(to_numpy(encoder.W_dec.T))
feature_df["neuron_kurt"] = to_numpy(feature_kurtosis)
feature_kurtosis_enc = scipy.stats.kurtosis(to_numpy(encoder.W_enc))
feature_df["neuron_kurt_enc"] = to_numpy(feature_kurtosis_enc)
feature_df["is_rare"] = to_numpy(is_rare)
# %%
sorted_W_enc = encoder.W_enc.abs().sort(dim=0).values
sorted_W_enc_sq = sorted_W_enc.pow(2)
sorted_W_enc_sq_sum = sorted_W_enc.pow(2).sum(0)
for k in [1, 2, 5, 10, 100]:
    feature_df[f"fve_top_{k}"] = to_numpy(sorted_W_enc_sq[-k:, :].sum(0) / sorted_W_enc_sq_sum)
feature_df["fve_next_9"] = feature_df["fve_top_10"] - feature_df["fve_top_1"]
feature_df.sort_values("neuron_kurt", ascending=False)
# %%
px.histogram(feature_df, x="fve_top_1", histnorm="percent", cumulative=True, marginal="box").show()
px.histogram(feature_df, x="fve_top_5", histnorm="percent", cumulative=True, marginal="box").show()
px.histogram(feature_df, x="fve_top_10", histnorm="percent", cumulative=True, marginal="box").show()
px.histogram(feature_df, x="fve_top_100", marginal="box").show()
# %%
px.scatter(feature_df.query("~is_rare"), x="fve_top_1", y="fve_next_9", hover_name=feature_df.query("~is_rare").index, color_continuous_scale="Portland", marginal_x="histogram", marginal_y="histogram", title="Fraction of Squared Sum Explained by Top Neuron vs Next 9 Neurons", opacity=0.2, labels={"fve_top_1":"Frac Explained by Top Neuron", "fve_next_9":"Frac Explained by Next 9 Neurons"})
# %%
sorted_W_dec = encoder.W_dec.T.abs().sort(dim=0).values
sorted_W_dec_sq = sorted_W_dec.pow(2)
sorted_W_dec_sq_sum = sorted_W_dec.pow(2).sum(0)
for k in [1, 2, 5, 10, 100]:
    feature_df[f"fve_top_{k}_dec"] = to_numpy(sorted_W_dec_sq[-k:, :].sum(0) / sorted_W_dec_sq_sum)
feature_df["fve_next_9_dec"] = feature_df["fve_top_10_dec"] - feature_df["fve_top_1_dec"]
px.scatter(feature_df.query("~is_rare"), x='fve_top_1', y="fve_top_1_dec").show()
# %%
px.scatter(feature_df.query("~is_rare"), x="fve_top_1_dec", y="fve_next_9_dec", hover_name=feature_df.query("~is_rare").index, color_continuous_scale="Portland", marginal_x="histogram", marginal_y="histogram", title="Fraction of Squared Sum Explained by Top Neuron vs Next 9 Neurons", opacity=0.2, labels={"fve_top_1_dec":"Frac Explained by Top Neuron", "fve_next_9_dec":"Frac Explained by Next 9 Neurons"})
# %%
feature_df["1-sparse"] = (feature_df["fve_top_1_dec"]>0.35) & (feature_df["fve_next_9_dec"]<0.1)
feature_df["10-sparse"] = (feature_df["fve_top_10_dec"]>0.35) & (~feature_df["1-sparse"])
print(f"Frac 1 sparse: {feature_df['1-sparse'].mean():.3f}")
print(f"Frac 10 sparse: {feature_df['10-sparse'].mean():.3f}")
def f(row):
    if row["1-sparse"]:
        return "1-sparse"
    elif row["10-sparse"]:
        return "10-sparse"
    else:
        return "dense"
feature_df["sparsity_label"] = feature_df.apply(f, axis=1)
px.scatter(feature_df.query("~is_rare"), x="fve_top_1_dec", y="fve_next_9_dec", hover_name=feature_df.query("~is_rare").index, color="sparsity_label", color_continuous_scale="Portland", marginal_x="histogram", marginal_y="histogram", title="Fraction of Squared Decoder Sum Explained by Top Neuron vs Next 9 Neurons", opacity=0.2, labels={"fve_top_1_dec":"Frac Explained by Top Neuron", "fve_next_9_dec":"Frac Explained by Next 9 Neurons"})
# %%
px.histogram(feature_df.query("~is_rare"), x="fve_top_1", log_y=True)
# %%
feature_df_baseline = pd.DataFrame()
rand_W_dec = torch.randn_like(encoder.W_dec.T)
feature_kurtosis = scipy.stats.kurtosis(to_numpy(rand_W_dec))
feature_df_baseline["neuron_kurt"] = to_numpy(feature_kurtosis)
feature_df_baseline["is_rare"] = to_numpy(is_rare)
# %%
sorted_W_dec = rand_W_dec.abs().sort(dim=0).values
sorted_W_dec_sq = sorted_W_dec.pow(2)
sorted_W_dec_sq_sum = sorted_W_dec.pow(2).sum(0)
for k in [1, 2, 5, 10, 100]:
    feature_df_baseline[f"fve_top_{k}"] = to_numpy(sorted_W_dec_sq[-k:, :].sum(0) / sorted_W_dec_sq_sum)
feature_df_baseline["fve_next_9"] = feature_df_baseline["fve_top_10"] - feature_df_baseline["fve_top_1"]
feature_df_baseline.sort_values("neuron_kurt", ascending=False)
# %%
px.histogram(feature_df_baseline, x="fve_top_1", histnorm="percent", cumulative=True, marginal="box").show()
px.histogram(feature_df_baseline, x="fve_top_5", histnorm="percent", cumulative=True, marginal="box").show()
px.histogram(feature_df_baseline, x="fve_top_10", histnorm="percent", cumulative=True, marginal="box").show()
px.histogram(feature_df_baseline, x="fve_top_100", marginal="box").show()
# %%
px.scatter(feature_df_baseline.query("~is_rare"), x="fve_top_1", y="fve_next_9", hover_name=feature_df_baseline.query("~is_rare").index, color_continuous_scale="Portland", marginal_x="histogram", marginal_y="histogram", title="")
# %%
temp_df = pd.concat([feature_df_baseline.query("~is_rare"), feature_df.query("~is_rare")])
temp_df = temp_df.reset_index(drop=True)
temp_df["category"] = ["baseline"]*(len(temp_df)//2) + ["real"]*(len(temp_df)//2)
px.histogram(temp_df, "neuron_kurt", color="category", barmode="overlay", marginal="box", range_x=(-5, 50), nbins=5000, title="Neuron Kurtosis (real vs random baseline, clipped at 50)")

# %%
feature_df["enc_dec_sim"] = to_numpy((encoder.W_dec * encoder.W_enc.T).sum(-1) / encoder.W_enc.norm(dim=0) / encoder.W_dec.norm(dim=-1))
px.histogram(feature_df.query("~is_rare"), x="enc_dec_sim", title="Encoder Decoder Cosine Sim (non-rare features)")
# %%
U, S, Vh = torch.linalg.svd((model.W_out[0]))
print(U.shape)
line(S)
# %%
W_enc_svd = encoder.W_enc.T @ U
W_enc_svd_null = W_enc_svd[:, 512:].pow(2).sum(-1)
W_enc_svd_all = W_enc_svd[:, :].pow(2).sum(-1)
feature_df["enc_null_frac"] = to_numpy(W_enc_svd_null / W_enc_svd_all)

W_dec_svd = encoder.W_dec @ U
W_dec_svd_null = W_dec_svd[:, 512:].pow(2).sum(-1)
W_dec_svd_all = W_dec_svd[:, :].pow(2).sum(-1)
feature_df["dec_null_frac"] = to_numpy(W_dec_svd_null / W_dec_svd_all)
# px.histogram(feature_df, x="enc_null_frac", color="is_rare", barmode="overlay").show()
# px.histogram(feature_df, x="dec_null_frac", color="is_rare", barmode="overlay").show()
# px.scatter(feature_df, x="dec_null_frac", y="enc_null_frac", color="is_rare").show()
fig = px.histogram(feature_df.query("~is_rare"), x=["enc_null_frac", "dec_null_frac"], barmode="overlay", marginal="box", title="Fraction of feature in W_out null space (non-rare)")
fig.add_vline(x=0.75, line_dash="dash", line_color="gray")
# %%
fig = px.histogram(feature_df.query("~is_rare"), x=["enc_null_frac", "enc_null_frac_baseline"], barmode="overlay", marginal="box", title="Fraction of feature in W_out null space (non-rare)")
fig.add_vline(x=0.75, line_dash="dash", line_color="gray").show()
fig = px.histogram(feature_df.query("~is_rare"), x=["dec_null_frac", "dec_null_frac_baseline"], barmode="overlay", marginal="box", title="Fraction of feature in W_out null space (non-rare)")
fig.add_vline(x=0.75, line_dash="dash", line_color="gray")

# %%
# print(feature_df["enc_null_frac_baseline"].mean())
# print(feature_df["enc_null_frac_baseline"].std())
# print(feature_df["dec_null_frac_baseline"].mean())
# print(feature_df["dec_null_frac_baseline"].std())

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
torch.set_grad_enabled(False)
feature_U = (encoder.W_dec @ model.W_out[0]) @ model.W_U
vocab_kurts = scipy.stats.kurtosis(to_numpy(feature_U.T))
feature_df["vocab_kurt"] = vocab_kurts
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
vocab_df = pd.DataFrame({
    "token": np.arange(d_vocab),
    "str_token": nutils.process_tokens(np.arange(d_vocab)),
})
vocab_df["is_upper"] = vocab_df["str_token"].apply(lambda s: s!=s.lower() and s==s.upper())
vocab_df["is_word"] = vocab_df["str_token"].apply(lambda s: s.replace(nutils.SPACE, "").isalpha())
vocab_df["has_space"] = vocab_df["str_token"].apply(lambda s: len(s)>0 and s[0]==nutils.SPACE)
vocab_df["is_capital"] = vocab_df["str_token"].apply(lambda s: len(s)>0 and ((s[0]==nutils.SPACE and s[1:]==s[1:].capitalize()) or (s[0]!=nutils.SPACE and s==s.capitalize())))
vocab_df
# %%
temp_df = copy.deepcopy(vocab_df)
def f(row):
    # print(row)
    if row.is_capital and row.has_space:
        return "Capital"
    elif not row.has_space and row.is_word:
        return "Fragment"
    else:
        return "Other"
temp_df["cond"] = temp_df.apply(f, axis=1)
temp_df["x"] = to_numpy(feature_U[f_id])
px.histogram(temp_df, x="x", color="is_capital", barmode="overlay", marginal="box", histnorm="percent", hover_name="str_token").show()
px.histogram(temp_df, x="x", color="cond", barmode="overlay", marginal="box", histnorm="percent", hover_name="str_token").show()
# %%
