# %%
from utils import *

# %%
board_seqs_int_train = t.load(
    os.path.join(
        section_dir,
        "data/board_seqs_int_train.pth",
    )
)
board_seqs_int_test = t.load(
    os.path.join(
        section_dir,
        "data/board_seqs_int_valid.pth",
    )
)

probes = {}
probe_names = ["linear", "flipped", "placed"]
for probe_name in probe_names:
    probe = []
    for layer in range(8):
        probe_in_layer = get_probe(layer, probe_name, "post")[0].detach()
        probe.append(probe_in_layer)
    probe : Float[Tensor, "layer d_model row col options"]= t.stack(probe, dim=0)
    probes[probe_name] = probe

# %%
START_VALID = 0
NUM_GAMES_VALID = 10000
BATCH_SIZE = 500

PRED_LOGITS = 0 # 0.1 orde 1 oder 10 wäre auch interessant (also für diff machts keinen unterschied aber für final schon)

def bundle_fake_cache(fake_cache):
    bundled_fake_cache = {}
    key_names = list(set([".".join(key.split(".")[2:]) for key in fake_cache]))
    for key_name in key_names:
        act_name_results = {act_name : result for act_name, result in fake_cache.items() if key_name in act_name}
        stacked_result : Float[Tensor, "batch layer pos neurons"] = t.stack(list(act_name_results.values()), dim=1)
        bundled_fake_cache[key_name] = stacked_result
    return bundled_fake_cache

# %%
W_out = model.W_out.detach()
b_out = model.b_out.detach()
b_out_repeat = einops.repeat(b_out, "layer d_model -> layer pos d_model", pos=59)

def get_masks(flipped_final_real, flipped_final_pred):
    mask = t.zeros_like(flipped_final_real).to(dtype=t.int)
    only_real_mask = t.zeros_like(flipped_final_real).to(dtype=t.int)
    only_pred_mask = t.zeros_like(flipped_final_real).to(dtype=t.int)
    mask[:, 0] = flipped_final_real[:, 0]
    for layer in range(1, 8):
        change_real = (flipped_final_real[:, layer - 1] != flipped_final_real[:, layer])
        change_pred = (flipped_final_real[:, layer - 1] != flipped_final_pred[:, layer]) # This is correct!
        mask[:, layer] = (change_real | change_pred).to(dtype=t.int)
        only_real_mask[:, layer] = change_real.to(dtype=t.int)
        only_pred_mask[:, layer] = change_pred.to(dtype=t.int)
    return mask, only_real_mask, only_pred_mask

def evaluate_rules(bundled_fake_cache_valid, avg_mlp_post):
    results_dict = {
        "avg_neuron_count" : t.zeros(8, 59).to(device),
        "abs_mean_diff_flipped" : t.zeros(8, 59, 8, 8).to(device),
        "abs_mean_diff_not_flipped" : t.zeros(8, 59, 8, 8).to(device),
        "TP_diff" : t.zeros(8, 59, 8, 8).to(device),
        "FP_diff" : t.zeros(8, 59, 8, 8).to(device),
        "TN_diff" : t.zeros(8, 59, 8, 8).to(device),
        "FN_diff" : t.zeros(8, 59, 8, 8).to(device),
        "TP_final" : t.zeros(8, 59, 8, 8).to(device),
        # "FP_final" : t.zeros(8, 59, 8, 8).to(device),
        "TN_final" : t.zeros(8, 59, 8, 8).to(device),
        # "FN_final" : t.zeros(8, 59, 8, 8).to(device),
        "TP_normal" : t.zeros(8, 59, 8, 8).to(device),
        "FP_normal" : t.zeros(8, 59, 8, 8).to(device),
        "TN_normal" : t.zeros(8, 59, 8, 8).to(device),
        "FN_normal" : t.zeros(8, 59, 8, 8).to(device),
    }
    total_number_of_neurons = t.zeros(8, 59).to(device)
    total_number_of_predictions = t.zeros(8, 59).to(device)
    mask_sum = t.zeros(8, 59, 8, 8, device=device)
    mask_normal_sum = t.zeros(8, 59, 8, 8, device=device)

    probe = probes["flipped"]
    probe_lists = {}
    for layer in range(8):
        probe_lists[layer] = []
        for row in range(8):
            for col in range(8):
                probe_lists[layer].append(probe[layer, :, row, col, FLIPPED])
    linear_probe = probes["linear"]
        
    for batch in tqdm(range(START_VALID, START_VALID + NUM_GAMES_VALID, BATCH_SIZE)):
        mlp_post_real : Float[Tensor, "batch layer pos neurons"] = bundled_fake_cache_valid["mlp.hook_post"][batch:batch+BATCH_SIZE].to(device)
        resid_pre_real : Float[Tensor, "batch layer pos d_model"] = bundled_fake_cache_valid["hook_resid_pre"][batch:batch+BATCH_SIZE].to(device)
        attn_out_real : Float[Tensor, "batch layer pos d_model"] = bundled_fake_cache_valid["hook_attn_out"][batch:batch+BATCH_SIZE].to(device)

        mlp_out_real = einops.einsum(mlp_post_real, W_out, "batch layer pos neurons, layer neurons d_model -> batch layer pos d_model")
        flipped_logits_real = einops.einsum(mlp_out_real + attn_out_real + b_out_repeat, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_real = flipped_logits_real.argmax(dim=-1)

        if PRED_LOGITS == 0:
            num_games = mlp_post_real.shape[0]
            avg_mlp_post_new = einops.repeat(avg_mlp_post, "layer pos neurons -> batch layer pos neurons", batch=num_games)
            mlp_out_pred = einops.einsum(avg_mlp_post_new, W_out, "batch layer pos neurons, layer neurons d_model -> batch layer pos d_model")
            flipped_logits_pred = einops.einsum(mlp_out_pred + attn_out_real + b_out_repeat, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        else:
            flipped_logits_pred = t.ones_like(flipped_logits_real).to(device)
            flipped_logits_pred[:, :, :, :, :, FLIPPED] *= PRED_LOGITS
            flipped_logits_pred[:, :, :, :, :, NOT_FLIPPED] *= -PRED_LOGITS
        flipped_pred = flipped_logits_pred.argmax(dim=-1)

        resid_post_real = resid_pre_real + mlp_out_real + attn_out_real + b_out_repeat
        final_logits_real = einops.einsum(resid_post_real, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        flipped_final_real = (final_logits_real[:, :, :, :, :, 0] > final_logits_real[:, :, :, :, :, 1]).to(t.int)
        if PRED_LOGITS == 0:
            resid_post_pred = resid_pre_real + mlp_out_pred + attn_out_real + b_out_repeat
            final_logits_pred = einops.einsum(resid_post_pred, probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        else:
            final_logits_pred = t.ones_like(final_logits_real).to(device)
            final_logits_pred[:, :, :, :, :, FLIPPED] *= PRED_LOGITS
            final_logits_pred[:, :, :, :, :, NOT_FLIPPED] *= -PRED_LOGITS
        flipped_final_pred = (final_logits_pred[:, :, :, :, :, 0] > final_logits_pred[:, :, :, :, :, 1]).to(t.int)
        mask, only_real_mask, only_pred_mask = get_masks(flipped_final_real, flipped_final_pred)
        mask_sum += mask.sum(dim=0)

        abs_diff_flipped = (flipped_logits_real[:, :, :, :, :, FLIPPED] - flipped_logits_pred[:, :, :, :, :, FLIPPED]).abs().sum(dim=0)
        abs_diff_not_flipped = (flipped_logits_real[:, :, :, :, :, NOT_FLIPPED] - flipped_logits_pred[:, :, :, :, :, NOT_FLIPPED]).abs().sum(dim=0)
        TP = ((flipped_real == FLIPPED) & (flipped_pred == FLIPPED) & mask).sum(dim=0).float()
        FP = ((flipped_real == NOT_FLIPPED) & (flipped_pred == FLIPPED) & mask).sum(dim=0).float()
        TN = ((flipped_real == NOT_FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask).sum(dim=0).float()
        FN = ((flipped_real == FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask).sum(dim=0).float()
        results_dict["abs_mean_diff_flipped"] += abs_diff_flipped
        results_dict["abs_mean_diff_not_flipped"] += abs_diff_not_flipped
        results_dict["TP_diff"] += TP
        results_dict["FP_diff"] += FP
        results_dict["TN_diff"] += TN
        results_dict["FN_diff"] += FN

        flipped_change_real = (flipped_real == FLIPPED) & only_real_mask
        not_flipped_change_real = (flipped_real == NOT_FLIPPED) & only_real_mask
        flipped_change_pred = (flipped_pred == FLIPPED) & only_pred_mask
        not_flipped_change_pred = (flipped_pred == NOT_FLIPPED) & only_pred_mask
        TP_final = (flipped_change_real & flipped_change_pred).sum(dim=0).float()
        FP_final = (not_flipped_change_real & flipped_change_pred).sum(dim=0).float()
        TN_final = (not_flipped_change_real & not_flipped_change_pred).sum(dim=0).float()
        FN_final = (flipped_change_real & not_flipped_change_pred).sum(dim=0).float()
        # DICLAIMER: False Positive and False Negative where not done write, but I can get accuracy using mask_sum ...
        results_dict["TP_final"] += TP_final
        # results_dict["FP_final"] += FP_final
        results_dict["TN_final"] += TN_final
        # results_dict["FN_final"] += FN_final

        linear_logits = einops.einsum(resid_post_real, linear_probe, "batch layer pos d_model, layer d_model row col options -> batch layer pos row col options")
        mask_normal = linear_logits.argmax(dim=-1) != EMPTY
        mask_normal_sum += mask_normal.sum(dim=0)
        TP_normal = ((flipped_real == FLIPPED) & (flipped_pred == FLIPPED) & mask_normal).sum(dim=0).float()
        FP_normal = ((flipped_real == NOT_FLIPPED) & (flipped_pred == FLIPPED) & mask_normal).sum(dim=0).float()
        TN_normal = ((flipped_real == NOT_FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask_normal).sum(dim=0).float()
        FN_normal = ((flipped_real == FLIPPED) & (flipped_pred == NOT_FLIPPED) & mask_normal).sum(dim=0).float()
        results_dict["TP_normal"] += TP_normal
        results_dict["FP_normal"] += FP_normal
        results_dict["TN_normal"] += TN_normal
        results_dict["FN_normal"] += FN_normal

    results_dict["abs_mean_diff_flipped"] /= NUM_GAMES_VALID
    results_dict["abs_mean_diff_not_flipped"] /= NUM_GAMES_VALID
    results_dict["avg_neuron_count"] = total_number_of_neurons / total_number_of_predictions
    return results_dict, mask_sum, mask_normal_sum

def get_avg_mlp_over_pos(num_games = 1000):
    act_names = [utils.get_act_name("mlp_post", layer) for layer in range(8)]
    fake_cache = get_activation(board_seqs_int_train, act_names, num_games=num_games, start=10000)
    avg_mlp_post = t.stack([fake_cache[act_name].to(device) for act_name in act_names])
    avg_mlp_post = avg_mlp_post.mean(dim=1)
    avg_mlp_post_over_pos = avg_mlp_post.mean(dim=1)
    return avg_mlp_post_over_pos, avg_mlp_post

if __name__ == "__main__":
    # take 1 keyword argument PRED_LOGITS
    '''if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "flipping":
            PRED_LOGITS = 10
        elif mode == "not_flipping":
            PRED_LOGITS = -10
        elif mode == "none":
            PRED_LOGITS = 0
        else:
            print("Invalid mode, using default 0")
    else:
        print("No PRED_LOGITS given, using default 2")'''
    PRED_LOGITS = 0
    act_names = [utils.get_act_name("mlp_post", layer) for layer in range(8)]
    act_names += [utils.get_act_name("attn_out", layer) for layer in range(8)]
    act_names += [utils.get_act_name("resid_pre", layer) for layer in range(8)]
    act_names += [utils.get_act_name("resid_post", layer) for layer in range(8)]
    act_names += [f"blocks.{layer}.ln1.hook_normalized" for layer in range(8)]
    act_names += [f"blocks.{layer}.ln2.hook_normalized" for layer in range(8)]
    print("Get Fake Cache Valid")
    fake_cache_valid = get_activation(board_seqs_int_test, act_names, start=START_VALID, num_games=NUM_GAMES_VALID)
    bundled_fake_cache_valid = bundle_fake_cache(fake_cache_valid)
    print("Evaluate")
    avg_mlp_post_over_pos, avg_mlp_post = get_avg_mlp_over_pos(1000)
    results_dict, mask_sum, mask_normal_sum = evaluate_rules(bundled_fake_cache_valid, avg_mlp_post)
    print("Save")
    directory = "flipping_circuit_results"
    run_name = f"baseline_{PRED_LOGITS}"
    save_path = f"{directory}/{run_name}"
    os.makedirs(save_path, exist_ok=True)
    t.save(results_dict, f"{save_path}/flipped_circuit_results_dict.pth")
    t.save(mask_sum, f"{save_path}/flipped_circuit_mask_sum.pth")
    t.save(mask_normal_sum, f"{save_path}/flipped_circuit_mask_normal_sum.pth")
    print("Saved! and Done!")


