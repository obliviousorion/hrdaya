"""Export champion model weights, test beats, and enrolled templates to C arrays."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlee.resnet_se_model import ECG_ResNet_SE  # noqa: E402


def fuse_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    # W_fused = W * (gamma / sqrt(var + eps))
    # B_fused = (B - mean) * (gamma / sqrt(var + eps)) + beta
    scale = bn_w / torch.sqrt(bn_var + eps)
    if conv_w.dim() == 3:  # (out_ch, in_ch, k)
        w_fused = conv_w * scale.view(-1, 1, 1)
    else:
        w_fused = conv_w * scale.view(-1, 1)
    b_fused = (conv_b - bn_mean) * scale + bn_b
    return w_fused.detach().cpu().numpy(), b_fused.detach().cpu().numpy()


def main():
    private = ROOT / "private"
    out_dir = ROOT / "hardware" / "firmware"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load FP32 model for clean BatchNorm folding
    fp32_path = private / "resnet_se_hybrid_fp32.pt"
    n_classes = 89
    model = ECG_ResNet_SE(n_classes=n_classes, emb_dim=48)
    model.load_state_dict(torch.load(fp32_path, map_location="cpu", weights_only=True))
    model.eval()

    sd = model.state_dict()

    # Fuse Stem Conv + BN
    w_stem, b_stem = fuse_conv_bn(
        sd["stem.0.weight"], sd["stem.0.bias"],
        sd["stem.1.weight"], sd["stem.1.bias"],
        sd["stem.1.running_mean"], sd["stem.1.running_var"]
    )

    # Fuse Block 1
    # conv1 is depthwise (24, 1, 7) - no BN immediately following conv1, BN is after pointwise1
    w_b1_c1 = sd["block1.conv1.weight"].detach().cpu().numpy()
    b_b1_c1 = sd["block1.conv1.bias"].detach().cpu().numpy()
    w_b1_pw1, b_b1_pw1 = fuse_conv_bn(
        sd["block1.pointwise1.weight"], sd["block1.pointwise1.bias"],
        sd["block1.bn1.weight"], sd["block1.bn1.bias"],
        sd["block1.bn1.running_mean"], sd["block1.bn1.running_var"]
    )

    w_b1_c2 = sd["block1.conv2.weight"].detach().cpu().numpy()
    b_b1_c2 = sd["block1.conv2.bias"].detach().cpu().numpy()
    w_b1_pw2, b_b1_pw2 = fuse_conv_bn(
        sd["block1.pointwise2.weight"], sd["block1.pointwise2.bias"],
        sd["block1.bn2.weight"], sd["block1.bn2.bias"],
        sd["block1.bn2.running_mean"], sd["block1.bn2.running_var"]
    )

    # Block 1 SE weights
    w_b1_se_fc1 = sd["block1.se.fc1.weight"].detach().cpu().numpy()
    b_b1_se_fc1 = sd["block1.se.fc1.bias"].detach().cpu().numpy()
    w_b1_se_fc2 = sd["block1.se.fc2.weight"].detach().cpu().numpy()
    b_b1_se_fc2 = sd["block1.se.fc2.bias"].detach().cpu().numpy()

    # Transition 1
    w_trans1, b_trans1 = fuse_conv_bn(
        sd["transition1.0.weight"], sd["transition1.0.bias"],
        sd["transition1.1.weight"], sd["transition1.1.bias"],
        sd["transition1.1.running_mean"], sd["transition1.1.running_var"]
    )

    # Block 2
    w_b2_c1 = sd["block2.conv1.weight"].detach().cpu().numpy()
    b_b2_c1 = sd["block2.conv1.bias"].detach().cpu().numpy()
    w_b2_pw1, b_b2_pw1 = fuse_conv_bn(
        sd["block2.pointwise1.weight"], sd["block2.pointwise1.bias"],
        sd["block2.bn1.weight"], sd["block2.bn1.bias"],
        sd["block2.bn1.running_mean"], sd["block2.bn1.running_var"]
    )

    w_b2_c2 = sd["block2.conv2.weight"].detach().cpu().numpy()
    b_b2_c2 = sd["block2.conv2.bias"].detach().cpu().numpy()
    w_b2_pw2, b_b2_pw2 = fuse_conv_bn(
        sd["block2.pointwise2.weight"], sd["block2.pointwise2.bias"],
        sd["block2.bn2.weight"], sd["block2.bn2.bias"],
        sd["block2.bn2.running_mean"], sd["block2.bn2.running_var"]
    )

    # Block 2 SE
    w_b2_se_fc1 = sd["block2.se.fc1.weight"].detach().cpu().numpy()
    b_b2_se_fc1 = sd["block2.se.fc1.bias"].detach().cpu().numpy()
    w_b2_se_fc2 = sd["block2.se.fc2.weight"].detach().cpu().numpy()
    b_b2_se_fc2 = sd["block2.se.fc2.bias"].detach().cpu().numpy()

    # LayerNorm
    ln_w = sd["norm.weight"].detach().cpu().numpy()
    ln_b = sd["norm.bias"].detach().cpu().numpy()

    # Classification Head
    w_head = sd["head.weight"].detach().cpu().numpy()
    b_head = sd["head.bias"].detach().cpu().numpy()

    # 2. Extract 5 real test beats and subject templates
    tr = np.load(private / "features_train.npz", allow_pickle=True)
    te = np.load(private / "features_test.npz", allow_pickle=True)
    classes = tr["classes"].tolist()

    # Compute templates
    with torch.no_grad():
        emb_train = model.embed(torch.tensor(tr["X_raw"], dtype=torch.float32)).numpy()
    templates = np.stack([emb_train[tr["y"] == i].mean(axis=0) for i in range(len(classes))])
    # Normalize templates
    templates = templates / np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)

    # Pick 5 test beats: Person_01, Person_02, Person_03, Person_88, Person_87
    chosen_ids = ["Person_01", "Person_02", "Person_03", "Person_88", "Person_87"]
    test_beats = []
    test_labels = []
    for cid in chosen_ids:
        c_idx = classes.index(cid)
        matches = np.where(te["y"] == c_idx)[0]
        test_beats.append(te["X_raw"][matches[0]])
        test_labels.append(c_idx)

    # 3. Write C Header File
    h_path = out_dir / "model_data.h"
    with open(h_path, "w") as f:
        f.write("/* Auto-generated Model Weights & Test Vectors for Raspberry Pi Pico RP2040 */\n")
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"#define NUM_CLASSES {n_classes}\n")
        f.write("#define EMBED_DIM 48\n")
        f.write("#define BEAT_LENGTH 300\n")
        f.write(f"#define NUM_TEST_BEATS {len(test_beats)}\n\n")

        def write_array(name, arr):
            flat = arr.flatten()
            f.write(f"static const float {name}[{len(flat)}] = {{\n")
            for i, val in enumerate(flat):
                f.write(f"{val:.7f}f, ")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            f.write("\n};\n\n")

        write_array("W_STEM", w_stem)
        write_array("B_STEM", b_stem)

        write_array("W_B1_C1", w_b1_c1)
        write_array("B_B1_C1", b_b1_c1)
        write_array("W_B1_PW1", w_b1_pw1)
        write_array("B_B1_PW1", b_b1_pw1)
        write_array("W_B1_C2", w_b1_c2)
        write_array("B_B1_C2", b_b1_c2)
        write_array("W_B1_PW2", w_b1_pw2)
        write_array("B_B1_PW2", b_b1_pw2)
        write_array("W_B1_SE_FC1", w_b1_se_fc1)
        write_array("B_B1_SE_FC1", b_b1_se_fc1)
        write_array("W_B1_SE_FC2", w_b1_se_fc2)
        write_array("B_B1_SE_FC2", b_b1_se_fc2)

        write_array("W_TRANS1", w_trans1)
        write_array("B_TRANS1", b_trans1)

        write_array("W_B2_C1", w_b2_c1)
        write_array("B_B2_C1", b_b2_c1)
        write_array("W_B2_PW1", w_b2_pw1)
        write_array("B_B2_PW1", b_b2_pw1)
        write_array("W_B2_C2", w_b2_c2)
        write_array("B_B2_C2", b_b2_c2)
        write_array("W_B2_PW2", w_b2_pw2)
        write_array("B_B2_PW2", b_b2_pw2)
        write_array("W_B2_SE_FC1", w_b2_se_fc1)
        write_array("B_B2_SE_FC1", b_b2_se_fc1)
        write_array("W_B2_SE_FC2", w_b2_se_fc2)
        write_array("B_B2_SE_FC2", b_b2_se_fc2)

        write_array("LN_W", ln_w)
        write_array("LN_B", ln_b)

        write_array("W_HEAD", w_head)
        write_array("B_HEAD", b_head)

        # Enrolled Templates
        write_array("ENROLLED_TEMPLATES", templates)

        # Test Beats & Labels
        write_array("TEST_BEATS", np.array(test_beats))
        f.write(f"static const int TEST_LABELS[{len(test_labels)}] = {{" + ", ".join(map(str, test_labels)) + "};\n")
        f.write("static const char* const TEST_CLASS_NAMES[" + str(len(chosen_ids)) + "] = {\n")
        for cid in chosen_ids:
            f.write(f'    "{cid}",\n')
        f.write("};\n\n")

        f.write("static const char* const ALL_CLASS_NAMES[" + str(len(classes)) + "] = {\n")
        for cid in classes:
            f.write(f'    "{cid}",\n')
        f.write("};\n\n")

        f.write("#endif // MODEL_DATA_H\n")

    print(f"Exported clean Pico C weights to {h_path} ({h_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
