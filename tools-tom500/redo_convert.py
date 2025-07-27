import os, glob, h5py, numpy as np, nibabel as nib
from tqdm import tqdm

src_root = "/home/xuyang/code/EMCAD-TOM500/datasets/TOM500"
dst_root = "/home/xuyang/code/EMCAD-TOM500/datasets/TOM500_tun"

train_npz = os.path.join(dst_root, "train_npz")
test_h5   = os.path.join(dst_root, "test_vol_h5")
list_dir  = os.path.join(dst_root, "..", "lists", "lists_Synapse")
os.makedirs(train_npz, exist_ok=True)
os.makedirs(test_h5,   exist_ok=True)
os.makedirs(list_dir,  exist_ok=True)

# ---------- 工具 ----------
def clip_norm(x):
    x = np.clip(x, -125, 275)
    return ((x + 125) / 400).astype(np.float32)

def convert_split(split):
    img_dir = os.path.join(src_root, split, "image")
    lbl_dir = os.path.join(src_root, split, "label")
    imgs = sorted(os.listdir(img_dir))
    lbls = sorted(os.listdir(lbl_dir))
    assert len(imgs) == len(lbls)

    for idx, (img_nii, lbl_nii) in enumerate(tqdm(zip(imgs, lbls), desc=f"Converting {split}")):
        img = nib.load(os.path.join(img_dir, img_nii)).get_fdata()
        lbl = nib.load(os.path.join(lbl_dir, lbl_nii)).get_fdata()

        img = clip_norm(img)

        # 1. 训练 .npz
        if split == "train":
            for s in range(min(img.shape[-1], 20)):
                np.savez_compressed(
                    os.path.join(train_npz, f"case{idx:04d}_slice{s:03d}.npz"),
                    image=img[:, :, s],
                    label=lbl[:, :, s].astype(np.uint8)
                )
        # 2. 测试 .h5
        else:
            with h5py.File(os.path.join(test_h5, f"case{idx:04d}.npy.h5"), "w") as f:
                f.create_dataset("image", data=img)
                f.create_dataset("label", data=lbl.astype(np.uint8))

# ---------- 主流程 ----------
print("🔄 重新生成 train_npz & test_vol_h5 …")
convert_split("train")
convert_split("val")

# 3. 重建列表
with open(os.path.join(list_dir, "train.txt"), "w") as f:
    for npz in sorted(glob.glob(os.path.join(train_npz, "*.npz"))):
        f.write(os.path.basename(npz).replace(".npz", "") + "\n")

with open(os.path.join(list_dir, "test.txt"), "w") as f:
    for h5 in sorted(glob.glob(os.path.join(test_h5, "*.h5"))):
        f.write(os.path.basename(h5).replace(".npy.h5", "") + "\n")

print("✅ 数据+列表已全部重建完成！")