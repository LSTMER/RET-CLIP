import lmdb
import pickle
import base64
from io import BytesIO
from tqdm import tqdm
import os
import pandas as pd
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from vit_concept_map import SmartFundusCrop


class DRDataset(Dataset):
    def __init__(self, csv_files, img_dirs, transform=None, img_ext='.jpg'):
        self.transform = transform
        self.img_ext = img_ext
        self.preprocess = SmartFundusCrop(target_size=1024)

        if isinstance(csv_files, str): csv_files = [csv_files]
        if isinstance(img_dirs, str): img_dirs = [img_dirs]
        assert len(csv_files) == len(img_dirs), "CSV 文件数量必须与图片文件夹数量一致！"
        df_list = []
        for csv_path, img_path in zip(csv_files, img_dirs):
            # 注意：确保你的 CSV 里有 ID, RATE, EX... 这些列
            temp_df = pd.read_csv(csv_path, dtype={'ID': str})
            temp_df['img_root'] = img_path
            df_list.append(temp_df)
        self.data = pd.concat(df_list, ignore_index=True)

        self.grade_map = {
            0: "正常的", 1: "轻度非增殖性糖尿病视网膜病变(Mild NPDR)",
            2: "中度非增殖性糖尿病视网膜病变(Moderate NPDR)",
            3: "重度非增殖性糖尿病视网膜病变(Severe NPDR)", 4: "增殖性糖尿病视网膜病变(PDR)"
        }
        self.lesion_map = {
            'EX': '硬性渗出', 'HE': '视网膜出血', 'MA': '微血管瘤',
            'SE': '软性渗出(棉絮斑)', 'MHE': '玻璃体积血', 'BRD': '玻璃体混浊'
        }

    def generate_text(self, row):
        rate = int(row['RATE'])
        present_lesions = []
        # 注意：这里假设 CSV 列名和 lesion_map key 一致
        for col, cn_name in self.lesion_map.items():
            if col in row and int(row[col]) == 1:
                present_lesions.append(cn_name)

        if rate == 0:
            templates = [
                "一张正常的眼底照片，视网膜结构清晰。",
                "眼底图像显示无明显病变，视神经和黄斑结构正常。",
                "健康的眼底图像，无糖尿病视网膜病变迹象。"
            ]
            return random.choice(templates)

        grade_desc = self.grade_map.get(rate, "糖尿病视网膜病变")
        if present_lesions:
            random.shuffle(present_lesions)
            lesions_str = "、".join(present_lesions)
            templates = [
                f"一张可见{lesions_str}的眼底照片，。",
                f"一张主要病灶包含{lesions_str}的眼底照片。",
                f"一张眼底照片，可见{lesions_str}病灶"
            ]
            return random.choice(templates)
        else:
            return f"一张{grade_desc}的眼底照片。"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.data.iloc[idx]
        img_name = os.path.join(row['img_root'], row['ID'] + self.img_ext)
        try:
            image_raw = Image.open(img_name).convert('RGB')
            image = self.preprocess(image_raw)
        except FileNotFoundError:
            print(f"Warning: Image {img_name} not found.")
            image = Image.new('RGB', (224, 224))

        # 注意：我们在 __getitem__ 里不应用 transform，因为我们要保存原图（或Resize后的图）到 LMDB
        # transform 留给训练时的 DataLoader 做

        text = self.generate_text(row)
        return image, text # 返回 PIL Image 和 文本字符串

# ==========================================
# 2. LMDB 转换函数
# ==========================================

def make_lmdb_from_dataset(dataset, output_path, max_size_gb=100):
    """
    Args:
        dataset: 你的 DRDataset 实例
        output_path: 输出文件夹路径
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    path_imgs = os.path.join(output_path, "imgs")
    path_pairs = os.path.join(output_path, "pairs")

    if not os.path.exists(path_imgs): os.makedirs(path_imgs)
    if not os.path.exists(path_pairs): os.makedirs(path_pairs)

    # 估算 LMDB 大小
    map_size = int(max_size_gb * 1024 * 1024 * 1024)

    env_imgs = lmdb.open(path_imgs, map_size=map_size)
    env_pairs = lmdb.open(path_pairs, map_size=map_size)

    txn_imgs = env_imgs.begin(write=True)
    txn_pairs = env_pairs.begin(write=True)

    print(f"开始转换，共 {len(dataset)} 个样本...")

    sample_idx = 0

    # 遍历 Dataset
    for idx in tqdm(range(len(dataset))):
        try:
            # 获取数据 (PIL Image, Text String)
            image, text = dataset[idx]

            # --- 1. 图片处理 ---
            # 为了节省空间和 I/O，建议统一 Resize 到 256 或 512
            image = image.resize((256, 256), Image.BICUBIC)

            # 转为 Base64 Bytes
            buff = BytesIO()
            image.save(buff, format="JPEG", quality=90)
            img_b64 = base64.urlsafe_b64encode(buff.getvalue())

            # ★ 关键策略：构造左右眼对 ★
            # 因为 RET-CLIP 要求双眼输入，我们复制一份
            img_pair_b64 = [img_b64, img_b64]

            # 存入 imgs (Key: patient_id)
            # 这里我们简单用 idx 作为 patient_id
            patient_id = idx
            txn_imgs.put(
                key=f"{patient_id}".encode('utf-8'),
                value=pickle.dumps(img_pair_b64)
            )

            # --- 2. 文本处理 ---
            # 存入 pairs (Key: sample_idx)
            # 格式: (patient_id, text_id, raw_text)
            pair_data = (patient_id, idx, text)

            txn_pairs.put(
                key=f"{sample_idx}".encode('utf-8'),
                value=pickle.dumps(pair_data)
            )
            sample_idx += 1

            # 定期提交
            if sample_idx % 1000 == 0:
                txn_imgs.commit()
                txn_pairs.commit()
                txn_imgs = env_imgs.begin(write=True)
                txn_pairs = env_pairs.begin(write=True)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue

    # 写入元数据
    txn_imgs.put(b'num_images', str(sample_idx).encode('utf-8')) # 这里 image count 等于 sample count
    txn_pairs.put(b'num_samples', str(sample_idx).encode('utf-8'))

    txn_imgs.commit()
    txn_pairs.commit()
    env_imgs.close()
    env_pairs.close()
    print(f"完成！LMDB 保存在: {output_path}")

# ==========================================
# 3. 运行配置 (请修改这里)
# ==========================================

if __name__ == "__main__":
    # --- 配置训练集 ---
    train_csvs = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/train.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/train.csv"]
    train_imgs = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/process_image/",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/train_process/"]

    print("正在创建 Training LMDB...")
    train_ds = DRDataset(train_csvs, train_imgs, img_ext='.jpg') # 确保后缀正确
    make_lmdb_from_dataset(train_ds, "./lmdb_output/train_lmdb")

    # --- 配置验证集 (如果有) ---
    val_csvs = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/concept_annotation/split/valid.csv",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/mfiddr/valid.csv"]
    val_imgs = ["/storage/luozhongheng/luo/concept_base/concept_dataset/new_dataset/process_image",
        "/storage/luozhongheng/luo/concept_base/concept_dataset/train_process/"]
    print("正在创建 Validation LMDB...")
    val_ds = DRDataset(val_csvs, val_imgs, img_ext='.jpg')
    make_lmdb_from_dataset(val_ds, "./lmdb_output/val_lmdb")
