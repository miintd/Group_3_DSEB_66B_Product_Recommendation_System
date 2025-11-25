"""
    Tạo một file product_images_expanded.csv
"""
import os
import pandas as pd
# Các thư mục gốc cần quét
root_dirs = [
    r"F:\Python\Project Code\img\CLOTHING",
    r"F:\Python\Project Code\img\MEN",
    r"F:\Python\Project Code\img\WOMEN",
    r"F:\Python\Project Code\img\DRESSES",
    r"F:\Python\Project Code\img\TOPS",
    r"F:\Python\Project Code\img\TROUSERS"
]
records = []
for root_dir in root_dirs:
    if not os.path.exists(root_dir):
        print(f"Root folder không tồn tại, bỏ qua: {root_dir}")
        continue
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                product_id = os.path.basename(dirpath)
                image_path = os.path.join(dirpath, fname)
                records.append((product_id.strip(), image_path))
#Mục đích: thu thập tất cả file ảnh trong root_dirs và ánh xạ product_id → image_path.

df = pd.DataFrame(records, columns=["product_id", "image_path"])

output_path = "product_images_expanded.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"File '{output_path}' đã được tạo thành công ({len(df)} ảnh).")
print(df.head()) 
