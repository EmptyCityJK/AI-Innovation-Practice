import os

def count_images_in_folder(folder_path):
    # 支持的图片扩展名
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    count = 0

    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                count += 1

    return count

if __name__ == "__main__":
    folder_path = "./data"
    image_count = count_images_in_folder(folder_path)
    print(f"图片总数: {image_count}")
