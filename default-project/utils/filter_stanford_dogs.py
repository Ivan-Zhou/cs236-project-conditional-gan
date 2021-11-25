import os
import pandas as pd
import shutil


class FilterDataset:
    def __init__(self, data_dir, dst_dir, top_k=10):
        self.data_dir = data_dir
        self.dst_dir = dst_dir
        self._make_dir(self.dst_dir)
        self.top_k = top_k
    
    @staticmethod
    def _get_stats(data_dir):
        stats_data = []
        for class_dir in os.listdir(data_dir):
            img_list = os.listdir(os.path.join(data_dir, class_dir))
            stats_data.append({
                "class": class_dir,
                "example_count": len(img_list)
            })
        df_stats = pd.DataFrame(stats_data)
        df_stats = df_stats.sort_values(by="example_count", ascending=False, ignore_index=True)
        return df_stats

    @staticmethod
    def _make_dir(dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)
    
    def _create_sub_dataset(self, folders):
        for folder in folders:
            src = os.path.join(self.data_dir, folder)
            dst = os.path.join(self.dst_dir, folder)
            self._make_dir(dst)
            print(f"Copy files from {src} to {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)


    def __call__(self):
        df_stats = self._get_stats(self.data_dir)
        top_k_classes = df_stats["class"][:self.top_k].tolist()
        self._create_sub_dataset(top_k_classes)

def main():
    top_k=10
    data_dir="default-project/data/Images"
    dst_dir = "default-project/data/stanford_dogs_top_10"
    dataset_filder = FilterDataset(data_dir, dst_dir, top_k)
    dataset_filder()


if __name__ == "__main__":
    main()
