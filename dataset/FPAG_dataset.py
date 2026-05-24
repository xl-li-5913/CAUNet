from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from dataset.noise import Simplex_CLASS
import cv2
from skimage.feature import local_binary_pattern
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops


class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(), \
                                          ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class FPAG(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        self.img_paths = self.load_dataset()

    def load_dataset(self):
        return glob.glob(os.path.join(self.img_path, "good", "*.png"))

    def __len__(self):
        return len(self.img_paths)

    def _normalize(self, mat, eps=1e-8):
        mn = mat.min()
        mx = mat.max()
        return np.divide(mat - mn, mx - mn + eps)

    def _s_membership(self, mat, k=12.0, t=0.5):
        return 1.0 / (1.0 + np.exp(-k * (mat - t)))

    def _random_simplex_fallback(self, img):
        size = img.shape[0]
        h_noise = np.random.randint(10, int(np.floor(np.divide(size, 8))))
        w_noise = np.random.randint(10, int(np.floor(np.divide(size, 8))))
        sh = np.random.randint(1, size - h_noise)
        sw = np.random.randint(1, size - w_noise)

        simplex = self.simplexNoise.rand_3d_octaves(
            (3, h_noise, w_noise), octaves=6, persistence=0.6
        )

        canvas = np.zeros((size, size, 3), dtype=np.float32)
        canvas[sh:sh + h_noise, sw:sw + w_noise, :] = 0.2 * simplex.transpose(1, 2, 0)
        return np.clip(img + canvas, 0.0, 1.0)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0

        img_normal = self.transform(img)

        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = gray / 255.0
        h_img, w_img = gray.shape

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(np.square(gx) + np.square(gy))
        edge = self._normalize(edge)

        mean = cv2.boxFilter(gray, cv2.CV_32F, (9, 9))
        mean_sq = cv2.boxFilter(np.square(gray), cv2.CV_32F, (9, 9))
        var = np.maximum(0.0, mean_sq - np.square(mean))
        var = self._normalize(var)

        lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")
        lbp = self._normalize(lbp.astype(np.float32))

        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        sal = np.abs(gray - blur)
        sal = self._normalize(sal)

        mu_edge = self._s_membership(edge, 12.0, 0.3)
        mu_var = self._s_membership(var, 12.0, 0.25)
        mu_lbp = self._s_membership(lbp, 10.0, 0.4)
        mu_sal = self._s_membership(sal, 10.0, 0.3)

        mu = 0.30 * mu_edge + 0.25 * mu_var + 0.20 * mu_lbp + 0.25 * mu_sal
        mu = gaussian_filter(mu, sigma=3.0)
        mu = np.clip(mu, 0.0, 1.0)

        binary = mu > 0.45
        labeled = label(binary)
        regions = regionprops(labeled, intensity_image=mu)

        canvas = np.zeros((h_img, w_img, 3), dtype=np.float32)
        injected = 0

        area_min = int(np.floor(np.divide(h_img * w_img, 2000)))
        mu_min = 0.4
        regions = sorted(regions, key=lambda r: r.area, reverse=True)

        for reg in regions[:3]:
            if reg.area < area_min:
                continue
            if reg.mean_intensity < mu_min:
                continue

            minr, minc, maxr, maxc = reg.bbox
            mask = labeled[minr:maxr, minc:maxc] == reg.label
            ys, xs = np.where(mask)

            if len(ys) == 0:
                continue

            sel = np.random.randint(0, len(ys))
            cy = minr + ys[sel]
            cx = minc + xs[sel]

            mu_reg = float(np.clip(reg.mean_intensity, 0.0, 1.0))
            min_size, max_size = 8, 64
            patch_size = int(min_size + (max_size - min_size) * mu_reg)

            half = int(np.floor(np.divide(patch_size, 2)))
            y0 = max(0, cy - half)
            x0 = max(0, cx - half)
            y1 = min(h_img, y0 + patch_size)
            x1 = min(w_img, x0 + patch_size)

            ph = y1 - y0
            pw = x1 - x0
            if ph <= 0 or pw <= 0:
                continue

            if ph <= 16:
                freq = 64
            elif ph <= 32:
                freq = 32
            else:
                freq = 16

            simplex = self.simplexNoise.rand_3d_octaves(
                (3, ph, pw), octaves=6, persistence=0.6, frequency=freq
            )

            amp = 0.2 * mu_reg
            canvas[y0:y1, x0:x1, :] += amp * 2.0 * simplex.transpose(1, 2, 0)
            injected += 1

        if injected == 0:
            img_noise = self._random_simplex_fallback(img)
        else:
            img_noise = np.clip(img + canvas, 0.0, 1.0)

        img_noise = self.transform(img_noise)
        return img_normal, img_noise, os.path.basename(img_path)


class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img / 255., (256, 256))
        ## Normal
        img = self.transform(img)
        ## simplex_noise

        if gt == 0:
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path.split('/')[-1])


if __name__ == "__main__":
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # =========================
    # 1. 构造 transform
    # =========================
    data_transform, _ = get_data_transforms(256, 256)

    root_dir = os.path.join(
        "..", "..", "..", "dataset", "MVTec", "hazelnut", "train"
    )

    dataset = FPAG(
        root=root_dir,
        transform=data_transform
    )

    # =========================
    # 2. 取一个样本
    # =========================
    img_normal, img_noise, name = dataset[0]

    # tensor -> numpy
    img_normal = img_normal.permute(1, 2, 0).cpu().numpy()
    img_noise = img_noise.permute(1, 2, 0).cpu().numpy()

    # =========================
    # 3. 反归一化（用于显示）
    # =========================
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_normal_vis = np.clip(img_normal * std + mean, 0.0, 1.0)
    img_noise_vis = np.clip(img_noise * std + mean, 0.0, 1.0)

    # =========================
    # 4. 构造注入 mask
    # =========================
    diff = np.abs(img_noise_vis - img_normal_vis)
    diff_gray = np.mean(diff, axis=2)

    # 自适应阈值
    thr = np.percentile(diff_gray, 98)
    mask = diff_gray > thr

    # =========================
    # 5. 可视化
    # =========================
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_normal_vis)
    plt.title("Normal image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(img_noise_vis)
    plt.title("Injected image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Injection mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

