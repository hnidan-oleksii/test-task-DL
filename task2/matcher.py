import torch
import kornia as K
from kornia.feature import DISK, LightGlueMatcher, laf_from_center_scale_ori
from kornia_moons.viz import draw_LAF_matches
import cv2
from rasterio import open as rast_open


class LightGlue_DISK_Matcher():
    def __init__(self, num_features=4096, image_size=None, device=None):
        self.num_features = num_features
        self.image_size = image_size
        self.device = device if device else K.utils.get_cuda_or_mps_device_if_available()

    @torch.inference_mode
    def match(self, img_path1, img_path2):
        """Match images of shape (C x H x W) using LightGlue and DISK."""
        image1 = self.__prepare_image(img_path1)
        image2 = self.__prepare_image(img_path2)

        matcher = LightGlueMatcher("disk").eval().to(self.device)
        disk = DISK.from_pretrained("depth").to(self.device)

        # tensors with H and W as values
        hw1 = torch.tensor(image1.shape[-2:], device=self.device)
        hw2 = torch.tensor(image2.shape[-2:], device=self.device)

        # stacking images into batch of size 2 x C x H x W
        inp = torch.cat([image1.unsqueeze(0), image2.unsqueeze(0)], dim=0)
        feats1, feats2 = disk(inp, self.num_features, pad_if_not_divisible=True)
        kps1, descs1 = feats1.keypoints, feats1.descriptors
        kps2, descs2 = feats2.keypoints, feats2.descriptors

        lafs1 = laf_from_center_scale_ori(
                kps1[None],
                torch.ones((1, len(kps1), 1, 1), device=self.device)
        )
        lafs2 = laf_from_center_scale_ori(
                kps2[None],
                torch.ones((1, len(kps2), 1, 1), device=self.device)
        )

        _, idxs = matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)

        return {
            'image1': image1,
            'image2': image2,
            'kps1': kps1,
            'kps2': kps2,
            'idxs': idxs
        }

    def draw_matches(self, image1, image2, kps1, kps2, idxs, output=None):
        '''Draw matches on parallel images and write to output path, if provided.'''
        mkpts1, mkpts2 = self.__get_matching_keypoints(kps1, kps2, idxs)

        # if there are 0 inliers, list will be empty, so initializing it with None
        try:
            _, inliers = cv2.findFundamentalMat(
                mkpts1.cpu().numpy(), mkpts2.cpu().numpy(), cv2.USAC_MAGSAC, 1.0, 0.9999, 100000
            )
            inliers = inliers > 0
        except:
            inliers = None

        fig, ax = draw_LAF_matches(
            laf_from_center_scale_ori(kps1[None].cpu()),
            laf_from_center_scale_ori(kps2[None].cpu()),
            idxs.cpu(),
            K.tensor_to_image(image1.cpu()),
            K.tensor_to_image(image2.cpu()),
            inliers,
            return_fig_ax=True,
            draw_dict={
                "inlier_color": (0.2, 1, 0.2),
                "tentative_color": (1, 1, 0.2, 0.3),
                "feature_color": None,
                "vertical": False
            },
        )
        if output:
            fig.savefig(output)

    def __prepare_image(self, path):
        '''Load image from path and resize, if self.image_size is provided'''
        if path.endswith('.jp2'):
            # read JP2 image in CxHxW shape
            with rast_open(path, "r", driver='JP2OpenJPEG') as rst_src:
                raster_image = rst_src.read()
                image = torch.from_numpy(raster_image).float() / 255.0
        else:
            image = K.io.load_image(path, K.io.ImageLoadType.RGB32, self.device)

        if self.image_size:
            image = K.geometry.resize(image, size=self.image_size, interpolation='area')
        return image.to(self.device)

    def __get_matching_keypoints(self, kp1, kp2, idxs):
        mkpts1 = kp1[idxs[:, 0]]
        mkpts2 = kp2[idxs[:, 1]]
        return mkpts1, mkpts2
