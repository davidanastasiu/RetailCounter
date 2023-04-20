import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import torch.nn.functional as F



class Extractor(object):
    def __init__(self, model, use_cuda=True):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)
        self.size = (640, 640)

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        # import pdb;pdb.set_trace()
        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        # import pdb;pdb.set_trace()
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
            # print(features.shape)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor('model')
    feature = extr(img)
    print(feature.shape)
