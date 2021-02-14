from PIL import Image, ImageDraw, ImageStat
import cv2
import numpy as np


class Ops:
    def __init__(self, imageType='rgb'):  # rgb or grey
        self.__imageType = imageType
        pass

    def __isSingle(self, images):
        if isinstance(images, str):
            return True
        if isinstance(images, (tuple, list)):
            return False
        if isinstance(images, np.ndarray):
            if self.__imageType == "grey":
                return len(images.shape) == 2
            else:
                return len(images.shape) == 3
        raise TypeError("invalid format")

    def __validateOperation(self, opsType):
        if self.__imageType == 'grey' and (opsType == 'contrast' or opsType == "saturation"
                                           or opsType == "brightness"):
            raise TypeError("invalid operation type : " + opsType)

    def __pad3dim(self,image):
        image = image.reshape(list(image.shape)+[1])
        return np.pad(image, ((0, 0), (0, 0), (0, 2)), 'constant',constant_values=0)

    def __dePad3dim(self,image):
        return image[:, :, 0]

    def perspectiveTransform(self, images):
        def getMaskCoord(imshape):
            vertices = np.array([[(0.01 * imshape[1], 0.09 * imshape[0]),
                                  (0.13 * imshape[1], 0.32 * imshape[0]),
                                  (0.15 * imshape[1], 0.32 * imshape[0]),
                                  (0.89 * imshape[1], 0.5 * imshape[0])]], dtype=np.int32)
            return vertices

        def getPerspectiveMatrices(X_img):
            offset = 15
            img_size = (X_img.shape[1], X_img.shape[0])
            src = np.float32(getMaskCoord(X_img.shape))
            dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                              [img_size[0] - offset, img_size[1]]])
            perspective_matrix = cv2.getPerspectiveTransform(src, dst)
            return perspective_matrix

        def transform(image):
            perspective_matrix = getPerspectiveMatrices(image)
            image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]),
                                        flags=cv2.INTER_LINEAR)
            return image

        if self.__isSingle(images):
            return transform(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(transform(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def addNoise(self, images, type='gussian'):
        def change(image):
            type = type.lower()
            if (type == 'blur'):
                return cv2.blur(image, (5, 5))
            elif (type == 'gussian'):
                return cv2.GaussianBlur(image, (5, 5), 0)
            elif (type == 'median'):
                return cv2.medianBlur(image, 5)
            elif (type == 'bilateral'):
                return cv2.bilateralFilter(image, 9, 75, 75)

        if self.__isSingle(images):
            return change(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def addSaltPepperNoise(self, images, salt_vs_pepper=0.2, amount=0.004):
        def change(X_imgs):
            if self.__imageType == 'grey':
                X_imgs = self.__pad3dim(X_imgs)
            X_imgs_copy = X_imgs.copy()
            row, col, _ = X_imgs_copy.shape
            num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
            num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
            for X_img in X_imgs_copy:
                # Add Salt noise
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
                X_img[coords[0], coords[1]] = 1
                # Add Pepper noise
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
                X_img[coords[0], coords[1]] = 0
            if self.__imageType == 'grey':
                X_imgs_copy = self.__dePad3dim(X_imgs_copy)
            return X_imgs_copy

        if self.__isSingle(images):
            return change(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def augmentBrightness(self, images, brightness):
        self.__validateOperation("brightness")
        def change(image, bright):
            image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image1[:, :, 2] = image1[:, :, 2] * bright
            image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
            return image1

        if self.__isSingle(images):
            return change(images, brightness)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i], brightness))
            return self.__returnAsNumpyArray(imgs)

    def rotation(self, images, ang_range):
        def change(image):
            if self.__imageType == 'grey':
                image = self.__pad3dim(image)
            ang_rot = np.random.uniform(ang_range) - ang_range / 2
            rows, cols, ch= image.shape
            Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)
            image = np.array(cv2.warpAffine(image, Rot_M, (cols, rows)))
            if self.__imageType == 'grey':
                image = self.__dePad3dim(image)
            return image

        if self.__isSingle(images):
            return change(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def shear(self, images, shear_range):
        def change(image):
            if self.__imageType == 'grey':
                image = self.__pad3dim(image)
            rows, cols, ch = image.shape
            pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
            pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
            pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
            pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
            shear_M = cv2.getAffineTransform(pts1, pts2)
            image = np.array(cv2.warpAffine(image, shear_M, (cols, rows)))
            if self.__imageType == 'grey':
                image = self.__dePad3dim(image)
            return image

        if self.__isSingle(images):
            return change(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def translation(self, images, trans_range):
        def change(image):
            if self.__imageType == 'grey':
                image = self.__pad3dim(image)
            rows, cols, ch = image.shape
            tr_x = trans_range * np.random.uniform() - trans_range / 2
            tr_y = trans_range * np.random.uniform() - trans_range / 2
            Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
            image = cv2.warpAffine(image, Trans_M, (cols, rows))
            if self.__imageType == 'grey':
                image = self.__dePad3dim(image)
            return image

        if self.__isSingle(images):
            return change(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(change(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def horizontalFlip(self, images):
        if self.__isSingle(images):
            return cv2.flip(images, 0)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(cv2.flip(images[i], 0)))
            return self.__returnAsNumpyArray(imgs)

    def verticalFlip(self, images):
        if self.__isSingle(images):
            return cv2.flip(images, 1)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(cv2.flip(images[i], 1)))
            return self.__returnAsNumpyArray(imgs)

    def transposeFlip(self, images):
        if self.__isSingle(images):
            return cv2.flip(images, -1)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(cv2.flip(images[i], -1)))
            return self.__returnAsNumpyArray(imgs)

    def changeContrast(self, images, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))
        self.__validateOperation("contrast")

        def contrast(c):
            return 128 + factor * (c - 128)

        def change(img):
            return np.array(Image.fromarray(img).point(contrast))

        if self.__isSingle(images):
            return contrast(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def saturation(self, images, ratio=0.5):
        self.__validateOperation("saturation")
        import PIL.ImageEnhance as enhance
        if self.__isSingle(images):
            images = Image.fromarray(images)
            converter = enhance.Color(images)
            return converter.enhance(ratio)
        else:
            imgs = []
            for i in range(len(images)):
                img = Image.fromarray(images[i])
                converter = enhance.Color(img)
                imgs.append(np.array(converter.enhance(ratio)))
            return self.__returnAsNumpyArray(imgs)

    def convertGrayscale(self, images):
        def convert(image):
            image = Image.fromarray(image).convert('L')
            image = np.asarray(image, dtype="int32")
            return image

        if self.__isSingle(images):
            return convert(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(convert(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def halftoneImage(self, images, sample=10, scale=1, percentage=0, angles=[0, 15, 30, 45], style='color',
                      antialias=False):
        class Halftone(object):
            def make(self, img, sample=10, scale=1, percentage=0, angles=[0, 15, 30, 45], style='color',
                     antialias=False):
                img = Image.fromarray(img)
                if style == 'grayscale':
                    angles = angles[:1]
                    gray_im = img.convert('L')
                    dots = self.__halftone(img, gray_im, sample, scale, angles, antialias)
                    new = dots[0]
                else:
                    cmyk = self.__gcr(img, percentage)
                    dots = self.__halftone(img, cmyk, sample, scale, angles, antialias)
                    new = Image.merge('CMYK', dots)
                return new

            def __gcr(self, im, percentage):
                cmyk_im = im.convert('CMYK')
                if not percentage:
                    return cmyk_im
                cmyk_im = cmyk_im.split()
                cmyk = []
                for i in range(4):
                    cmyk.append(cmyk_im[i].load())
                for x in range(im.size[0]):
                    for y in range(im.size[1]):
                        gray = min(cmyk[0][x, y], cmyk[1][x, y], cmyk[2][x, y]) * percentage / 100
                        for i in range(3):
                            cmyk[i][x, y] = cmyk[i][x, y] - gray
                        cmyk[3][x, y] = gray
                return Image.merge('CMYK', cmyk_im)

            def __halftone(self, im, cmyk, sample, scale, angles, antialias):
                antialias_scale = 4

                if antialias is True:
                    scale = scale * antialias_scale

                cmyk = cmyk.split()
                dots = []

                for channel, angle in zip(cmyk, angles):
                    channel = channel.rotate(angle, expand=1)
                    size = channel.size[0] * scale, channel.size[1] * scale
                    half_tone = Image.new('L', size)
                    draw = ImageDraw.Draw(half_tone)
                    for x in range(0, channel.size[0], sample):
                        for y in range(0, channel.size[1], sample):
                            box = channel.crop((x, y, x + sample, y + sample))
                            mean = ImageStat.Stat(box).mean[0]
                            diameter = (mean / 255) ** 0.5
                            box_size = sample * scale
                            draw_diameter = diameter * box_size
                            box_x, box_y = (x * scale), (y * scale)
                            x1 = box_x + ((box_size - draw_diameter) / 2)
                            y1 = box_y + ((box_size - draw_diameter) / 2)
                            x2 = x1 + draw_diameter
                            y2 = y1 + draw_diameter

                            draw.ellipse([(x1, y1), (x2, y2)], fill=255)

                    half_tone = half_tone.rotate(-angle, expand=1)
                    width_half, height_half = half_tone.size
                    xx1 = (width_half - im.size[0] * scale) / 2
                    yy1 = (height_half - im.size[1] * scale) / 2
                    xx2 = xx1 + im.size[0] * scale
                    yy2 = yy1 + im.size[1] * scale

                    half_tone = half_tone.crop((xx1, yy1, xx2, yy2))

                    if antialias is True:
                        w = (xx2 - xx1) / antialias_scale
                        h = (yy2 - yy1) / antialias_scale
                        half_tone = half_tone.resize((w, h), resample=Image.LANCZOS)

                    dots.append(half_tone)
                return dots

        h = Halftone()

        if self.__isSingle(images):
            return np.array(h.make(images, sample, scale, percentage, angles, style, antialias))
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(h.make(images[i], sample, scale, percentage, angles, style, antialias)))
            return self.__returnAsNumpyArray(imgs)

    def __rgb_to_hsv(self, rgb):
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    def __hsv_to_rgb(self, hsv):
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __returnAsNumpyArray(self, data):
        try:
            return np.array(data).reshape([-1] + list(data[0].shape))
        except:
            return data

    def shiftHue(self, images, hout):
        if self.__isSingle(images):
            hsv = self.__rgb_to_hsv(images)
            hsv[..., 0] = hout
            return np.array(self.__hsv_to_rgb(hsv))
        else:
            imgs = []
            for i in range(len(images)):
                hsv = self.__rgb_to_hsv(images[i])
                hsv[..., 0] = hout
                hsv = np.array(self.__hsv_to_rgb(hsv))
                imgs.append(np.array(hsv))
            return self.__returnAsNumpyArray(imgs)

    def randomErasing(self, images, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        import random, math
        def erase(img):
            img = img.copy()
            for attempt in range(50):
                area = img.shape[0] * img.shape[2]
                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1 / r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w <= img.shape[0] and h <= img.shape[1]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)
                    if img.shape[0] == 3:
                        img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                        img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                        img[x1:x1 + h, y1:y1 + w, 2] = mean[2]
                    else:
                        img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                    return np.array(img)

        if self.__isSingle(images):
            if self.__imageType == 'grey':
                images = self.__pad3dim(images)
            images =  erase(images)
            if self.__imageType == 'grey':
                images = self.__dePad3dim(images)
            return images
        else:
            imgs = []
            for i in range(len(images)):
                img=images[i]
                if self.__imageType == 'grey':
                    img = self.__pad3dim(img)
                img = erase(img)
                if self.__imageType == 'grey':
                    img = self.__dePad3dim(img)
                imgs.append(img)
            return self.__returnAsNumpyArray(imgs)

    def cropFromCentre(self, images, width, height):
        def crop(img):
            img = Image.fromarray(img)
            old_width, old_height = img.size
            left = (old_width - width) / 2
            top = (old_height - height) / 2
            right = (old_width + width) / 2
            bottom = (old_height + height) / 2
            img = img.crop((left, top, right, bottom))
            return np.array(img)

        if self.__isSingle(images):
            return crop(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(crop(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def resizeImage(self, images, width=80, height=80):
        def resize(image):
            image = Image.fromarray(image)
            return np.array(image.resize((width, height), Image.ANTIALIAS))

        if self.__isSingle(images):
            return resize(images)
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(resize(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def resizeImageWithAspectRatio(self, images, size=128, padding=True, pad_option=0):
        def resize(img, size, padding, pad_option):
            height, width = img.shape[0], img.shape[1]
            if height > width:
                ratio = float(size) / height
                new_height = size
                new_width = int(ratio * width)
                img = self.resize_image(img, new_width, new_height)
                pad_size = size - new_width
                if padding:
                    if pad_size % 2 == 0:
                        img = np.pad(img, ((0, 0), (pad_size // 2, pad_size // 2), (0, 0)), 'constant',
                                     constant_values=pad_option)
                    else:
                        img = np.pad(img, ((0, 0), (pad_size // 2, pad_size // 2 + 1), (0, 0)), 'constant',
                                     constant_values=pad_option)
            else:
                ratio = float(size) / width
                new_width = size
                new_height = int(ratio * height)
                img = self.resizeImage(img, new_width, new_height)
                pad_size = size - new_height
                if padding:
                    if(self.__imageType == 'grey'):
                        if pad_size % 2 == 0:
                            paddingSize = ((pad_size // 2, pad_size // 2), (0, 0))
                        else:
                            paddingSize = ((pad_size // 2, pad_size // 2 + 1), (0, 0))
                    else:
                        if pad_size % 2 == 0:
                            paddingSize = ((pad_size // 2, pad_size // 2), (0, 0), (0, 0))
                        else:
                            paddingSize = ((pad_size // 2, pad_size // 2 + 1), (0, 0), (0, 0))
                    img = np.pad(img, paddingSize, 'constant', constant_values=pad_option)
            return img

        if self.__isSingle(images):
            return np.array(resize(images, size, padding, pad_option))
        else:
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(resize(images[i], size, padding, pad_option)))
            return self.__returnAsNumpyArray(imgs)

    def readImage(self, filename):
        if self.__isSingle(filename):
            return np.array(Image.open(filename))
        else:
            imgs = []
            for i in range(len(filename)):
                imgs.append(np.array(Image.open(filename[i])))
            return self.__returnAsNumpyArray(imgs)

    def pca_color_augmenataion(self, data):
        import numpy as np
        def data_aug(img, evecs_mat):
            mu = 0
            sigma = 0.1
            feature_vec = np.matrix(evecs_mat)
            se = np.zeros((3, 1))
            se[0][0] = np.random.normal(mu, sigma) * evals[0]
            se[1][0] = np.random.normal(mu, sigma) * evals[1]
            se[2][0] = np.random.normal(mu, sigma) * evals[2]
            se = np.matrix(se)
            val = feature_vec * se
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        img[i, j, k] = float(img[i, j, k]) + float(val[k])
            return img

        res = data.reshape([-1, 3])
        m = res.mean(axis=0)
        res = res - m
        R = np.cov(res, rowvar=False)
        from numpy import linalg as LA
        evals, evecs = LA.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        evecs = evecs[:, :3]
        # evecs_mat = np.column_stack((evecs))
        m = np.dot(evecs.T, res.T).T
        img = []
        for i in range(len(data)):
            img.append(data_aug(data[i], m))
        return np.array(img)
