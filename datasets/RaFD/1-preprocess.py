import os
import numpy as np
import cv2


def tuple_shape(shape):
    r_data = []
    for p in shape:
        r_data.append([p.x, p.y])
    return r_data

def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
    for p in shape:
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, color, thickness)
    return img

# for *.txt output
def str_landmark(land_in, name, img_size):
    out_landmark = ''
    out_landmark += '{}'.format(name)
    out_landmark += ' {}-{}'.format(img_size[0], img_size[1])
    for p in land_in:
        out_landmark += ' {}-{}'.format(p[0], p[1])
    out_landmark += '\n'
    return out_landmark


def crop_and_generate_landmark_img():

    def xy(shape):
        x_min, x_max, y_min, y_max = shape[0][0], shape[0][0], shape[0][1], shape[0][1]
        for p in shape:
            x_min = min([x_min, p[0]])
            x_max = max([x_max, p[0]])
            y_min = min([y_min, p[1]])
            y_max = max([y_max, p[1]])
        return (x_min + x_max) / 2, (y_min + y_max) / 2, max([(x_max - x_min) / 2, (y_max - y_min) / 2])
        # return x_min, y_min, max([(x_max - x_min) / 2, (y_max - y_min) / 2])

    def generate():
        labs_ = open(land_txt, 'r').readlines()
        f = open(land_norm_txt, 'w')
        for _, l in enumerate(labs_):
            print('\r{} {}/{}'.format(land_txt, _ + 1, len(labs_)), end='')
            l = l.strip().split()
            name = l[0]
            w_ori, h_ori = [int(_) for _ in l[1].split('-')]
            shape = []
            for l_ in l[2:]:
                w, h = [float(_) for _ in l_.split('-')]
                shape.append([w, h])
            img_size = 512
            p = 1.2
            x_c, y_c, r = xy(shape)
            ll = r * p
            zoom_p = (2 * ll) / img_size
            x_o, y_o = x_c - ll, y_c - ll
            img = cv2.imread(os.path.join(image_path, name))

            img_roi = img[int(y_o):int(y_o + 2*ll), int(x_o):int(x_o + 2*ll), :].copy()

            for i in range(len(shape)):
                s = shape[i]
                shape[i] = [(s[0] - x_o) / zoom_p, (s[1] - y_o) / zoom_p]
            if img_roi.size == 0:
                continue
            img_crop = cv2.resize(img_roi, (img_size, img_size))
            lab_template = np.zeros((img_size, img_size, 3))
            img_show = drawCircle(img_crop.copy(), shape, radius=1, color=(255, 255, 255), thickness=8)
            land_crop = drawCircle(lab_template, shape, radius=1, color=(255, 255, 255), thickness=8)

            f.write(str_landmark(shape, name, (img_size, img_size)))

            cv2.imwrite('{}/{}'.format(image_path_crop, name), img_crop)
            cv2.imwrite('{}/{}'.format(image_path_show, name), img_show)
            cv2.imwrite('{}/{}'.format(land_path_crop, name), land_crop)
        f.close()
        print()

    datasets = ['RaFD45', 'RaFD90', 'RaFD135']
    for dataset in datasets:
        image_path = os.path.join(dataset, 'image')
        image_path_crop = os.path.join(dataset, 'image_crop')
        image_path_show = os.path.join(dataset, 'image_show')
        land_path_crop = os.path.join(dataset, 'landmark_crop')
        land_txt = os.path.join(dataset, 'landmark.txt')
        land_norm_txt = os.path.join(dataset, 'landmark_crop.txt')
        if not os.path.exists(image_path_crop):
            os.mkdir(image_path_crop)
        if not os.path.exists(image_path_show):
            os.mkdir(image_path_show)
        if not os.path.exists(land_path_crop):
            os.mkdir(land_path_crop)
        generate()

crop_and_generate_landmark_img()
