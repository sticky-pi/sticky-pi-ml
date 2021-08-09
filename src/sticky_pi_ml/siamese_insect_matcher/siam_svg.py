import logging
import datetime
import cv2
import numpy as np
import io
from xml.etree import ElementTree
from sticky_pi_ml.image import SVGImage, BufferImage, Image
from sticky_pi_ml.annotations import Annotation
from sticky_pi_ml.utils import iou, iou_match_pairs
from shapely.geometry import Polygon
import tempfile
import base64
import os
import shutil


class DifferentDeviceError(Exception):
    pass


class SiamSVG(SVGImage):

    def __init__(self, path: str):
        """
        A class to handle annotation data for the siamese insect matcher.
        They are a special type of images that are annotated and paired.
        The two images are jpeg embedded in an SVG and stacked vertically.
        Annotations are paths. Paths can be grouped by two.
        Grouping encodes object similarity across the two frames.
        :param path: the path to the svg file to be read
        """

        self._im0 = None
        self._im1 = None
        self._annotation_pairs = []
        self._path = path
        self._get_images()
        self._parse_annots()

    def __repr__(self):
        return os.path.basename(self._path)

    @staticmethod
    def target_basename(f1, f2):
        bf1 = os.path.basename(f1)
        bf2 = os.path.basename(f2)
        return '.'.join(bf1.split('.')[0:2] + bf2.split('.')[1:3])

    @staticmethod
    def merge_two_images(im0: Image, im1: Image, dest_dir: str, include_metadata: bool = True,
                         prematch: bool = True, prematch_iou_threshold: float = 0.75):

        if im0.device != im1.device:
            raise DifferentDeviceError

        target = "%s.%s.%s.svg" % (im0.device,
                                   im0.datetime.strftime('%Y-%m-%d_%H-%M-%S'),
                                   im1.datetime.strftime('%Y-%m-%d_%H-%M-%S'))

        target = os.path.join(dest_dir, im0.device, target)
        tmp_svg = tempfile.mktemp(suffix='.svg')

        try:
            height, width = im0.read().shape[0:2]
            height2, width2 = im1.read().shape[0:2]

            assert height == height2 and width == width2

            if include_metadata:
                desc = 'desc="' + str(im0.metadata) + '"'

            else:
                desc = ''

            encoded_string1 = im0._img_buffer()
            encoded_string2 = im1._img_buffer()

            with open(tmp_svg, 'w+') as f:
                f.write('<svg width="' + str(width) + '"' +
                        ' height="' + str(height * 2) + '"' +
                        ' xmlns:xlink="http://www.w3.org/1999/xlink"' +
                        ' xmlns="http://www.w3.org/2000/svg"' +
                        ' >')
                #     f.write('<metadata  id="sticky_pi"> "%s" </metadata>' % str(self.metadata()))

                f.write('<image %s width="%i" height="%i" x="0" y="0" xlink:href="data:image/jpeg;base64,%s"/>' % (
                    desc,
                    width, height, str(encoded_string1, 'utf-8')))
                f.write('<image %s width="%i" height="%i" x="0" y="%i" xlink:href="data:image/jpeg;base64,%s"/>' % (
                    desc,
                    width, height, height, str(encoded_string2, 'utf-8')))

                pairs = []
                # simple iou matcher to simplify annotation by grouping obvious matches (iou >.75)
                if prematch:

                    an0 = im0.annotations
                    an1 = im1.annotations
                    arr = np.zeros((len(an0), len(an1)), dtype=np.float)

                    for m, a0 in enumerate(an0):
                        for n, a1 in enumerate(an1):
                            score = iou(Polygon(np.squeeze(a0.contour)),
                                        Polygon(np.squeeze(a1.contour)))
                            arr[m, n] = score

                    pairs = iou_match_pairs(arr, prematch_iou_threshold)

                    for idx, (i, j) in enumerate(pairs):
                        if i is not None:
                            an0[i].set_name('0_%03d' % i)
                        if j is not None:
                            an1[j].set_name('1_%03d' % j)
                        if i is not None and j is not None:
                            f.write("<g id='auto-grp-%i'>\n%s\n%s" % (idx,
                                                                      an0[i].svg_element(),
                                                                      an1[j].svg_element(offset=(0, height))))

                            c0 = an0[i].center
                            c1 = an1[j].center

                            f.write(
                                '<line x1="%i" y1="%i" x2="%i" y2="%i" stroke="#000"  stroke-width="3" marker-end="url(#arrowhead)"/>' % (
                                c0.real,
                                c0.imag,
                                c1.real,
                                c1.imag + height))
                            f.write("</g>")

                        elif i is not None:
                            f.write(an0[i].svg_element())
                        elif j is not None:
                            f.write(an1[j].svg_element(offset=(0, height)))

                else:
                    for i, a in enumerate(im0.annotations):
                        a.set_name('0_%03d' % i)
                        f.write(a.svg_element())

                    for i, a in enumerate(im1.annotations):
                        a.set_name('1_%03d' % i)
                        f.write(a.svg_element(offset=(0, height)))
                f.write('</svg>')

            if not os.path.isdir(os.path.dirname(target)):
                os.mkdir(os.path.dirname(target))

            shutil.move(tmp_svg, target)
            logging.info(target)
            return target
        except Exception as e:
            os.remove(tmp_svg)
            logging.error(e)
            raise e

    @property
    def annotation_pairs(self):
        return self._annotation_pairs

    def _parse_annots(self):
        doc = ElementTree.parse(self._path)
        groups = doc.findall('.//{http://www.w3.org/2000/svg}g')
        for g in groups:
            p = g.findall('.//{http://www.w3.org/2000/svg}path')
            if len(p) != 2:
                raise Exception("Not two paths in group %s in file %s" % (g.attrib['id'], self._path))

            p0, p1 = p
            c0, c1 = self._svg_path_to_contour(p0), self._svg_path_to_contour(p1)
            bbox0, bbox1 = cv2.boundingRect(c0[0]), cv2.boundingRect(c1[0])
            x0, y0, _, _ = bbox0
            x1, y1, _, _ = bbox1

            if y0 > y1:
                p0, p1 = p1, p0

            a0 = self._parse_one_path_to_annot(p0, self._im0, self._offset0)
            a1 = self._parse_one_path_to_annot(p1, self._im1, self._offset1)
            self._annotation_pairs.append((a0, a1))

    def _parse_one_path_to_annot(self, p, im, offset):
        style = self._style_to_dic(p)
        contours = self._svg_path_to_contour(p)
        all_annotations = []
        for c in contours:
            if c is not None:
                c = c - offset
                a = Annotation(c, style['stroke'], parent_image=im)
                all_annotations.append(a)
        assert len(all_annotations) == 1
        return all_annotations[0]

    def _get_images(self):

        doc = ElementTree.parse(self._path)
        ims = doc.findall('.//{http://www.w3.org/2000/svg}image')

        if len(ims) != 2:
            raise Exception("Cannot extract images from %s" % self._path)

        attrs = ims[0].attrib
        if 'w' in attrs:
            im_w = ims[0].attrib['w']
        elif 'width' in attrs:
            im_w = ims[0].attrib['width']
        else:
            raise Exception('Embedded image %s does not have width' % (self._path))

        if 'h' in attrs:
            im_h = ims[0].attrib['h']
        elif 'width' in attrs:
            im_h = ims[0].attrib['height']
        else:
            raise Exception('Embedded image %s does not have height' % (self._path))

        svg_im_shape = (int(im_h), int(im_w))

        a0, b0 = self._get_one_image(ims[0])
        a1, b1 = self._get_one_image(ims[1])

        if ims[1].attrib['y'] < ims[0].attrib['y']:
            a1, a0 = a0, a1
            b1, b0 = b0, b1

        jpg_im_shape = a0.shape
        self._scale_in_svg = np.array(svg_im_shape) / np.array(jpg_im_shape[0:2])
        self._offset0 = (0, 0)
        self._offset1 = (0, a1.shape[0])
        device, dtstr0, dtstr1, _ = os.path.basename(self._path).split('.')
        dt0 = datetime.datetime.strptime(dtstr0, '%Y-%m-%d_%H-%M-%S')
        dt1 = datetime.datetime.strptime(dtstr1, '%Y-%m-%d_%H-%M-%S')

        self._im0 = BufferImage(b0, device=device, datetime=dt0)
        self._im1 = BufferImage(b1, device=device, datetime=dt1)

        self._device = device
        self._metadata = None

    def extract_jpeg(self, target=None, as_buffer=False, id=0):
        doc = ElementTree.parse(self._path)
        ims = doc.findall('.//{http://www.w3.org/2000/svg}image')
        if len(ims) != 2:
            raise Exception("Unexpected number of images in %s" % self._path)
        return self._extract_jpeg_id(ims, id, target, as_buffer)

    def _get_buffer(self, im):
        utf_str = im.attrib['{http://www.w3.org/1999/xlink}href'].split(',')[1]
        utf_str = utf_str.strip('\"\'')

        buffer = io.BytesIO()
        buffer.write(base64.b64decode(utf_str))
        buffer.seek(0)
        return buffer

    def _get_one_image(self, im):
        buffer = self._get_buffer(im)
        bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
        img = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_COLOR)
        return img, buffer
