
import io
import json
import os
import cv2
import datetime
from xml.etree import ElementTree
import numpy as np
from ast import literal_eval
import logging
import tempfile
import base64
import PIL
import PIL.Image
import PIL.ExifTags
import shutil
from cairosvg import svg2png
import svgpathtools
from typing import Union
import pandas as pd
import requests

from sticky_pi_ml.utils import datetime_to_string, string_to_datetime
from sticky_pi_ml.annotations import Annotation, DictAnnotation
from sticky_pi_ml.utils import md5


class ImageSeries(list):
    def __init__(self, device: str, start_datetime: Union[str, datetime.datetime],
                 end_datetime: Union[str, datetime.datetime]):
        super().__init__()
        self._device = device

        if not isinstance(start_datetime, datetime.datetime):
            start_datetime = string_to_datetime(start_datetime)

        if not isinstance(end_datetime, datetime.datetime):
            end_datetime = string_to_datetime(end_datetime)

        self._start_datetime = start_datetime
        self._end_datetime = end_datetime
        self._info_dict = {'device': self._device,
                           'start_datetime': self._start_datetime,
                           'end_datetime': self._end_datetime}

    def __repr__(self):
        return self.name

    @property
    def info_dict(self):
        return self._info_dict

    @property
    def start_datetime(self):
        return self._start_datetime

    @property
    def end_datetime(self):
        return self._end_datetime

    @property
    def name(self):
        return "%s.%s.%s" % (self._device,
                             datetime_to_string(self._start_datetime),
                             datetime_to_string(self._end_datetime))

    def populate_from_client(self, client, cache_image_dir=None):
        from sticky_pi_api.client import BaseClient
        assert isinstance(client, BaseClient)

        client_resp = client.get_images_with_uid_annotations_series([self._info_dict],
                                                                    what_annotation='data',
                                                                    what_image='image')
        df = pd.DataFrame(client_resp)
        if len(df) == 0:
            logging.warning('No image for %s' % self._info_dict)
            return

        if 'algo_name' not in df.columns:
            logging.info('No annotations for the requested images. Fetching all!')
            df['algo_version'] = None
            df['algo_name'] = ""

        df = df.sort_values(by=['algo_version', 'datetime'])
        df = df.drop_duplicates(subset=['id'], keep='last')
        logging.info(f'{len(df)} Images matching')

        annotated_images = []
        if not "json" in df.columns:
            logging.warning('No annotations')
            return annotated_images

        logging.info(f'{sum([0 if j is None else 1 for j in df.json])} annotations')
        for _, r in df.iterrows():
            r_dict = r.to_dict()
            if not r_dict['json']:
                continue
            if not os.path.isfile(r_dict['url']):
                if cache_image_dir is None or not os.path.isdir(cache_image_dir):
                    raise FileNotFoundError(f'The requested image appears to be a remote url: {r["url"]}.'
                                            f'For this type of resource, a valid cache image directory is needed!')

                filename = os.path.basename(r_dict['url']).split('?')[0]
                logging.info(f'Downloading {filename}')
                target = os.path.join(cache_image_dir, filename)
                if os.path.isfile(target):
                    local_md5 = md5(target)
                else:
                    local_md5 = None

                if r_dict['md5'] != local_md5:
                    resp = requests.get(r_dict['url']).content
                    with open(target, 'wb') as file:
                        file.write(resp)

                local_url = os.path.join(cache_image_dir, filename)
            else:
                local_url = r_dict['url']

            im = ImageJsonAnnotations(local_url, json_str=r_dict['json'])
            annotated_images.append(im)

        self.clear()
        for im in sorted(annotated_images, key=lambda x: x.datetime):
            self.append(im)


class Image(object):
    def __init__(self, path: str):
        self._path = path
        self._filename = os.path.basename(path)
        self._md5 = None
        file_info = self._device_datetime_info(self._filename)
        self._datetime = file_info['datetime']
        self._device = file_info['device']
        self._annotations = []
        self._metadata = None
        self._shape = None
        self._cached_image = None

    def __repr__(self):
        return "%s: %s.%s" % (self.__class__, self._device, self.datetime_str)

    def __str__(self):
        return self.__repr__()

    def _device_datetime_info(self, filename: str):
        fields = filename.split('.')

        if len(fields) != 3:
            raise Exception("Wrong file name, three dot-separated fields expected")

        device = fields[0]
        try:
            if len(device) != 8:
                raise ValueError()
            int(device, base=16)
        except ValueError:
            raise Exception("Invalid device name field in file: %s" % device)

        datetime_string = fields[1]
        try:
            date_time = datetime.datetime.strptime(datetime_string, '%Y-%m-%d_%H-%M-%S')
            # date_time = self._timezone.localize(date_time)
        except ValueError:
            raise Exception("Could not retrieve datetime from filename")

        return {'device': device,
                'datetime': date_time,
                'filename': filename}

    # when automatically annotating an image, we can tag the version
    @property
    def algo_version(self):
        try:
            return self._metadata['algo_version']
        except KeyError:
            logging.warning('No detector version')
            return None

    def tag_detector_version(self, name, version):
        # we force metadata parsing if it was not
        _ = self.metadata
        self._metadata['md5'] = self.md5
        self._metadata.update(self._device_datetime_info(self._filename))
        self._metadata['algo_name'] = name
        self._metadata['algo_version'] = version

    def annotation_dict(self, as_json: bool = True):
        try:
            metadata_to_pass = {k: self._metadata[k] for k in
                                ['device', 'datetime', 'algo_name', 'algo_version', 'md5']}

        except KeyError:
            meta_to_pass = {}

        out = {'annotations': [a.to_dict() for a in self._annotations],
               'metadata': metadata_to_pass}
        if as_json:
            out = json.dumps(out)
        return out

    @property
    def n_annotations(self):
        return len(self._annotations)

    @property
    def md5(self):
        if self._md5 is None:
            self._md5 = md5(self._path)
        return self._md5

    @property
    def annotations(self):
        return self._annotations

    @property
    def shape(self):
        if self._shape is None:
            self.read()
        return self._shape

    @property
    def device(self):
        return self._device

    @property
    def path(self):
        return self._path

    @property
    def datetime(self):
        return self._datetime

    @property
    def datetime_str(self):
        return self._datetime.strftime('%Y-%m-%d_%H-%M-%S')

    @property
    def filename(self):
        return self._filename

    def clear_cache(self, clear_annot_cached_conv=True):
        self._cached_image = None
        if clear_annot_cached_conv:
            for a in self._annotations:
                a.clear_cached_conv()

    def read(self, cache=False):
        if self._cached_image is None:
            im = self._get_array()
        else:
            im = self._cached_image

        if cache and self._cached_image is None:
            self._cached_image = im

        self._shape = im.shape
        return im

    def _get_array(self):
        out = cv2.imread(self._path)
        if out is None:
            raise Exception('Could not read image file %s' % self._path)
        return out

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._decode_metadata()
        return self._metadata

    def _decode_metadata(self):
        with PIL.Image.open(self._path) as img:
            out = {
                PIL.ExifTags.TAGS[k]: v
                for k, v in img._getexif().items()
                if k in PIL.ExifTags.TAGS
            }
            # cast to float for compatibility
            for k, v in out.items():
                if isinstance(v, PIL.TiffImagePlugin.IFDRational):
                    out[k] = float(v)

        try:
            out['Make'] = literal_eval(out['Make'])
        except ValueError as e:
            logging.warning('Missing custom metadata in %s, Make is `%s`' % (self._path, out['Make']))
        return out

    def _img_buffer(self):
        with open(self._path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string

    def to_svg(self, target, embed_jpeg=True, include_metadata=True):

        # tmp_img = tempfile.mktemp(suffix='.jpg')
        tmp_svg = tempfile.mktemp(suffix='.svg')

        try:
            height, width = self.read().shape[0:2]
            if include_metadata:
                meta = self.metadata
                desc = 'desc="' + str(self.metadata) + '"'

            else:
                desc = ''

            encoded_string = self._img_buffer()

            with open(tmp_svg, 'w+') as f:
                f.write('<svg width="' + str(width) + '"' +
                        ' height="' + str(height) + '"' +
                        ' xmlns:xlink="http://www.w3.org/1999/xlink"' +
                        ' xmlns="http://www.w3.org/2000/svg"' +
                        ' >')
                #     f.write('<metadata  id="sticky_pi"> "%s" </metadata>' % str(self.metadata()))
                if embed_jpeg:
                    f.write('<image %s width="%i" height="%i" x="0" y="0" xlink:href="data:image/jpeg;base64,%s"/>' % (
                        desc,
                        width, height, str(encoded_string, 'utf-8')))

                for a in self._annotations:
                    f.write(a.svg_element())
                f.write('</svg>')

            shutil.move(tmp_svg, target)

        except Exception as e:
            os.remove(tmp_svg)
            raise e

    def to_png(self, target, show_datetime=False, scale=1):
        tmp_png = tempfile.mktemp(suffix='.png')
        tmp_svg = tempfile.mktemp(suffix='.svg')

        try:
            self.to_svg(tmp_svg, embed_jpeg=False, include_metadata=False)
            with open(tmp_svg, 'r') as f:
                svg2png(file_obj=f, write_to=tmp_png, scale=scale)

            png = PIL.Image.open(tmp_png)
            bg = cv2.resize(self.read(), png.size)

            background = PIL.Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA))

            alpha_composite = PIL.Image.alpha_composite(background, png)
            alpha_composite.save(tmp_png, 'PNG')
            shutil.move(tmp_png, target)
        finally:
            os.remove(tmp_svg)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def set_annotations(self, annotations):
        self._annotations = annotations

    # def _get_file_fields(self, path):
    #
    #     filename = os.path.basename(path)
    #     device, date_time = device_datetime_from_filename(filename)
    #
    # return {'device': device,
    #         'datetime': date_time}


class SVGImage(Image):
    def __init__(self, path):
        super().__init__(path)
        # the shape of the image within the svg document
        # will have to scale the contours  to match the actual dimensions of the embedded image
        self._scale_in_svg = None
        self._parse_metadata()
        self._parse_annotations()

    def _parse(self, file):

        self.update(self._device_datetime_info(self._filename))
        self['md5'] = md5(self.extract_jpeg(as_buffer=True))

    def _img_buffer(self):
        encoded_string = base64.b64encode(self.extract_jpeg(as_buffer=True).read())
        return encoded_string

    def _style_to_dic(self, p):
        style = p.attrib['style'].split(';')
        d = {}
        for s in style:
            k, v = s.split(':')
            d[k] = v
        return d

    def _parse_annotations(self):
        doc = ElementTree.parse(self._path)
        self._annotations = []
        paths = doc.findall('.//{http://www.w3.org/2000/svg}path')
        for p in paths:
            style = self._style_to_dic(p)
            contours = self._svg_path_to_contour(p)
            for c in contours:
                if c is not None:
                    a = Annotation(c, style['stroke'], parent_image=self)
                    self._annotations.append(a)

    def _parse_metadata(self):
        doc = ElementTree.parse(self._path)

        ims = doc.findall('.//{http://www.w3.org/2000/svg}image')
        if len(ims) != 1:
            raise Exception("Cannot extract image from %s" % self._path)

        attrs = ims[0].attrib
        if 'w' in attrs:
            im_w = ims[0].attrib['w']
        elif 'width' in attrs:
            im_w = ims[0].attrib['width']
        else:
            raise Exception('Embedded image %s does not have width' % self._path)

        if 'h' in attrs:
            im_h = ims[0].attrib['h']
        elif 'width' in attrs:
            im_h = ims[0].attrib['height']
        else:
            raise Exception('Embedded image %s does not have height' % self._path)

        svg_im_shape = (int(im_h), int(im_w))
        jpg_im_shape = self._get_array().shape
        self._scale_in_svg = np.array(svg_im_shape) / np.array(jpg_im_shape[0:2])

        try:
            str = ims[0].attrib['desc']
            if str == '':
                raise Exception('Empty desc field')
        except KeyError:
            logging.warning('Cannot find a desc attribute in image. Maybe a legacy SVG')
            try:
                sticky_data = doc.findall('.//{http://www.w3.org/2000/svg}sticky')
                if len(sticky_data) != 1:
                    raise KeyError('One and only one sticky metadata field should exist in svg image')
                str = sticky_data[0].attrib['metadata']
            except KeyError:
                logging.warning("No metadata in %s", self._path)
                self._metadata = {}
                return

        try:
            self._metadata = literal_eval(str)

        except ValueError as e:
            logging.error('Cannot parse metadata is %s. String was is `%s`' % (self._path, self._metadata))
            logging.error(e)

        if not isinstance(self._metadata['Make'], dict):
            logging.warning('Missing custom metadata in %s, Make is `%s`' % (self._path, self._metadata['Make']))

    def _svg_path_to_contour(self, p, n_point_per_segment=2):
        string = p.attrib['d']
        tvals = np.linspace(0, 1, n_point_per_segment)
        try:
            path = svgpathtools.parse_path(string)
        except IndexError:
            logging.warning('Cannot parse svg path from string: %s in %s' % (p.attrib, self._path))
            return []

        sub_paths = []
        last_end = None
        for i in path:
            if i.start != last_end:
                sub_paths.append(svgpathtools.Path())
            sub_paths[-1].append(i)
            last_end = i.end

        out = []
        for sp in sub_paths:
            arr = np.array([s.poly()(tvals) for s in sp])
            starts = arr[0:len(sp) - 1, n_point_per_segment - 1]
            ends = arr[1:len(sp), 0]

            # these should be ~ 0 as previous points end where following start

            sum_magnitude = np.sum(np.abs(starts - ends))
            # sometimes, we have small numerical difference (1e-16) or so
            if sum_magnitude > 1e-3:
                raise Exception('SVG path interrupted %s' % str(path))
            flat = arr[:, 0:n_point_per_segment - 1].flatten()
            ctr = np.round(np.array([[flat.real / self._scale_in_svg[0], flat.imag / self._scale_in_svg[1]]])).astype(
                int)
            ctr = ctr.transpose((2, 0, 1))
            # ignore contours that do not have 3 points
            if ctr.shape[0] > 2:
                out.append(ctr)
            else:
                logging.warning("polygon with only %i points in %s: %s" % (ctr.shape[0], self._path, str(p.attrib)))
        return out

    def _decode_metadata(self):
        return self._metadata

    def _get_array(self):
        buffer = self.extract_jpeg(as_buffer=True)
        bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
        img = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_COLOR)
        return img

    def extract_jpeg(self, target=None, as_buffer=False):
        doc = ElementTree.parse(self._path)
        ims = doc.findall('.//{http://www.w3.org/2000/svg}image')
        if len(ims) != 1:
            raise Exception("Cannot extract image from %s" % target)
        return self._extract_jpeg_id(ims, 0, target, as_buffer)

    def _extract_jpeg_id(self, ims, id=0, target=None, as_buffer=False):
        utf_str = ims[id].attrib['{http://www.w3.org/1999/xlink}href'].split(',')[1]
        utf_str = utf_str.strip('\"\'')

        if as_buffer:
            buffer = io.BytesIO()
            buffer.write(base64.b64decode(utf_str))
            buffer.seek(0)
            return buffer

        with open(target, 'wb') as f:
            f.write(base64.b64decode(utf_str))


class ArrayImage(Image):
    def __init__(self, array, device, datetime):
        self._array = array
        self._device = device
        self._datetime = datetime
        self._path = None
        self._filename = None
        self._annotations = []
        self._metadata = {}
        self._shape = None
        self._cached_image = None
        self._md5 = None

    def filename(self):
        raise NotImplementedError

    def read(self, cache=True):
        self._shape = self._array.shape
        return self._array


class BufferImage(Image):
    def __init__(self, buffer, device, datetime):
        self._buffer = buffer
        self._array = None
        self._device = device
        self._datetime = datetime
        self._path = None
        self._filename = None
        self._annotations = []
        self._metadata = {}
        self._shape = None
        self._cached_image = None
        self._md5 = None

    def filename(self):
        raise NotImplementedError

    def read(self, cache=True):
        if self._array is not None:
            return self._array
        self._buffer.seek(0)
        bytes_as_np_array = np.frombuffer(self._buffer.read(), dtype=np.uint8)
        array = cv2.imdecode(bytes_as_np_array, cv2.IMREAD_COLOR)

        if cache:
            self._array = array
        self._shape = array.shape
        return array


class ImageJsonAnnotations(Image):
    def __init__(self, path, json_str=None, json_path=None):
        super().__init__(path)
        if json_path is None and json_str is None:
            raise Exception("json must be provided, either as a string or path")

        if json_path is not None:
            assert os.path.isfile(json_path)
            dic = json.load(open(json_path, 'r'))
        elif json_str is not None:
            dic = json.loads(json_str)
        else:
            raise Exception

        _ = self.metadata
        self._metadata.update(dic['metadata'])
        annot_dic_list = dic['annotations']

        for ad in annot_dic_list:
            self._annotations.append(DictAnnotation(ad, parent_image=self))
