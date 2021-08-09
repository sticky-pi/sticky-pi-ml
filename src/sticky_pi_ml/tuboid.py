import copy
import os
import cv2
import numpy as np
from sticky_pi_ml.utils import pad_to_square
from sticky_pi_ml.annotations import Annotation
from sticky_pi_ml.image import ImageSeries
from sticky_pi_ml.utils import string_to_datetime, md5
from typing import List


class Tuboid(list):
    def __init__(self, annotations: List[Annotation],
                 matcher_version: str,
                 parent_series: ImageSeries = None):

        if annotations is None:
            annotations = []

        annotations = sorted(annotations, key=lambda x: x.datetime)
        d = None
        for a in annotations:
            if d is not None:
                assert a.device == d
            else:
                d = a.device

        super().__init__(annotations)

        self._parent_series = parent_series
        self._device = d
        self._id = -1
        self._matcher_version = matcher_version
        self._all_timestamps = [a.datetime.timestamp() for a in self]
        self._all_timestamps = np.array(self._all_timestamps).astype(np.float)
        self._all_bboxes = np.array([a.bbox for a in self]).astype(np.float)
        self._cached_conv = {}

    def __repr__(self):
        return "%s, %s(%s), %s(%s) (N=%i)" % (self._device, self.head_datetime, self.head.center,
                                              self.tail_datetime, self.tail.center, len(self))

    def __hash__(self):
        return hash(self.__repr__())

    @classmethod
    def set_instances_id(cls, tuboids: List):
        tuboids.sort(key=lambda x: x.head_datetime)
        for i, tb in enumerate(tuboids):
            tb.set_id(i)

    @property
    def id(self):
        return self._id

    @property
    def matcher_version(self):
        return self._matcher_version

    @property
    def parent_series(self):
        return self._parent_series

    def all_annotation_sub_images(self, masked=True, scale_width=None):
        out = []
        parent_images = []
        scale_ratios = []
        centers = []
        for a in self:
            out.append(a.subimage(masked=masked))
            parent_images.append(a.parent_image)
            centers.append(a.center)
        if scale_width is None:
            return zip(out, parent_images, [1] * len(out))
        for i, o in enumerate(out):
            scale_ratios.append(scale_width / max(out[i].shape[0:2]))
            out[i] = pad_to_square(o, scale_width)

        return zip(out, parent_images, scale_ratios, centers)

    def set_id(self, i):
        self._id = i
        # for n in self.subgraph.nodes(data=True):
        #     n[1].update({'tuboid_id': i})

    def bbox_at_datetime(self, datetime):
        if datetime < self.head_datetime or datetime > self.tail_datetime:
            return None, None

        x = np.array([datetime.timestamp()]).astype(np.float)
        if datetime in self._all_timestamps:
            inferred = False
        else:
            inferred = True

        out = []
        for i in range(4):
            out.append(int(np.interp(x, self._all_timestamps, self._all_bboxes[:, i])))

        return out, inferred

    @property
    def device(self):
        return self._device

    @property
    def tail(self):
        return self[-1]

    @property
    def tail_datetime(self):
        return self[-1].datetime

    @property
    def head_datetime(self):
        return self[0].datetime

    @property
    def head(self):
        return self[0]


class TiledTuboid(list):
    _tile_width = 224
    tiles_tuboid_filename = 'tuboid.jpg'
    context_tuboid_filename = 'context.jpg'
    metadata_tuboid_filename = 'metadata.txt'
    _max_tuboid_duration = 24 * 3600

    def __init__(self, tuboid_dir):
        super().__init__()
        self._tuboid_dir = os.path.normpath(tuboid_dir)

        fields = os.path.basename(self._tuboid_dir).split('.')
        if len(fields) == 5:
            self._device, series_start_datetime, series_end_datetime, self._matcher_version,  self._tuboid_id = fields

        # legacy tuboids do not have a version field :(
        elif len(fields) == 4:
            self._device, series_start_datetime, series_end_datetime, self._tuboid_id = fields
            self._matcher_version = None
        else:
            raise Exception('Irregular tuboid dir name %s' % self._tuboid_dir)

        self._parent_series = ImageSeries(device=self._device, start_datetime=series_start_datetime,
                                          end_datetime=series_end_datetime)
        self._id = int(self._tuboid_id)

        first_shot_datetime = None

        self._n_tiles = 0

        with open(os.path.join(self._tuboid_dir, self.metadata_tuboid_filename), 'r') as f:
            while True:
                line = f.readline().rstrip()
                if not line:
                    break
                prefix, center_real, center_imag, scale = line.split(',')
                device, annotation_datetime = prefix.split('.')
                annotation_datetime = string_to_datetime(annotation_datetime)
                assert device == self._device
                if first_shot_datetime is None:
                    first_shot_datetime = annotation_datetime
                if (annotation_datetime - first_shot_datetime).total_seconds() <= self._max_tuboid_duration:
                    self._n_tiles += 1
                center = float(center_real) + 01j * float(center_imag)
                scale = float(scale)
                o = {'datetime': annotation_datetime, 'center': center, 'scale': scale}
                self.append(o)
        assert self._n_tiles > 1, f'Only {self._n_tiles} tiles found. need at least 2'

    @property
    def md5(self):
        return md5(os.path.join(self._tuboid_dir, self.metadata_tuboid_filename))

    @property
    def n_tiles(self):
        return self._n_tiles

    @property
    def directory(self):
        return self._tuboid_dir

    def iter_tiles(self):
        for i in range(self._n_tiles):
            yield self.get_tile(i)

    def get_scale(self, item: int) -> float:
        return self[item]['scale']

    def get_tile(self, item: int) -> np.ndarray:
        assert item < self._n_tiles
        row = item // 4
        col = item % 4
        im = cv2.imread(os.path.join(self._tuboid_dir, self.tiles_tuboid_filename))
        tile = im[row * self._tile_width: row * self._tile_width + self._tile_width,
                  col * self._tile_width: col * self._tile_width + self._tile_width,
                  :]

        out = copy.deepcopy(self[item])
        out['array'] = tile
        return out

    @classmethod
    def from_tuboid(cls, tuboid: Tuboid, tuboid_root_dir: str):

        assert tuboid.parent_series is not None
        assert os.path.isdir(tuboid_root_dir)

        tile_width = cls._tile_width

        series_id = tuboid.parent_series.name + '.' + tuboid.matcher_version
        tuboid_dir = "%s.%04d" % (series_id, tuboid.id)
        # tempdir = tempfile.mkdtemp()

        tuboid_dir = os.path.join(tuboid_root_dir, series_id, tuboid_dir)
        os.makedirs(tuboid_dir, exist_ok=True)

        # we only save the first day of images
        images_to_save = []
        metadata_lines = []
        for i, (im, par_im, scale, center) in enumerate(tuboid.all_annotation_sub_images(scale_width=tile_width)):
            prefix = os.path.splitext(par_im.filename)[0]
            metadata_lines.append("%s,%f,%f,%f\n" % (prefix, center.real, center.imag, scale))
            if (tuboid[i].datetime - tuboid.head_datetime).total_seconds() <= cls._max_tuboid_duration:
                images_to_save.append(im)

        assert len(metadata_lines) > 2

        n_to_save = len(images_to_save)
        n_rows = 1 + (n_to_save - 1) // 4
        out_array = np.zeros((n_rows * tile_width, tile_width * 4, 3), dtype=np.uint8)

        for i, im in enumerate(images_to_save):
            row = i // 4
            col = i % 4
            out_array[row * cls._tile_width: row * cls._tile_width + cls._tile_width,
                      col * cls._tile_width: col * cls._tile_width + cls._tile_width,
                      :] = im

        arr = np.copy(tuboid.head.parent_image_array(cache=False))
        bbox = tuboid.head.bbox
        cv2.rectangle(arr, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(0, 0, 0),
                      thickness=7)
        cv2.rectangle(arr, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(255, 255, 0),
                      thickness=4)

        with open(os.path.join(tuboid_dir, cls.metadata_tuboid_filename), 'w') as f:
            f.writelines(metadata_lines)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        cv2.imwrite(os.path.join(tuboid_dir, cls.tiles_tuboid_filename), out_array,
                    params=encode_param)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        cv2.imwrite(os.path.join(tuboid_dir, cls.context_tuboid_filename), arr,
                    params=encode_param)

        return TiledTuboid(tuboid_dir)
