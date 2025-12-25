import itertools

import torch
from doctr import models
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class OCR:
    def __init__(
        self,
        det_arch='db_resnet50', reco_arch='crnn_vgg16_bn',
        pretrained=True, straighten_pages=False, debug=False, gpu=False
    ):
        super().__init__()
        self.debug = debug

        self.model = models.ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=pretrained,
            straighten_pages=straighten_pages
        )

        if gpu and torch.cuda.is_available():
            if self.debug:
                print("Using GPU")
            device = torch.device("cuda:0")
            self.model.to(device)
        else:
            if self.debug:
                print("Using CPU")

        self.model.eval()

    @staticmethod
    def json_to_dataframe(result):
        pages = pd.DataFrame.from_dict(pd.json_normalize(result.export()))
        blocks = pages.explode("blocks")
        blocks['block_idx'] = np.arange(blocks.shape[0])
        blocks['index'] = blocks['block_idx']
        blocks = blocks.set_index('index')

        blocks = blocks.join(pd.json_normalize(blocks.pop('blocks')))
        blocks = blocks.rename(columns={'geometry': 'block_geometry'})
        lines = blocks.explode("lines")
        lines['line_idx'] = np.arange(lines.shape[0])
        lines['index'] = np.arange(lines.shape[0])
        lines = lines.set_index('index')
        lines = lines.join(pd.json_normalize(lines.pop('lines')), lsuffix='.lines')
        lines = lines.rename(columns={'geometry': 'line_geometry'})
        words = lines.explode("words")
        words['word_idx'] = np.arange(words.shape[0])
        words['index'] = np.arange(words.shape[0])
        words = words.set_index('index')

        words = words.join(pd.json_normalize(words.pop('words')), lsuffix='.words')
        words = words.rename(columns={'geometry': 'word_geometry'})

        words = words.dropna(subset=['word_geometry'])
        words["word_geometry"] = words.word_geometry.apply(
            lambda x: {"x1": x[0][0], "y1": x[0][1], "x2": x[1][0], "y2": x[1][1]}
        )

        words = words.join(pd.json_normalize(words.pop('word_geometry')))

        return words

    def has_text_detector(self, images):
        out = self.model.det_predictor(images)
        return [len(o["words"]) > 0 for o in out]

    def from_image(self, doc):
        results = []
        det = self.has_text_detector(doc)
        filtered_doc = list(itertools.compress(doc, det))
        try:
            document = self.model(filtered_doc)
            i = 0
            for d in det:
                if d:
                    page = self.json_to_dataframe(document.pages[i])
                    results.append(page)
                    i += 1
                else:
                    results.append(None)
        except Exception as e:
            if self.debug:
                print(e)

        return results
