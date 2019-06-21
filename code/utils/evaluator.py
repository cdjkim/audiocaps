import colorlog
#import pprint
import contextlib

import rougescore
import numpy as np

from utils.pycocoevalcap.eval import COCOEvalCap
from utils.pycocotools.coco import COCO

#pp = pprint.PrettyPrinter().pprint


class Evaluator(object):
    def __init__(self, word_vector=None):
        self.word_vector = word_vector

    def evaluation_with_dict(self, pred_answers_dict, method="coco"):
        colorlog.info("Run evaluation...")

        # Dict to list pairs
        predictions = []
        answers = []
        for pred_answers in pred_answers_dict.values():
            prediction = pred_answers['prediction']
            answer = pred_answers['answers']

            # Type checking
            if type(prediction) == str:
                prediction = prediction.split()
            if type(answer[0]) == str:
                answer = [answer_.split() for answer_ in answer]

            predictions.append(prediction)
            answers.append(answer)

        if method == "coco":
            eval_result = self._coco_evaluation(predictions, answers)
        else:
            raise NotImplementedError

        return eval_result

    def evaluation(self, predicts, answers, method="coco"):
        """Wrapper method for evaluation

        Args:
            predicts: list of tokens list
            answers: list of tokens list. For multiple GTs, list of list of tokens list.
            method: evaluation method. ("rouge")

        Returns:
            Dictionary with metric name in key metric result in value
        """
        colorlog.info("Run evaluation...")
        if method == "rouge":
            eval_result = self._rouge_evaluation(predicts, answers)
        elif method == "coco":
            eval_result = self._coco_evaluation(predicts, answers)
        else:
            raise NotImplementedError

        #pp(eval_result)
        return eval_result

    def _rouge_evaluation(self, predicts, answers):
        rouge_1s = []
        rouge_2s = []
        rouge_ls = []
        for predict, answer in zip(predicts, answers):
            answer = [w.replace('_UNK', '_UNKNOWN') for w in answer]

            rouge_1 = rougescore.rouge_1(predict, [answer], 0.5)
            rouge_2 = rougescore.rouge_2(predict, [answer], 0.5)
            rouge_l = rougescore.rouge_l(predict, [answer], 0.5)

            rouge_1s.append(rouge_1)
            rouge_2s.append(rouge_2)
            rouge_ls.append(rouge_l)

        return {"rouge_1": np.mean(rouge_1s), "rouge_2": np.mean(rouge_2s), "rouge_l": np.mean(rouge_ls)}

    def _coco_evaluation(self, predicts, answers):
        coco_res = []
        ann = {'images': [], 'info': '', 'type': 'captions', 'annotations': [], 'licenses': ''}

        for i, (predict, _answers) in enumerate(zip(predicts, answers)):
            predict_cap = ' '.join(predict)

            if type(_answers) == str:
                _answers = [_answers]
            answer_caps = []
            for _answer in _answers:
                answer_cap = ' '.join(_answer).replace('_UNK', '_UNKNOWN')
                answer_caps.append(answer_cap)

            ann['images'].append({'id': i})
            for answer_cap in answer_caps:
                ann['annotations'].append({'caption': answer_cap, 'id': i, 'image_id': i})
            coco_res.append({'caption': predict_cap, 'id': i, 'image_id': i})

        with contextlib.redirect_stdout(None):
            coco = COCO(ann)
            coco_res = coco.loadRes(coco_res)
            coco_eval = COCOEvalCap(coco, coco_res)
            coco_eval.evaluate()

        return coco_eval.eval
