import torch
import numpy as np
import random
PROMPT_TEMPELTE = [
    'the {target} is {relation} the {anchor}',
    'the {target} is {relation} a {anchor}',
    'this is a {target}. placed {relation} the {anchor}',
    'there is a {target}. it is {relation} the {anchor}',
    'this is a {target} and it is {relation} the {anchor}'
]

next_to_words = [
    'next to',
    'surrounding',
    'near',
    'beside'
]


class Prompt():
    def __init__(self):
        super(Prompt, self).__init__()
        self.prompt_templete = PROMPT_TEMPELTE
        self.length = len(self.prompt_templete)
        self.next_to_dis = 2.5
        self.next_to_words = next_to_words

    @torch.no_grad()
    def getRelation(self, target_center, anchor_center):
        diff = target_center-anchor_center
        if diff[0]*diff[0]+diff[1]*diff[1] <= self.next_to_dis:
            return random.sample(self.next_to_words, 1)[0]
        relation = []
        if target_center[0] + 1 <= anchor_center[0]:
            relation.append("to the left of")
        elif target_center[0] - 1 >= anchor_center[0]:
            relation.append("to the right of")
        if target_center[1] + 1 <= anchor_center[1]:
            relation.append("in front of")
        elif target_center[1] - 1 >= anchor_center[1]:
            relation.append("behind")
        return random.sample(relation, 1)[0]

    @torch.no_grad()
    def getPrompt(self, target, target_center, anchor, anchor_center):
        relation = self.getRelation(target_center, anchor_center)
        pos = np.random.randint(low=0, high=self.length)
        return self.prompt_templete[pos].format(target=target, relation=relation, anchor=anchor)
