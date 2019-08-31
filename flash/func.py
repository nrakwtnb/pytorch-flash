
def weight_clipping(clip_value):
    def weight_clipping_(model, **args):
        for p in model.parameters():
            p.data.clamp_(-clip_value, clip_value)
    return weight_clipping_

