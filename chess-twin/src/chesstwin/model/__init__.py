from chesstwin.config import Config
from chesstwin.encoding.moves import VOCAB_LIST
from .resnet import PolicyNetwork


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(config : Config):
    """ Reads relevant information from our ground truth in config and returns our Policy Head for inference. """
    return PolicyNetwork(
        in_planes=config.encoding.num_board_planes, 
        num_filters=config.model.num_filters,
        num_blocks=config.model.num_blocks,
        num_moves=len(VOCAB_LIST)
    )
