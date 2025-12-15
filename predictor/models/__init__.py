from predictor.models.autobot import AutoBotEgo, AutoBotJoint
from predictor.models.mtr.MTR import MotionTransformer
from predictor.models.wayformer.wayformer import Wayformer
from predictor.models.constvel.constvel import ConstVel
from predictor.models.map_constvel.map_constvel import MapConstVel
from predictor.models.smart.smart import SMART
from predictor.models.ep.ep_diffuser import EPDiffuser
# from unitraj.models.smart.smart import SMART
import warnings
from pathlib import Path

__all__ = ["AutoBotEgo", "AutoBotJoint", "MotionTransformer", "Wayformer", "ConstVel", "SMART"]

name_constructor_map = {
    'autobotEgo': AutoBotEgo,
    'autobotJoint': AutoBotJoint,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    "smart": SMART,
    'constvel': ConstVel,
    'mapconstvel': MapConstVel,
    'ep_diffuser': EPDiffuser,
}


def build_model(config):
    model = name_constructor_map[config.model.model_name](
        config=config
    )

    return model

def load_predictor(config, ckpt_path, device):

    if not ckpt_path:
        return build_model(config)
    
    ckpt_path = Path(ckpt_path)

    
    if not ckpt_path.exists() and not ckpt_path.is_file():
        warnings.warn(f"Given checkpoint {ckpt_path} does not exist; Initialize a untrained model!")
        
        return build_model(config)
        
    model = name_constructor_map[config.model.model_name](
        config=config
    )
    model = type(model).load_from_checkpoint(ckpt_path, config=config, map_location=device).eval()

    return model
