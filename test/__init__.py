from src.MockingBirdOnlyForUse import MockingBird, Params
from pathlib import Path

params = Params("我在干什么", Path("temp3.wav"), steps=4, min_stop_token=4, style_idx=-1)
params.synt_path = "azusa_200k.pt"
params.save_path = "test.wav"
MockingBird.init(
    Path(r"pretrained_encoder.pt"),
    Path(r"g_hifigan.pt"),
    "HifiGan",
)
out = MockingBird.genrator_voice(params)
