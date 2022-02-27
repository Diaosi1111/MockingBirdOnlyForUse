from src.MockingBirdOnlyForUse import MockingBird, Params
from pathlib import Path

params = Params("我爱你，你爱我", Path("temp.wav"))
params.synt_path = "azusa_200k.pt"
params.save_path = "test.wav"
MockingBird.init(
    Path(r"pretrained_encoder.pt"),
    Path(r"g_hifigan.pt"),
    "HifiGan",
)
out = MockingBird.genrator_voice(params)
