from modules.scripts import basedir
import os

EXTENSION_NAME = 'Pick CLIP'

CLIP_DIRECTORY = "Components"
EXTENSION_ROOT = basedir()

ext2abs = lambda *x: os.path.join(EXTENSION_ROOT,*x)
makesuffix = lambda x: f"_TEMPCLIP-{x.split('.')[0]}"

clip_models=[]
clipmodel = None