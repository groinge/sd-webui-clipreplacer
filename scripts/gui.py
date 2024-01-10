import gradio as gr
from modules import scripts,paths,ui_components,script_callbacks,shared,sd_models
from modules.ui_common import create_refresh_button
import scripts.clipreplacer.common as cmn
from scripts.clipreplacer.script import *

def checkpoints_no_pickles():
    sd_models.list_models()
    return ['None'] + [checkpoint for checkpoint in sd_models.checkpoint_tiles() if checkpoint.split(' ')[0].endswith('.safetensors')]

class script(scripts.Script):
        def title(self):
            return cmn.EXTENSION_NAME

        def show(self,is_img2img):
            return scripts.AlwaysVisible
        
        def ui(self,is_img2img):
            with gr.Accordion(label=cmn.EXTENSION_NAME,open=False):
                with gr.Row(variant='default'):
                    dropdown = gr.Dropdown(value="None",label='Use CLIP from checkpoint:',choices=checkpoints_no_pickles())
                    create_refresh_button(dropdown, lambda: None, lambda: {"choices": checkpoints_no_pickles()}, "list_checkpoints_for_clip")
                    dropdown.change(
                        fn=clip_selected,
                        inputs=dropdown,
                        outputs=dropdown
                    )


            return [dropdown]
        
        def process_batch(self,p,*components,**kwargs):
            if cmn.clipmodel and cmn.makesuffix(cmn.clipmodel) not in shared.sd_model.sd_checkpoint_info.name:
                replace_clip()
            return p

"""
def on_ui_settings():
    section = ('clip_switch', cmn.EXTENSION_NAME)
    shared.opts.add_option(
        "change_clip_model",
        shared.OptionInfo(
            'None',
            "Use CLIP model from checkpoint:",
            gr.Dropdown,
            {"interactive": True,'choices': checkpoints_no_pickles()},
            refresh=checkpoints_no_pickles,
            section=section)
    )
script_callbacks.on_ui_settings(on_ui_settings)
"""

def clip_selected(mod):
    cmn.clipmodel = None if mod == "None" else mod.split(' ')[0]
    if cmn.clipmodel:
        replace_clip()
    elif '_TEMPCLIP-' in shared.sd_model.sd_checkpoint_info.name:
        reload_checkpoint(shared.sd_model)

    return gr.update(value=mod)

