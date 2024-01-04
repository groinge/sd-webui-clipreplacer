import gradio as gr
from modules import scripts,paths,ui_components,script_callbacks,shared

import scripts.clipreplacer.common as cmn
from scripts.clipreplacer.script import *

class script(scripts.Script):
        def title(self):
            return cmn.EXTENSION_NAME

        def show(self,is_img2img):
            return scripts.AlwaysVisible
        
        def ui(self,is_img2img):
            with gr.Accordion(label=cmn.EXTENSION_NAME,open=False):
                with gr.Row(variant='default'):
                    get_clip_models()
                    dropdown = gr.Dropdown(value="None",choices=cmn.clip_models)
                    refresh_button = ui_components.ToolButton(value='\U0001f504')
                    dropdown.change(
                        fn=clip_selected,
                        inputs=dropdown,
                        outputs=dropdown
                    )
                    refresh_button.click(
                        fn=lambda: gr.update(choices=get_clip_models()),
                        outputs=dropdown
                    )
                if len(cmn.clip_models) == 1:
                    with gr.Row():
                        ui_components.FormHTML(value="CLIP models should be placed in the webui/models/components folder and have the file-extension .clip.pt")
                        ui_components.FormHTML(value="You can extract CLIP models from your checkpoints using the stable-diffusion-webui-model-toolkit extension.")

            return [dropdown]
        
        def process_batch(self,p,*components,**kwargs):
            cmn.clipname = components[0]
            if cmn.clipname != 'None' and cmn.makesuffix(cmn.clipname) not in shared.sd_model.sd_checkpoint_info.name:
                replace_clip()
            return p

def on_ui_settings():
    section = ('clip_switch', cmn.EXTENSION_NAME)
    shared.opts.add_option(
        "change_clip_model",
        shared.OptionInfo(
            'None',
            "Change clip model",
            gr.Dropdown,
            {"interactive": True,'choices': cmn.clip_models},
            refresh=get_clip_models,
            section=section)
    )
script_callbacks.on_ui_settings(on_ui_settings)

def clip_selected(mod):
    cmn.clipmodel = None if mod == "None" else mod
    model = shared.sd_model
    if cmn.clipmodel:
        replace_clip()
    elif '_TEMPCLIP-' in model.sd_checkpoint_info.name:
        reload_checkpoint(model)

    return gr.update(value=mod)

def get_clip_models():
    cmn.clip_models = []
    for file in os.listdir(os.path.join(paths.models_path,'Components')):
        if file.endswith(('.clip.pt','.clip-v2.pt')):
            cmn.clip_models.append(file)
    cmn.clip_models.sort()
    cmn.clip_models = ['None'] + cmn.clip_models
    return cmn.clip_models

