import gradio as gr
import torch
import safetensors.torch
from modules import paths,script_callbacks,script_loading,devices,shared
from modules.sd_models import *
from tqdm import tqdm
from copy import deepcopy
networks = script_loading.load_module(os.path.join(paths.extensions_builtin_dir,'Lora','networks.py'))

import scripts.clipreplacer.common as cmn

class Timer():
    def record(x,y):pass

def replace_clip():
    from modules import sd_hijack
    model = shared.sd_model

    filename = os.path.join(paths.models_path,'Stable-diffusion',cmn.clipmodel)
    sd_c = get_clip_from_checkpoint(filename)
    
    #Clear loras from loaded model
    with torch.no_grad():
        for module in model.modules():
            networks.network_restore_weights_from_backup(module)

    sd_hijack.model_hijack.undo_hijack(model)

    model.sd_checkpoint_info = modifycpinfo(model)
    
    sd_m = model.state_dict()
    
    print(f'Merging clip from \"{cmn.clipmodel}\" to \"{os.path.basename(model.sd_checkpoint_info.filename)}\"...')
    for k in tqdm(list(sd_c.keys()), desc='Merging'):
        sd_m[k] = sd_c[k]

    model.load_state_dict(sd_m)
    sd_hijack.model_hijack.hijack(model)
    model_data.set_sd_model(model)

    del sd_m; del sd_c

def reload_checkpoint(model):
    name = os.path.split(model.sd_checkpoint_info.filename)[1]
    og_cpi = get_closet_checkpoint_match(name)

    sd_unet.apply_unet("None")
    send_model_to_cpu(model)
    sd_hijack.model_hijack.undo_hijack(model)

    load_model_weights(model, og_cpi, None, Timer())

    sd_hijack.model_hijack.hijack(model)

    script_callbacks.model_loaded_callback(model)

    if not model.lowvram:
        model.to(devices.device)
    
    model_data.set_sd_model(model)
    sd_unet.apply_unet()

    
def modifycpinfo(model):
    name = os.path.basename(model.sd_checkpoint_info.filename)
    new_checkpoint_info = deepcopy(get_closet_checkpoint_match(name))

    cpname,ext = os.path.splitext(new_checkpoint_info.name)
    new_checkpoint_info.name = f"{cpname}{cmn.makesuffix(cmn.clipmodel)}{ext}"
    new_checkpoint_info.name_for_extra = f"{cpname}{cmn.makesuffix(cmn.clipmodel)}"
    new_checkpoint_info.title = f"{cpname}{cmn.makesuffix(cmn.clipmodel)}"
    return new_checkpoint_info 

versions = {
    'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight':"v1",
    'cond_stage_model.model.token_embedding.weight':"v2",
    'conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight':'xl'
}

def get_clip_from_checkpoint(filename):
    with safetensors.torch.safe_open(filename,framework='pt',device=devices.get_optimal_device_name()) as checkpoint:
        keys = checkpoint.keys()
        
        #compatibility testing
        for version in versions.keys():
            if version in keys:
                m = shared.sd_model
                if (versions[version] == 'v1' and m.is_sd1) or (versions[version] == 'v2' and m.is_sd2) or (versions[version] == 'xl' and m.is_sdxl):
                    break
        else:
            raise gr.Error(cmn.EXTENSION_NAME+f': CLIP model from \"{cmn.clipmodel}\" is not compatible with the currently loaded checkpoint.')

        clipmodel = {}
        for key in keys:
            if key.startswith('cond'):
                clipmodel[key] = checkpoint.get_tensor(key)

        return clipmodel
    
    