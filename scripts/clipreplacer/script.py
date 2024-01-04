import gradio as gr
import torch
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

    clip_filename = os.path.join(paths.models_path,cmn.CLIP_DIRECTORY,cmn.clipmodel)
    sd_c = get_state_dic(clip_filename)
    
    #compatibility testing
    tews = sd_c.get('embeddings.token_embedding.weight') 
    if tews == None:
        tews = sd_c.get('token_embedding.weight')
    x = list(tews.shape)[1]
    if x == 768 and model.is_sd1:
        prefix = "cond_stage_model.transformer.text_model."
    elif x == 1024 and model.is_sd2:
        prefix = "cond_stage_model.model."
    elif x == 1280 and model.is_sdxl:
        prefix = "conditioner.embedders.1.model"
    else:
        del sd_c
        raise gr.Error(cmn.EXTENSION_NAME+': CLIP model is not compatible with the current checkpoint.')

    #Clear loras from loaded model
    with torch.no_grad():
        for module in model.modules():
            networks.network_restore_weights_from_backup(module)

    sd_hijack.model_hijack.undo_hijack(model)

    model.sd_checkpoint_info = modifycpinfo(model)
    
    sd_m = model.state_dict()
    
    for k in tqdm(list(sd_c.keys()), desc='Merging clip to sd model'):
        sd_m[prefix+k] = sd_c[k]

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
    name = os.path.split(model.sd_checkpoint_info.filename)[1]
    new_checkpoint_info = deepcopy(get_closet_checkpoint_match(name))

    cpname,ext = os.path.splitext(new_checkpoint_info.name)
    new_checkpoint_info.name = f"{cpname}{cmn.makesuffix(cmn.clipmodel)}{ext}"
    new_checkpoint_info.name_for_extra = f"{cpname}{cmn.makesuffix(cmn.clipmodel)}"
    new_checkpoint_info.title = f"{cpname}{cmn.makesuffix(cmn.clipmodel)}"
    return new_checkpoint_info 

def get_state_dic(filename):
    if 'safetensors' in filename:
        file = safetensors.torch.load_file(filename, device=shared.device)
    else:
        file = torch.load(filename, map_location=shared.device)
    return file.get('state_dict') or file