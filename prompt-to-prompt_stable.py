from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import cv2
import os
import math
import torch.nn as nn

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"


MY_TOKEN = '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'
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# ldm_stable = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', use_auth_token=MY_TOKEN).to(device)
ldm_stable = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', use_auth_token=MY_TOKEN).to(device)
tokenizer = ldm_stable.tokenizer


class LocalBlend:
    
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, query):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet, query)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet, query)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str, query):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, query):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(-1, keepdim=True).pow(1. / self.power)
        out = x.div(norm+1e-7)
        return out
        
        
class AttentionRelation(AttentionControl):
    
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, query):
        # if place_in_unet == 'up' and query.shape[0] == 4096 and self.cur_step == NUM_DIFFUSION_STEPS and is_cross:
        #     target_ids = tokenizer.encode(self.prompts[1])[1:-1]
        #     words = ['<start>']
        #     for cur_idx in target_ids:
        #         words.append(tokenizer.decode(cur_idx))
        #     words.append('<end>')
        #     replaced = (self.alphas.squeeze() == 0)
            
        #     attn_target = attn[attn.shape[0]//2:, :, :]
        #     replaced_len = replaced.sum()

        #     all_embed = attn_target.permute(0, 2, 1)
        #     all_embed = all_embed / all_embed.sum(axis=-1, keepdims=True)
        #     all_embed = torch.einsum('hwi,ic->hwc', all_embed, query)
        #     all_embed = self.l2norm(all_embed)

        #     replaced_embed = all_embed[:, replaced, :]
        #     replaced_embed = replaced_embed.sum(axis=1) / replaced_len

        #     sim = torch.einsum('hc,hwc->hw', replaced_embed, all_embed)
        #     h = sim.shape[0]
        #     sim = sim.sum(axis=0) / h
        #     print(sim.shape)        
            
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)

        num_pixels = 1024
        length = int(math.sqrt(num_pixels))
        if place_in_unet == 'up' and query.shape[0] == 1024 and self.cur_step == NUM_DIFFUSION_STEPS and is_cross: self.cnt += 1
        if self.cnt == 3:
            attention_maps = self.get_average_attention()
            attentions = []
            for location in ['down', 'up']:
                for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(1, -1, length, length, item.shape[-1])[0]
                        attentions.append(cross_maps)

            attentions = torch.cat(attentions, dim=0)
            attentions = attentions.sum(axis=0) / attentions.shape[0]
            # self.visualize(attentions, 3, 'sitting.jpg')
            
            tau = 0.002
            attentions = attentions.reshape(num_pixels, -1)
            attentions[attentions < tau] = 0
            attentions = attentions / (attentions.sum(0, keepdim=True)+1e-7)
            replaced = (self.alphas.squeeze() == 0)
            # print(replaced)
            embeddings = torch.einsum('ic,iw->cw', query, attentions)
            embeddings = embeddings.permute(1, 0)
            embeddings = self.l2norm(embeddings)
            replaced_embeddings = embeddings[replaced, :]
            replaced_embeddings = replaced_embeddings.sum(axis=0) / replaced_embeddings.shape[0]
            replaced_embeddings = replaced_embeddings.unsqueeze(0)
            sim = torch.einsum('wc,sc->ws', embeddings, replaced_embeddings)
            
            target_ids = tokenizer.encode(self.prompts[1])[1:-1]
            words = []
            for cur_idx in target_ids:
                words.append(tokenizer.decode(cur_idx))
            words_sim = sim[1:len(words)+1]
            words_sim = (torch.exp(1-words_sim).squeeze() - 1)
            self.words_dict = {}
            for idx, word in enumerate(words):
                if replaced[idx]: self.words_dict[word] = 0.0
                else: self.words_dict[word] = float(words_sim[idx].data.detach().cpu())
            self.default_v = max(0.0, float(0.8 - ((words_sim < 0.1).sum() / len(words)).data.detach().cpu()))
            self.cnt = 0
            self.words_dict['default_'] = self.default_v
        return attn

    def get_time_res(self):
        return self.words_dict, self.default_v

    def visualize(self, attentions, idx, name):
        idx_attention = attentions[:, :, idx]
        idx_attention = idx_attention * 255.0 / idx_attention.max()
        idx_attention = idx_attention.unsqueeze(-1).expand(*idx_attention.shape, 3)
        idx_attention = idx_attention.cpu().numpy().astype(np.uint8)
        idx_attention = np.array(Image.fromarray(idx_attention).resize((256, 256)))
        cv2.imwrite(name, idx_attention)

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionRelation, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, prompts):
        super(AttentionRelation, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.l2norm = Normalize(2)
        self.cnt = 0
        self.prompts = prompts
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            ## 等价于直接返回atte_base.unqeueeze(0)，也就是和att_replace的维度一样，由[8, 256, 256]变为[1, 8, 256, 256]
            # return 0.5 * attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape) + 0.5 * att_replace
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    @abc.abstractmethod
    def mask_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str, query):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet, query)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                # attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * self.mask_cross_attention(attn_base, attn_repalce, query)
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
    
    def mask_cross_attention(self, attn_base, att_replace):
        raise att_replace
    
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


def visualize_attn(attentions, idx, name):
        idx_attention = attentions[:, :, idx]
        idx_attention = idx_attention * 255.0 / idx_attention.max()
        idx_attention = idx_attention.unsqueeze(-1).expand(*idx_attention.shape, 3)
        idx_attention = idx_attention.cpu().numpy().astype(np.uint8)
        idx_attention = np.array(Image.fromarray(idx_attention).resize((256, 256)))
        cv2.imwrite(name, idx_attention)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace
    
    def mask_cross_attention(self, attn_base, att_replace, query):
        replaced_words = (self.alphas == 0).squeeze()
        replaced_words_len = replaced_words.sum()
        att_replace_words = att_replace[:, :, :, replaced_words]
        att_replace_query = att_replace_words.sum(axis=-1) / replaced_words_len
        att_replace_query = att_replace_query.squeeze()
        replace_query = torch.mm(att_replace_query, query)
        # 控制这里attention过softmax的温度
        tau = 0.01
        att_replace_img = torch.einsum('hc,ic->hi', replace_query, query) / tau
        att_replace_img = att_replace_img.softmax(dim=-1).unsqueeze(-1).repeat(1, 1, att_replace.shape[2])

        att_combination = torch.einsum('hcw,hcw->hcw', att_replace_img, att_replace.squeeze()) + torch.einsum('hcw,hcw->hcw',(1-att_replace_img), attn_base.squeeze())
        return att_combination.unsqueeze(0)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace
    
    def mask_cross_attention(self, attn_base, att_replace):
        raise att_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def save_images(targrt_dir, image_array, new_attention=False):
    if not os.path.exists(targrt_dir): os.mkdir(targrt_dir)
    for idx, cur_img in enumerate(image_array):
        if new_attention: img_path = os.path.join(targrt_dir, str(idx)+'_new'+'.jpg')
        else: img_path = os.path.join(targrt_dir, str(idx)+'.jpg')
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, cur_img)


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    save_images('./cross_attention', images)
    # ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    save_images('./self_attention_comp', images)
    # ptp_utils.view_images(np.concatenate(images, axis=1))
    
    
def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS,
                                                  guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE)
    save_images('./display', images)
    # ptp_utils.view_images(images)
    return images, x_t


# g_cpu = torch.Generator().manual_seed(666)
# prompts = ["A dog standing on the grass"]
# controller = AttentionStore()
# image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)
# show_cross_attention(controller, res=16, from_where=("up", "down"))


@torch.no_grad()
def image2latent(image):
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if type(image) is Image:
        image = np.array(image)
    else: image = np.squeeze(image, axis=0)
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    # input image density range [-1, 1]
    latents = ldm_stable.vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents


def get_cross_attention(prompt, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, new_attention=False):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    save_images('./cross_attention', images, new_attention)
    return images[1:len(images)-1]


def get_attention_map(prompt, latent=None, generator=None, new_attention=False):
    controller = AttentionStore()
    image, x_t = run_and_display([prompt], controller, latent=latent, run_baseline=False, generator=generator)
    return get_cross_attention(prompt, controller, res=16, from_where=("up", "down"), new_attention=new_attention), x_t, image


def calculate_steps_dict(cross_attention1, cross_attention2, prompts):
    words_1, words_2 = prompts[0].split(), prompts[1].split()
    mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
    valuable_mapper = mapper[0][1:len(words_2)+1]
    is_replace = (valuable_mapper == -1)
    if is_replace.sum() + len(words_1) < len(words_2):
        is_replace[len(words_1)+is_replace.sum():len(words_2)] = True

    padding = np.zeros_like(cross_attention2[0])
    dis = [0] * len(words_2)
    h, w, z = cross_attention1[0].shape
    for i in range(len(words_2)):
        if is_replace[i]:
            dis[i] = np.abs(cross_attention2[i]-padding).sum() / (h*w*z)
        else:
            dis[i] = np.abs(cross_attention2[i]-cross_attention1[valuable_mapper[i]-1]).sum() / (h*w*z)
    
    words_steps_dict = {}
    dis_anchor = 0
    for i in range(len(words_2)):
        if is_replace[i]: dis_anchor = max(dis_anchor, dis[i])
    dis_anchor = min(dis_anchor*1.5, 255)
    
    # 记录超过阈值的attention_map数量
    cnt_thres = 0
    for i in range(len(words_2)):
        if is_replace[i]:
            cur_weight = 0
        else:
            cur_weight = 0.8 - dis[i] / dis_anchor 
            cur_weight = max(cur_weight, 0)
            if cur_weight < 0.1: cnt_thres += 1
        words_steps_dict[words_2[i]] = float(cur_weight)
    default_steps = max(0.0, float(0.4 * (1 - cnt_thres/len(words_2) - is_replace.sum()/len(words_2))))
    words_steps_dict['default_'] = default_steps
    return words_steps_dict, default_steps


def adaptive_calculate_steps(prompts, g_cpu):
    cross_attention1, x_t, image1 = get_attention_map(prompts[0], generator=g_cpu)
    # latent1 = image2latent(image1)
    # latent1 = x_t[-1:, :, :, :]
    cross_attention2, _, _ = get_attention_map(prompts[1], latent=x_t, new_attention=True)
    # cross_attention2, _, _ = get_attention_map(prompts[1], latent=x_t)
    words_steps_dict, default_step = calculate_steps_dict(cross_attention1, cross_attention2, prompts)
    return words_steps_dict, default_step

random_seed = 888
g_cpu = torch.Generator().manual_seed(random_seed)


prompts = [
    "A man standing on the grass",
    "A man sitting on the grass , holding a cup"
]

# prompts = [
#     "a slim girl sitting on the bench",
#     "a fat girl sitting on the bench"
# ]

# prompts = [
#     "a boy on the road",
#     "a boy sitting on the road"
# ]

# prompts = [
#     "A bucket full with apples is lying on the table",
#     "A bucket a few with apples is lying on the table"
# ]

# prompts = [
#     "A dog standing on the grass",
#     "A dog sitting on the grass"
# ]

# prompts = [
#     "A boy with black hair",
#     "A boy with yellow hair"
# ]

words_steps_dict, default_step = adaptive_calculate_steps(prompts, g_cpu)

controller_ini = AttentionRelation(prompts)
_ = run_and_display([prompts[1]], controller_ini, latent=None, generator=torch.Generator().manual_seed(random_seed))
words_steps_dict, default_step = controller_ini.get_time_res()


print(words_steps_dict)
controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS,
                             cross_replace_steps=words_steps_dict, 
                             self_replace_steps=default_step)
_ = run_and_display(prompts, controller, latent=None, generator=torch.Generator().manual_seed(random_seed))


# # %%
# prompts = ["a photo of a house on a mountain",
#            "a photo of a house on a mountain at fall"]


# controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                              self_replace_steps=.4)
# _ = run_and_display(prompts, controller, latent=None, generator=torch.Generator().manual_seed(random_seed))


# # %%
# prompts = ["a photo of a house on a mountain",
#            "a photo of a house on a mountain at winter"]


# controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                              self_replace_steps=.4)
# _ = run_and_display(prompts, controller, latent=x_t)


# # %%
# prompts = ["soup",
#            "pea soup"] 

# lb = LocalBlend(prompts, ("soup", "soup"))

# controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                              self_replace_steps=.4,
#                              local_blend=lb)
# _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


# # %%
# prompts = ["a smiling bunny doll"] * 2

# ### pay 3 times more attention to the word "smiling"
# equalizer = get_equalizer(prompts[1], ("smiling",), (5,))
# controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                                self_replace_steps=.4,
#                                equalizer=equalizer)
# _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


# # %%
# prompts = ["pink bear riding a bicycle"] * 2

# ### we don't wont pink bikes, only pink bear.
# ### we reduce the amount of pink but apply it locally on the bikes (attention re-weight + local mask )

# ### pay less attention to the word "pink"
# equalizer = get_equalizer(prompts[1], ("pink",), (-1,))

# ### apply the edit on the bikes 
# lb = LocalBlend(prompts, ("bicycle", "bicycle"))
# controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                                self_replace_steps=.4,
#                                equalizer=equalizer,
#                                local_blend=lb)
# _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


# # %%
# prompts = ["soup",
#            "pea soup with croutons"] 
# lb = LocalBlend(prompts, ("soup", "soup"))
# controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                              self_replace_steps=.4, local_blend=lb)
# _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


# # %%
# prompts = ["soup",
#            "pea soup with croutons"] 


# lb = LocalBlend(prompts, ("soup", "soup"))
# controller_a = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, 
#                                self_replace_steps=.4, local_blend=lb)

# ### pay 3 times more attention to the word "croutons"
# equalizer = get_equalizer(prompts[1], ("croutons",), (3,))
# controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                                self_replace_steps=.4, equalizer=equalizer, local_blend=lb,
#                                controller=controller_a)
# _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


# # %%
# prompts = ["potatos",
#            "fried potatos"] 
# lb = LocalBlend(prompts, ("potatos", "potatos"))
# controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                              self_replace_steps=.4, local_blend=lb)
# _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)


# # %%
# prompts = ["potatos",
#            "fried potatos"] 
# lb = LocalBlend(prompts, ("potatos", "potatos"))
# controller = AttentionRefine(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8, 
#                              self_replace_steps=.4, local_blend=lb)

# ### pay 10 times more attention to the word "fried"
# equalizer = get_equalizer(prompts[1], ("fried",), (10,))
# controller = AttentionReweight(prompts, NUM_DIFFUSION_STEPS, cross_replace_steps=.8,
#                                self_replace_steps=.4, equalizer=equalizer, local_blend=lb,
#                                controller=controller_a)
# _ = run_and_display(prompts, controller, latent=x_t, run_baseline=False)
