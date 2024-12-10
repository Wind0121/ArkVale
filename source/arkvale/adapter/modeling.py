import warnings
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict
from functools import wraps


import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache

# use these classes just for hint
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaRMSNorm,
)

from arkvale.infer_state import InferState
from arkvale import kernels


def _arkvale_rms_norm_forward(self: LlamaRMSNorm, hidden_states):
    return kernels.rms_norm(hidden_states, self.weight, self.variance_epsilon)


def _arkvale_attn_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    infer_state: InferState = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    cur_id: int = self.layer_idx
    state = infer_state
    n_layers = state.n_layers

    if cur_id == 0:
        # decode 在这里分配好 page
        state.begin_forward(bsz, q_len)

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    )

    kvc = state.kv_caches[cur_id]
    budget = state.layer2budget[cur_id]

    n_pf_layers = state.n_prefetch_layers
    may_do_pf = q_len == 1 and n_pf_layers is not None
    do_send_pf = do_recv_pf = False
    if may_do_pf:
        pf_dst_id = cur_id + n_pf_layers
        if pf_dst_id < n_layers:
            dst_budget = state.layer2budget[pf_dst_id]
            if dst_budget is not None and dst_budget < state.n_pages:
                do_send_pf = True
        pf_src_id = cur_id - n_pf_layers
        if pf_src_id >= 0 and budget is not None and budget < state.n_pages:
            do_recv_pf = True

    if do_send_pf:
        query_states1 = (
            state.attn_layers[pf_dst_id]
            .q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
        )
        kernels.qkq_apply_rotary_in_place(
            query_states,
            key_states,
            query_states1,
            kvc.seq_len,
            rope_scale=self.rotary_emb.scaling_factor,
            rope_theta=self.rotary_emb.base,
        )
        scores = state.estimate_scores(pf_dst_id, query_states1)
        eids, rids = state.select_topk(pf_dst_id, scores)
        if rids[..., 0].any():
            rids = rids.cpu()
            state.on_decode_prefetch[cur_id % (n_pf_layers + 1)] = True
            with torch.cuda.stream(state.prefetch_streams[cur_id % (n_pf_layers + 1)]):
                # state.estimate_select_recall(pf_dst_id, query_states1)
                state.recall(pf_dst_id, eids, rids)
    else:
        kernels.qk_apply_rotary_in_place(
            query_states,
            key_states,
            kvc.seq_len,
            rope_scale=self.rotary_emb.scaling_factor,
            rope_theta=self.rotary_emb.base,
        )

    if q_len > 1:
        state.attn_layers[cur_id] = self
        # 注意这里使用的是 state.alloc_page，在 gpu 上分配 page id
        kvc.prefill_alloc_n_tokens(q_len, state.alloc_page)

    # 这里就是将 kv states 添加到 cache 的入口
    state.append_paged_kv_cache(cur_id, key_states, value_states)

    if q_len > 1:
        if budget is not None:
            with torch.cuda.stream(state.prefill_backup_stream):
                # 在 prefill backup 流上进行操作，异步地拷贝到 CPU 中
                state.prefill_backup_pages(cur_id)
                # event 用于监控 cuda 流
                evt = torch.cuda.Event()
                evt.record(state.prefill_backup_stream)
                # 进行记录，backup 未完成是不能进行 evict 的
                state.prefill_backup_events[cur_id] = evt
            # 计算 digest，并将其存放到 digest cache 中
            state.prefill_save_digests(cur_id, key_states)
        # 进行 prefill 的注意力计算
        attn_output = state.prefill_sdpa(cur_id, query_states)
        # 用 query 最后一个 token 来筛选 top-k 个 prefill page，防止超出 budget
        infer_state.prefill_evict_extra_pages(
            cur_id, query_states[:, -1:, ...].contiguous()
        )
    else:
        attn_page_ids = kvc.c2p
        if budget is not None and kvc.n_pages > budget:     # page 大小超出 budget 后，就需要考虑 select 了
            # 这个分支到后面是必须进入的
            if do_recv_pf:
                if state.on_decode_prefetch[pf_src_id % (n_pf_layers + 1)]:
                    state.default_stream.wait_stream(
                        state.prefetch_streams[pf_src_id % (n_pf_layers + 1)]
                    )
                    state.on_decode_prefetch[pf_src_id % (n_pf_layers + 1)] = False
            else:
                _, eids, _ = state.estimate_select_recall(cur_id, query_states)
            # if state.use_sparse_attn:
            #     attn_page_ids = eids
            assert not state.use_sparse_attn

        attn_output = state.decode_sdpa(cur_id, query_states, attn_page_ids)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    if cur_id == n_layers - 1:
        state.end_forward(bsz, q_len)

    return attn_output, attn_weights, past_key_value


def enable_arkvale(
    self: LlamaForCausalLM,
    dtype: torch.dtype,
    device: torch.device,
    page_size=32,
    infer_state: InferState = None,
    **kwargs,
):
    # infer_state 一般都是 None
    if infer_state is None:
        config = self.model.config
        # 依据 model config 初始化 InferState
        infer_state = InferState(
            n_layers=config.num_hidden_layers,
            n_qo_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            page_size=page_size,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    # lm_head 是用于将 hidden_state 转化到 vocab_size 的线性层
    if hasattr(self, "lm_head"):
        _lm_head_forward = self.lm_head.forward
        # 意思是只对最后一个 token 进行转换，毕竟前面的 token 转换了也不会用
        self.lm_head.forward = lambda x: _lm_head_forward(x[:, -1:, :])

    for mod in self.modules():
        mod_cls = str(mod.__class__)
        # 遍历所有 module，对 Attention 和 Norm 改造
        if "Attention" in mod_cls:
            # 用 arkvale attention 替代
            mod.forward = (
                lambda mod: lambda *args, **kwargs: _arkvale_attn_forward(
                    mod, *args, infer_state=infer_state, **kwargs
                )
            )(mod)
        elif "RMSNorm" in mod_cls:
            # 用 arkvale norm 替代
            mod.forward = (
                lambda mod: lambda *args, **kwargs: _arkvale_rms_norm_forward(
                    mod, *args, **kwargs
                )
            )(mod)

    # 将 LlamaForCausalLM 的函数进行替代
    _old_self_prepare_inputs_for_generation = self.prepare_inputs_for_generation
    _old_self_forward = self.forward

    # 装饰器用于复制传入函数的 meta data，主要是 doc 和 name
    @wraps(_old_self_prepare_inputs_for_generation)
    def _new_self_prepare_inputs_for_generation(input_ids, *args, **kwargs):
        # 实际上没有什么改动，就是将 use_cache 置为 False，并且禁止使用 past_key_values
        kwargs["use_cache"] = False
        past_kv = kwargs.get("past_key_values", None)
        if past_kv is not None:
            assert past_kv == "dummy"
            input_ids = input_ids[:, -1:]
            kwargs["past_key_values"] = None
        return _old_self_prepare_inputs_for_generation(input_ids, *args, **kwargs)

    @wraps(_old_self_forward)
    def _new_self_forward(*args, **kwargs):
        ret = _old_self_forward(*args, **kwargs)    # 正常调用
        ret["past_key_values"] = "dummy"            # 返回的 kv cache 置为 dummy
        return ret

    # monkey patch
    self.prepare_inputs_for_generation = _new_self_prepare_inputs_for_generation
    self.forward = _new_self_forward

    return self
