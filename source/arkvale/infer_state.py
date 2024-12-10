from typing import List, Union, Dict, Tuple
from collections import defaultdict

import torch
from torch import Tensor

from .kv_cache import KvPool, KvCache
from . import kernels
from . import utils

Digest = Tuple[Tensor, Tensor]


class InferState:
    def __init__(
        self,
        n_layers,
        n_qo_heads,
        n_kv_heads,
        head_dim,
        page_size,
        dtype: torch.dtype,
        device: torch.device,
        page_budgets: Union[int, List[int]] = None,
        n_max_pages=None,
        n_unlimited_layers=None,
        n_max_bytes=None,
        page_topks: Union[int, List[int]] = None,
        n_max_cpu_pages=None,
        n_max_cpu_bytes=None,
        n_sink_pages=2,
        n_win_pages=2,
        use_sparse_attn=False,
        n_prefetch_layers=None,
        group_size=None,
        n_groups=None,
        **kwargs,
    ) -> None:
        self.n_layers = n_layers
        self.n_qo_heads = n_qo_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.dtype = dtype
        self.device = device

        # 计算 GPU 的最大 page 数
        if n_max_pages is None:
            assert n_max_bytes is not None
            n_max_pages = n_max_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
            # 这里就表示一个 kv page 的 byte size
            # kv state: (batch_size, seq_len, n_kv_heads, head_dim)，通常 n_kv_heads == n_heads，n_kv_heads * head_dim == hidden_size
        # 计算 CPU 的最大 page 数
        if n_max_cpu_pages is None:
            assert n_max_cpu_bytes is not None
            n_max_cpu_pages = n_max_cpu_bytes // (
                2 * page_size * n_kv_heads * head_dim * dtype.itemsize
            )
        # unlimited_layers 应该表示没有内存限制的 layer
        if n_unlimited_layers is None:
            n_unlimited_layers = 0
        # 除非已经指定好了所有 layer 的 page_budget，否则就全部层一样
        if not isinstance(page_budgets, (list, tuple)):
            page_budgets = [None] * n_unlimited_layers + [page_budgets] * (
                n_layers - n_unlimited_layers
            )
        # top-k 默认为 budget // 2
        # 这里是一个新的语法，如果 b 为假值，就返回 b，否则就返回 b // 2
        if page_topks is None:
            page_topks = [b and b // 2 for b in page_budgets]
        elif not isinstance(page_topks, (list, tuple)):
            # 给每个 layer 指定 top-k
            page_topks = [None] * n_unlimited_layers + [page_topks] * (
                n_layers - n_unlimited_layers
            )

        self.page_size = page_size
        self.n_max_pages = n_max_pages
        self.layer2budget = page_budgets
        self.budget2layers = defaultdict(list)
        for i, b in enumerate(page_budgets):
            self.budget2layers[b].append(i)
        self.layer2topk = page_topks
        self.n_sink_pages = n_sink_pages    # 暂不确定
        self.n_win_pages = n_win_pages      # 暂不确定
        assert n_win_pages >= 2
        self.use_sparse_attn = use_sparse_attn  # 通常为 False，应该是某种 attn 实现

        for i in range(n_layers):
            b = self.layer2budget[i]
            k = self.layer2topk[i]
            if b is not None:
                assert k is not None and k < b
                k = k - n_sink_pages - (n_win_pages - 1)    # 不是很理解，给 sink page 和 win page 预留空间
                assert k > 0
                self.layer2topk[i] = k

        self.layout = "NHD" # (n, head_num, dim) ?
        self._i32 = dict(dtype=torch.int32, device=self.device)
        self._u8 = dict(dtype=torch.uint8, device=self.device)
        self._fp = dict(dtype=self.dtype, device=self.device)

        self._pool = KvPool(n_max_pages, page_size, n_kv_heads, head_dim, dtype, device)    # 创建 KvPool
        self.kv_caches: List[KvCache] = [None] * self.n_layers  # 每一层一个 KvCache
        self.dg_caches: List[KvCache] = [None] * self.n_layers  # 每一层一个 digest cache
        self._cpu_pool = KvPool(
            n_max_cpu_pages, page_size, n_kv_heads, head_dim, dtype, torch.device("cpu")
        )
        self.cpu_kv_caches: List[KvCache] = [None] * self.n_layers
        self.topk_dout: Tensor = None
        self.topk_iout: Tensor = None
        self.topk_newi: Tensor = None
        self.topk_rids: Tensor = None
        self.topk_buff: Tensor = None

        self.kv_last_page_len = 0
        self.kv_last_page_lens: Tensor = None
        # budget->Tensor 的 dict，给 FlashInfer 准备的 meta data，可以跳过看
        self.kv_indptrs_tab: Dict[int, Tensor] = {b: None for b in self.budget2layers}
        self.kv_decode_indptrs_tab: Dict[int, Tensor] = {
            b: None for b in self.budget2layers
        }

        self.dg_last_page_len = 0
        self.dg_last_page_lens: Tensor = None
        # 同为给 FlashInfer 准备的 meta data，可以跳过看
        self.dg_indptrs: Tensor = None

        # workspace buffer，给 FlashInfer 准备
        wbufs = [torch.empty(16 * 1024 * 1024, **self._u8) for _ in self.budget2layers]
        self.prefill_handler = kernels.BatchPrefillWithPagedKVCacheWrapper(
            wbufs[0], self.layout
        )
        self.prefill_handler_tab = {
            b: kernels.BatchPrefillWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }
        self.decode_handler_tab = {
            b: kernels.BatchDecodeWithPagedKVCacheWrapper(w, self.layout)
            for b, w in zip(self.budget2layers, wbufs)
        }

        self.n_prefetch_layers = n_prefetch_layers  # 似乎是个 None
        self.attn_layers = [None] * n_layers

        # 不同的流，用于控制传输任务
        self.default_stream = torch.cuda.default_stream(self.device)
        self.prefill_backup_stream = torch.cuda.Stream(self.device)
        # event 用于记录 stream 是否完成
        self.prefill_backup_events = [None] * self.n_layers
        self.prefill_evicted_pages = [None] * self.n_layers
        self.decode_backup_stream = torch.cuda.Stream(self.device)
        self.prefetch_streams = None
        self.on_decode_prefetch = None
        # prefetch layer 不为 None 才能定义预取 stream
        if n_prefetch_layers is not None:
            self.prefetch_streams = [
                torch.cuda.Stream(self.device) for _ in range(n_prefetch_layers + 1)
            ]
            self.on_decode_prefetch = [False] * (n_prefetch_layers + 1)

        # 用于 GQA，暂时不用看
        assert group_size is None or n_groups is None
        if group_size is None and n_groups is None:
            n_groups = 1
            group_size = n_kv_heads
        elif group_size is None:
            assert n_kv_heads % n_groups == 0
            group_size = n_kv_heads // n_groups
        else:
            assert n_kv_heads % group_size == 0
            n_groups = n_kv_heads // group_size
        self.group_size = group_size
        self.n_groups = n_groups

    @property
    def seq_len(self):
        return self.kv_caches[0].seq_len

    @property
    def n_pages(self):
        return self.kv_caches[0].n_pages

    @property
    def batch_size(self):
        return self.kv_caches[0].batch_size

    def _prepare_prefill(self, bsz, q_len):
        # 每次 prefill 代表一批新的请求，所以就清空 pool
        self._pool.clear()
        # 给每个层创建 KvCache
        self.kv_caches = [
            KvCache(
                self._pool,
                bsz,
                self.layer2budget[i],
                self.n_sink_pages,
                self.n_win_pages,
                n_groups=self.n_groups,
            )
            for i in range(self.n_layers)
        ]
        # 给每个层创建 digest cache
        self.dg_caches = [
            self.layer2budget[i] and KvCache(self._pool, bsz)
            for i in range(self.n_layers)
        ]
        self._cpu_pool.clear()
        # 创建 cpu KvCache
        self.cpu_kv_caches = [
            self.layer2budget[i] and KvCache(self._cpu_pool, bsz)
            for i in range(self.n_layers)
        ]

        # 用于 top-k 的一些信息
        max_topk = max([k for k in self.layer2topk if k] + [0])
        max_cap = max([b for b in self.layer2budget if b] + [0])
        bsz1 = bsz * self.n_groups
        self.topk_dout = torch.empty([bsz1 * max_cap], **self._fp)
        self.topk_iout = torch.empty(
            [bsz1 * (max_topk + self.n_sink_pages + self.n_win_pages)], **self._i32
        )
        self.topk_newi = torch.empty([bsz1 * max_cap], **self._fp)
        self.topk_rids = torch.empty([bsz1 * (max_topk + 1)], **self._i32)
        self.topk_buff = torch.empty([bsz1, 1 << 10], **self._u8)

        # we do not pre-allocate real kv-pages before prefill
        # cpu kv cache 分配对应的 page id
        [kvc.prefill_alloc_n_tokens(q_len) for kvc in self.cpu_kv_caches if kvc]
        # 获取 num page
        n_kv_pages = (q_len + self.page_size - 1) // self.page_size
        # 获取最后一个 page 的 token 长度
        self.kv_last_page_len = (q_len - 1) % self.page_size + 1
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        for b in self.kv_indptrs_tab:
            self.kv_indptrs_tab[b] = kv_indptr = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )

        # 已经填充的 page num
        n_filled_kv_pages = n_kv_pages - 1
        # 给 digest cache 分配 digest 的 page id
        [
            # 注意这里填的 token num 是 n_filled_kv_pages，和论文中的设计一致，每个 page 只会有一个 token 大小的 digest
            dgc.prefill_alloc_n_tokens(n_filled_kv_pages, self.alloc_page)
            for dgc in self.dg_caches
            if dgc
        ]
        # 获取 digest page num
        n_dg_pages = (n_filled_kv_pages + self.page_size - 1) // self.page_size
        # 获取最后一个 digest page 的 token 长度
        self.dg_last_page_len = (n_filled_kv_pages - 1) % self.page_size + 1
        self.dg_last_page_lens = torch.tensor(
            [self.dg_last_page_len] * bsz, **self._i32
        )
        self.dg_indptrs = torch.arange(0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32)

        qo_indptr = torch.arange(0, bsz * q_len + 1, q_len, **self._i32)
        # 调用 kernel 的 begin_forward
        self.prefill_handler.begin_forward(
            qo_indptr,
            kv_indptr,
            self.kv_last_page_lens,
            self.n_qo_heads,
            self.n_kv_heads,
            self.head_dim,
        )

    def _finish_prefill(self, bsz, q_len):
        # 没有什么实质性的工作，主要还是调用 kernel 的 end_forward
        for b, ls in self.budget2layers.items():
            n_kv_pages = utils.all_eq(self.kv_caches[l].n_real_pages for l in ls)
            self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = torch.arange(
                0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
            )
            if b is not None and self.use_sparse_attn:
                # 这个分支基本不看
                topk = utils.all_eq(self.layer2topk[l] for l in ls)
                n_kv_pages = min(
                    self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                )
                self.kv_decode_indptrs_tab[b] = torch.arange(
                    0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                )
        self.prefill_handler.end_forward()

    def _prepare_decode(self, bsz):
        # 用于同步上一个 decode 的 backup 完成
        if self.kv_last_page_len + 1 >= self.page_size:
            self.default_stream.wait_stream(self.decode_backup_stream)
        # 拿到之前 KV Cache 所存储的 page num
        pre = [kvc.n_real_pages for kvc in self.kv_caches]
        # 所有层的 KV Cache 都分配一个 token
        n_new_kv_pages = utils.all_eq(
            kvc.decode_alloc_1_token(self.alloc_page) for kvc in self.kv_caches
        )
        self.kv_last_page_len = utils.all_eq(
            kvc.last_page_len for kvc in self.kv_caches
        )
        self.kv_last_page_lens = torch.tensor(
            [self.kv_last_page_len] * bsz, **self._i32
        )
        # 所有层 cpu kv cache 都分配一个 token
        [kvc.decode_alloc_1_token() for kvc in self.cpu_kv_caches if kvc]

        if n_new_kv_pages > 0:
            assert n_new_kv_pages == 1

            cur = [kvc.n_real_pages for kvc in self.kv_caches]
            for b, ls in self.budget2layers.items():
                n_new_kv_real_pages = utils.all_eq(cur[l] - pre[l] for l in ls)
                # 如果分配了新的 page id，就进入以下分支，否则说明只是替换了某个 page
                if n_new_kv_real_pages > 0:
                    n_kv_pages = self.kv_caches[ls[0]].n_real_pages
                    self.kv_decode_indptrs_tab[b] = self.kv_indptrs_tab[b] = (
                        torch.arange(0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32)
                    )
                    if b is not None and self.use_sparse_attn:
                        topk = utils.all_eq(self.layer2topk[l] for l in ls)
                        n_kv_pages = min(
                            self.n_sink_pages + topk + self.n_win_pages, n_kv_pages
                        )
                        self.kv_decode_indptrs_tab[b] = torch.arange(
                            0, bsz * n_kv_pages + 1, n_kv_pages, **self._i32
                        )
            # 获取所有的 digest cache
            dg_caches = [dgc for dgc in self.dg_caches if dgc]
            # if len(dg_caches) > 0 and self.kv_last_page_len == self.page_size:
            if len(dg_caches) > 0:
                # 所有的 digest cache 也分配一个 token
                n_new_dg_pages = utils.all_eq(
                    dgc.decode_alloc_1_token(self.alloc_page) for dgc in dg_caches
                )
                self.dg_last_page_len = utils.all_eq(
                    dgc.last_page_len for dgc in dg_caches
                )
                self.dg_last_page_lens = torch.tensor(
                    [self.dg_last_page_len] * bsz, **self._i32
                )
                if n_new_dg_pages > 0:
                    assert n_new_dg_pages == 1
                    n_dg_pages = utils.all_eq(dgc.n_real_pages for dgc in dg_caches)
                    self.dg_indptrs = torch.arange(
                        0, bsz * n_dg_pages + 1, n_dg_pages, **self._i32
                    )

                with torch.cuda.stream(self.decode_backup_stream):
                    # decode backup
                    [self.decode_backup_1_page(l) for l in range(self.n_layers)]
                [self.decode_save_1_digest(l) for l in range(self.n_layers)]

        # 通常来说只会有一个 budget 和 handle
        for b, h in self.decode_handler_tab.items():
            h.begin_forward(
                self.kv_decode_indptrs_tab[b],
                self.kv_last_page_lens,
                self.n_qo_heads,
                self.n_kv_heads,
                self.head_dim,
                self.page_size,
                data_type=self.dtype,
            )

    def _finish_decode(self, bsz):
        for handler in self.decode_handler_tab.values():
            handler.end_forward()

    def begin_forward(self, bsz, q_len):
        if q_len > 1:
            self._prepare_prefill(bsz, q_len)
        else:
            self._prepare_decode(bsz)

    def end_forward(self, bsz, q_len):
        if q_len > 1:
            self._finish_prefill(bsz, q_len)
        else:
            self._finish_decode(bsz)

    def append_paged_kv_cache(self, layer_idx: int, keys: Tensor, vals: Tensor):
        kvc = self.kv_caches[layer_idx]
        # 通过 flashinfer 内核来完成 kv cache 的实际添加
        kernels.append_paged_kv_cache(
            keys,
            vals,
            kvc.buffer,
            kvc.c2p,
            self.kv_indptrs_tab[self.layer2budget[layer_idx]],
            self.kv_last_page_lens,
            self.layout,
        )

    def save_digests(self, layer_idx: int, digest: Digest):
        dgc = self.dg_caches[layer_idx]
        kernels.append_paged_kv_cache(
            *digest,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            self.layout,
        )

    # 根据 query 与 digest 计算当前层的 page score
    def estimate_scores(
        self, layer_idx: int, query_states: Tensor, n_groups: int = None
    ):
        if n_groups is None:
            n_groups = self.n_groups
        dgc = self.dg_caches[layer_idx]
        return kernels.estimate_scores(
            query_states,
            dgc.buffer,
            dgc.c2p,
            self.dg_indptrs,
            self.dg_last_page_lens,
            dgc.seq_len,
            self.layout,
            n_groups,
        )

    def select_topk(self, layer_idx: int, scores: Tensor):
        bsz = self.batch_size * self.n_groups
        budget = self.layer2budget[layer_idx]
        topk = self.layer2topk[layer_idx]
        kvc = self.kv_caches[layer_idx]
        ns = self.n_sink_pages
        nw = self.n_win_pages
        eids_range = topk + ns + nw
        dout = self.topk_dout[: bsz * topk].view(bsz, topk)
        eids = self.topk_iout[: bsz * eids_range].view(bsz, eids_range)
        newi = self.topk_newi[: bsz * budget].view(bsz, budget)
        rids = self.topk_rids[: bsz * (topk + 1)].view(bsz, topk + 1)
        buff = self.topk_buff
        scores = scores.reshape(bsz, -1)
        cc2gp = kvc.cc2gp.reshape(bsz, -1)
        gc2cc = kvc.gc2cc.reshape(bsz, -1)
        kernels.select_topk(
            scores, dout, eids, newi, cc2gp, gc2cc, rids, buff, topk, ns, nw
        )
        return eids, rids

    def recall(self, layer_idx: int, eids: Tensor, rids: Tensor):
        nw = self.n_win_pages
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        bsz = kvc.batch_size
        gs = self.group_size
        ng = self.n_groups
        eids = eids.reshape(bsz, ng, -1)
        rids = rids.reshape(bsz, ng, -1)

        for i in range(kvc.batch_size):
            for j in range(self.n_groups):
                # 0 号位存储 recall num
                nr = rids[i, j, 0]
                if nr == 0:
                    continue
                heads = slice(j * gs, (j + 1) * gs)
                for ei, ri in zip(eids[i, j, -(nr + nw) : -nw], rids[i, j, 1 : nr + 1]):
                    # ei 是 page id，ri 是 cache id
                    kvc.pool[ei][..., heads, :].copy_(
                        cpu_kvc[i, ri][..., heads, :], non_blocking=True
                    )

    def estimate_select_recall(self, layer_idx: int, q: Tensor):
        # 计算 page 得分
        scores = self.estimate_scores(layer_idx, q)
        # 获取 evict page id 和 recall page id
        # 其中 evict page id 就是得分的排序
        eids, rids = self.select_topk(layer_idx, scores)
        self.recall(layer_idx, eids, rids)
        return scores, eids, rids

    # 这就是在 GPU 与 CPU 之间的拷贝函数
    def prefill_backup_pages(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        # 遍历两个 kv cache 的所有 page id
        for ci, gi in zip(cpu_kvc.c2p.reshape(-1), kvc.c2p.reshape(-1)):
            # 没想到最后还是落到了 Tensor.copy_ 这么个简单的函数上
            # non_blocking=True 表示异步进行
            cpu_kvc.buffer[ci].copy_(kvc.buffer[gi], non_blocking=True)

    def decode_backup_1_page(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        cpu_kvc = self.cpu_kv_caches[layer_idx]
        if kvc.budget is None:
            return
        for i in range(kvc.batch_size):
            # -1 是最新添加的 page，-2 表示倒数第二个 page，肯定是满了的
            cpu_kvc[i, -2].copy_(kvc[i, kvc.evict_idx], non_blocking=True)

    def _summarize_keys(self, filled_keys: Tensor) -> Digest:
        maxs = filled_keys.max(dim=2).values
        mins = filled_keys.min(dim=2).values
        centers = (maxs + mins) / 2
        dists = (
            (
                centers.reshape(*filled_keys.shape[:2], 1, -1, self.head_dim)
                - filled_keys
            )
            .abs()
            .mean(dim=2)
        )
        mins = centers - dists
        maxs = centers + dists
        return maxs, mins

    def prefill_save_digests(self, layer_idx: int, keys: Tensor):
        if self.layer2budget[layer_idx] is None:
            return
        bsz, q_len, *_ = keys.shape
        n_filled_pages = (q_len + self.page_size - 1) // self.page_size - 1
        # 获取已经填充好的 key tokens
        filled_keys = keys[:, : n_filled_pages * self.page_size, ...].reshape(
            bsz, n_filled_pages, self.page_size, self.n_kv_heads, self.head_dim
        )
        # _summarize_keys 对 key 获取 digest
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def decode_save_1_digest(self, layer_idx: int):
        kvc = self.kv_caches[layer_idx]
        if kvc.budget is None:
            return
        filled_keys = torch.cat(
            [kvc[i, kvc.evict_idx][:1][None, ...] for i in range(self.batch_size)],
            dim=0,
        )
        self.save_digests(layer_idx, self._summarize_keys(filled_keys))

    def prefill_evict_extra_pages(self, layer_idx: int, query_states: Tensor):
        kvc = self.kv_caches[layer_idx]
        bsz = kvc.batch_size
        ng = self.n_groups
        if kvc.budget is None:
            return
        if kvc.n_real_pages > kvc.budget:   # n_max_pages 和 budget 是不一样的，所以可以允许 real_page 短暂超出 budget
            # 如果 prefill 阶段的 page 就超出了 budget，就需要提前进行 top-k select
            ns = max(2, kvc.n_sink_pages)
            nw = kvc.n_win_pages
            assert ns + nw <= kvc.budget
            topk = kvc.budget - ns - nw
            # 相当于清空了映射关系
            kvc.c2p = torch.empty([bsz, kvc.budget], **kvc._i32)
            kvc.gc2cc = torch.empty([bsz, kvc.budget], **kvc._i32)
            ev_gpi = kvc.cc2gp.clone()
            kvc.cc2gp.fill_(-1)
            dout = self.topk_dout[: bsz * topk].view(bsz, topk)
            buff = self.topk_buff
            # 拿到所有 page 的 score
            scores = self.estimate_scores(layer_idx, query_states, n_groups=1)
            scores = scores.reshape(bsz, -1)
            # 选取 top-k 个页面，并修改映射关系
            kernels.prefill_select_topk(
                scores, dout, kvc.c2p, kvc.cc2gp, ev_gpi, kvc.gc2cc, buff, topk, ns, nw
            )
            self.prefill_evicted_pages[layer_idx] = ev_gpi
        # 这里就对 cc2gp 和 gc2cc 进行了 reshape
        kvc.cc2gp = (
            kvc.cc2gp.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.cc2gp.shape[-1])
            .contiguous()
        )
        kvc.gc2cc = (
            kvc.gc2cc.reshape(bsz, 1, -1)
            .expand(bsz, ng, kvc.gc2cc.shape[-1])
            .contiguous()
        )

    def alloc_page(self):
        # 有空闲就直接分配，否则就要进行 evict
        if len(self._pool._free_ids) <= 0:
            for i in range(self.n_layers):
                kvc = self.kv_caches[i]
                evt = self.prefill_backup_events[i]
                # 表示可以进行驱逐的 page id
                ev_gpi = self.prefill_evicted_pages[i]
                # 如果没有可以驱逐的 page id 就直接 continue
                if ev_gpi is None:
                    continue
                assert evt is not None
                self.default_stream.wait_event(evt) # 同步等待 event
                [
                    kvc.pool.free_page(pid)
                    for pid in ev_gpi.reshape(-1).tolist()
                    if pid >= 0
                ]
                self.prefill_backup_events[i] = None
                self.prefill_evicted_pages[i] = None
                break
        return self._pool.alloc_page()

    def prefill_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.prefill_handler.forward(q, kvc.buffer, page_ids)

    def decode_sdpa(self, layer_idx: int, q: Tensor, page_ids: Tensor = None):
        kvc = self.kv_caches[layer_idx]
        if page_ids is None:
            page_ids = kvc.c2p
        return self.decode_handler_tab[kvc.budget].forward(q, kvc.buffer, page_ids)
