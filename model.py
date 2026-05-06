import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_norm_adj_mat, ssl_loss, topk_sample, propgt_info


def linear_beta_schedule(T, b0, b1):
    return torch.linspace(b0, b1, T)


def cosine_beta_schedule(T, s=0.008):
    x = torch.linspace(0, T, T + 1)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    return torch.clip(1 - (ac[1:] / ac[:-1]), 1e-4, 0.9999)


def exp_beta_schedule(T, b_min=0.1, b_max=10.0):
    x = torch.linspace(1, 2 * T + 1, T)
    return 1 - torch.exp(-b_min / T - x * 0.5 * (b_max - b_min) / (T * T))


def sqrt_beta_schedule(T, max_beta=0.999):
    ab = lambda t: 1 - math.sqrt(t + 1e-4)
    return torch.tensor([min(1 - ab((i + 1) / T) / ab(i / T), max_beta) for i in range(T)]).float()


def extract(a, t, shape):
    out = a.to(t.device).gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))


class SinusoidalPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Denoiser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPE(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x, t, cond):
        return cond * self.gate(torch.cat((x, self.time_mlp(t)), dim=-1))


class LearnableRetriever(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, emb):
        proj = F.normalize(emb, dim=-1)
        sim = torch.matmul(proj, proj.T)
        sim = sim - 1e9 * torch.eye(sim.size(0), device=sim.device)
        val, idx = torch.topk(sim, self.k, dim=1)
        w = torch.softmax(val, dim=-1).unsqueeze(-1)
        return torch.sum(w * emb[idx], dim=1)


class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        T = config.timesteps
        sche = config.beta_sche
        if sche == 'linear':
            betas = linear_beta_schedule(T, config.beta_start, config.beta_end)
        elif sche == 'cosine':
            betas = cosine_beta_schedule(T)
        elif sche == 'exp':
            betas = exp_beta_schedule(T)
        else:
            betas = sqrt_beta_schedule(T)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)
        self.register_buffer('sqrt_ac', torch.sqrt(ac))
        self.register_buffer('sqrt_1m_ac', torch.sqrt(1.0 - ac))
        self.register_buffer('post_c1', betas * torch.sqrt(ac_prev) / (1.0 - ac))
        self.register_buffer('post_c2', (1.0 - ac_prev) * torch.sqrt(alphas) / (1.0 - ac))
        self.register_buffer('post_var', betas * (1.0 - ac_prev) / (1.0 - ac))
        self.T = T
        self.denoiser = Denoiser(config.embedding_dim)

    def q_sample(self, x0, t, noise):
        return extract(self.sqrt_ac, t, x0.shape) * x0 + extract(self.sqrt_1m_ac, t, x0.shape) * noise

    def p_losses(self, x0, t, cond):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.denoiser(xt, t, cond)
        return F.mse_loss(x0, pred), pred

    @torch.no_grad()
    def p_sample(self, xt, t, t_idx, cond):
        pred = self.denoiser(xt, t, cond)
        mean = extract(self.post_c1, t, xt.shape) * pred + extract(self.post_c2, t, xt.shape) * xt
        if t_idx == 0:
            return mean
        var = extract(self.post_var, t, xt.shape)
        return mean + torch.sqrt(var) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, x0, cond):
        noise = torch.randn_like(x0)
        t_full = torch.full((x0.shape[0],), self.T - 1, dtype=torch.long, device=x0.device)
        xt = self.q_sample(x0, t_full, noise)
        for n in reversed(range(self.T)):
            tv = torch.full((xt.shape[0],), n, dtype=torch.long, device=xt.device)
            xt = self.p_sample(xt, tv, n, cond)
        return xt


class Model(nn.Module):
    def __init__(self, config, dataset):
        super(Model, self).__init__()

        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_nodes = self.n_users + self.n_items
        self.i_v_feat = dataset.i_v_feat
        self.i_t_feat = dataset.i_t_feat
        self.embedding_dim = config.embedding_dim
        self.feat_embed_dim = config.embedding_dim
        self.dim_feat = self.feat_embed_dim
        self.reg_weight = config.reg_weight
        self.device = config.device
        self.cl_tmp = config.cl_tmp
        self.cl_loss_weight = config.cl_loss_weight
        self.causal_weight = config.causal_weight
        self.n_layers = config.n_layers
        self.num_user_co = config.num_user_co
        self.num_item_co = config.num_item_co
        self.user_aggr_mode = config.user_aggr_mode
        self.n_ii_layers = config.n_ii_layers
        self.n_uu_layers = config.n_uu_layers
        self.uu_co_weight = config.uu_co_weight
        self.ii_co_weight = config.ii_co_weight

        self.topK_users = dataset.topK_users
        self.topK_items = dataset.topK_items
        self.dict_user_co_occ_graph = dataset.dict_user_co_occ_graph
        self.dict_item_co_occ_graph = dataset.dict_item_co_occ_graph
        self.topK_users_counts = dataset.topK_users_counts
        self.topK_items_counts = dataset.topK_items_counts

        self.s_drop = config.s_drop
        self.m_drop = config.m_drop
        self.ly_norm = nn.LayerNorm(self.feat_embed_dim)

        self.self_i_attn1 = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)
        self.self_i_attn2 = nn.MultiheadAttention(1, 1, dropout=self.s_drop, batch_first=True)

        self.mutual_i_attn1 = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)
        self.mutual_i_attn2 = nn.MultiheadAttention(1, 1, dropout=self.m_drop, batch_first=True)

        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)

        self.prl = nn.PReLU().to(self.device)

        self.cal_bpr = torch.tensor([[1.0], [-1.0]]).to(self.device)

        self.norm_adj = get_norm_adj_mat(self, dataset.sparse_inter_matrix(form='coo')).to(self.device)
        self.user_co_graph = topk_sample(self.n_users, self.dict_user_co_occ_graph, self.num_user_co,
                                         self.topK_users, self.topK_users_counts, 'softmax',
                                         self.device)

        self.item_co_graph = topk_sample(self.n_items, self.dict_item_co_occ_graph, self.num_item_co,
                                         self.topK_items, self.topK_items_counts, 'softmax',
                                         self.device)

        self.i_mm_adj = dataset.i_mm_adj
        self.u_mm_adj = dataset.u_mm_adj

        self.stre_ii_graph = self.ii_co_weight * self.item_co_graph + (1.0 - self.ii_co_weight) * self.i_mm_adj
        self.stre_uu_graph = self.uu_co_weight * self.user_co_graph + (1.0 - self.uu_co_weight) * self.u_mm_adj

        if self.i_v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.i_v_feat, freeze=False).to(self.device)
            self.image_i_trs = nn.Linear(self.i_v_feat.shape[1], self.feat_embed_dim)
            self.user_v_prefer = torch.nn.Parameter(dataset.u_v_interest, requires_grad=True).to(self.device)
            self.image_u_trs = nn.Linear(self.i_v_feat.shape[1], self.feat_embed_dim)

        if self.i_t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.i_t_feat, freeze=False).to(self.device)
            self.text_i_trs = nn.Linear(self.i_t_feat.shape[1], self.feat_embed_dim)
            self.user_t_prefer = torch.nn.Parameter(dataset.u_t_interest, requires_grad=True).to(self.device)
            self.text_u_trs = nn.Linear(self.i_t_feat.shape[1], self.feat_embed_dim)

        self.w = config.w
        self.diff_weight = config.diff_weight
        self.timesteps = config.timesteps
        self.diff_b = Diffusion(config).to(self.device)
        self.diff_v = Diffusion(config).to(self.device)
        self.diff_t = Diffusion(config).to(self.device)
        self.retriever = LearnableRetriever(config.neighbors).to(self.device)
        self.sample_x = torch.zeros(self.n_items, self.embedding_dim, device=self.device)
        self.sample_v = torch.zeros(self.n_items, self.feat_embed_dim, device=self.device)
        self.sample_t = torch.zeros(self.n_items, self.feat_embed_dim, device=self.device)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.user_id_embedding.weight, std=0.1)
        nn.init.normal_(self.item_id_embedding.weight, std=0.1)
        nn.init.xavier_normal_(self.image_i_trs.weight)
        nn.init.xavier_normal_(self.text_i_trs.weight)
        nn.init.xavier_normal_(self.image_u_trs.weight)
        nn.init.xavier_normal_(self.text_u_trs.weight)

    def forward(self, pred_x, pred_v, pred_t):
        trs_item_v_feat = self.image_i_trs(self.image_embedding.weight)
        trs_item_t_feat = self.text_i_trs(self.text_embedding.weight)

        item_id = self.w * pred_x + (1 - self.w) * self.item_id_embedding.weight
        trs_item_v_feat = self.w * pred_v + (1 - self.w) * trs_item_v_feat
        trs_item_t_feat = self.w * pred_t + (1 - self.w) * trs_item_t_feat

        trs_user_v_prefer = self.image_u_trs(self.user_v_prefer)
        trs_user_t_prefer = self.text_u_trs(self.user_t_prefer)

        item_v_t = torch.cat((trs_item_v_feat, trs_item_t_feat), dim=-1)
        item_id_v_t = torch.cat((item_id, item_v_t), dim=-1)
        item_id_v_t = propgt_info(item_id_v_t, self.n_ii_layers, self.stre_ii_graph, last_layer=True)
        item_id_v_t = F.normalize(item_id_v_t)

        item_id_ii = item_id_v_t[:, :self.embedding_dim]
        gnn_i_v_feat = item_id_v_t[:, self.feat_embed_dim:-self.feat_embed_dim]
        gnn_i_t_feat = item_id_v_t[:, -self.feat_embed_dim:]

        user_v_t = torch.cat((trs_user_v_prefer, trs_user_t_prefer), dim=-1)
        user_id_v_t = torch.cat((self.user_id_embedding.weight, user_v_t), dim=-1)
        user_id_v_t = propgt_info(user_id_v_t, self.n_uu_layers, self.stre_uu_graph, last_layer=True)

        user_id_v_t = F.normalize(user_id_v_t)
        user_id_uu = user_id_v_t[:, :self.embedding_dim]
        gnn_u_v_prefer = user_id_v_t[:, self.embedding_dim:-self.feat_embed_dim]
        gnn_u_t_prefer = user_id_v_t[:, -self.feat_embed_dim:]

        item_v_feat, _ = self.self_i_attn1(gnn_i_v_feat.unsqueeze(2), gnn_i_v_feat.unsqueeze(2),
                                           gnn_i_v_feat.unsqueeze(2), need_weights=False)
        item_v_feat = self.ly_norm(gnn_i_v_feat + item_v_feat.squeeze())
        item_v_feat = self.prl(item_v_feat)

        item_t_feat, _ = self.self_i_attn2(gnn_i_t_feat.unsqueeze(2), gnn_i_t_feat.unsqueeze(2),
                                           gnn_i_t_feat.unsqueeze(2), need_weights=False)
        item_t_feat = self.ly_norm(gnn_i_t_feat + item_t_feat.squeeze())
        item_t_feat = self.prl(item_t_feat)

        i_t2v_feat, _ = self.mutual_i_attn1(item_t_feat.unsqueeze(2), item_v_feat.unsqueeze(2),
                                            item_v_feat.unsqueeze(2), need_weights=False)
        item_t2v_feat = self.ly_norm(item_v_feat + i_t2v_feat.squeeze())
        item_t2v_feat = self.prl(item_t2v_feat)
        self.item_t2v_feat = item_t2v_feat

        i_v2t_feat, _ = self.mutual_i_attn2(item_v_feat.unsqueeze(2), item_t_feat.unsqueeze(2),
                                            item_t_feat.unsqueeze(2), need_weights=False)
        item_v2t_feat = self.ly_norm(item_t_feat.squeeze() + i_v2t_feat.squeeze())
        item_v2t_feat = self.prl(item_v2t_feat)
        self.item_v2t_feat = item_v2t_feat

        user_v_prefer = self.prl(gnn_u_v_prefer)
        user_t_prefer = self.prl(gnn_u_t_prefer)

        item_v_t_feat = torch.cat((item_t2v_feat, item_v2t_feat), dim=-1)
        user_v_t_prefer = torch.cat((user_v_prefer, user_t_prefer), dim=-1)
        ego_feat_prefer = torch.cat((user_v_t_prefer, item_v_t_feat), dim=0)
        self.fin_feat_prefer = propgt_info(ego_feat_prefer, self.n_layers, self.norm_adj)

        ego_id_embed = torch.cat((user_id_uu, item_id_ii), dim=0)
        fin_id_embed = propgt_info(ego_id_embed, self.n_layers, self.norm_adj)

        fin_v = self.prl(self.fin_feat_prefer[:, :self.embedding_dim]) + fin_id_embed
        fin_t = self.prl(self.fin_feat_prefer[:, self.embedding_dim:]) + fin_id_embed

        representation = torch.cat((fin_v, fin_t), dim=-1)

        return representation

    def _run_diffusion(self):
        h_id = self.item_id_embedding.weight
        h_v = self.image_i_trs(self.image_embedding.weight)
        h_t = self.text_i_trs(self.text_embedding.weight)
        cond_b = self.retriever(h_id)
        cond_vt = h_id.detach()
        half = h_id.shape[0] // 2 + 1
        t = torch.randint(0, self.timesteps, (half,), device=self.device)
        t = torch.cat([t, self.timesteps - 1 - t], dim=0)[:h_id.shape[0]]
        loss_b, pred_b = self.diff_b.p_losses(h_id, t, cond_b)
        loss_v, pred_v = self.diff_v.p_losses(h_v, t, cond_vt)
        loss_t, pred_t = self.diff_t.p_losses(h_t, t, cond_vt)
        return loss_b + loss_v + loss_t, pred_b, pred_v, pred_t

    def loss(self, user_tensor, item_tensor):
        user_tensor_flatten = user_tensor.view(-1)
        item_tensor_flatten = item_tensor.view(-1)

        diff_loss, pred_x, pred_v, pred_t = self._run_diffusion()

        out = self.forward(pred_x, pred_v, pred_t)
        user_rep = out[user_tensor_flatten]
        item_rep = out[item_tensor_flatten]

        score = torch.sum(user_rep * item_rep, dim=1).view(-1, 2)
        bpr_score = torch.matmul(score, self.cal_bpr)
        bpr_loss = -torch.mean(nn.LogSigmoid()(bpr_score))

        i_mul_vt_cl_loss = ssl_loss(self.fin_feat_prefer[:, :self.feat_embed_dim],
                                    self.fin_feat_prefer[:, -self.feat_embed_dim:], item_tensor_flatten, self.cl_tmp)
        u_mul_vt_cl_loss = ssl_loss(self.fin_feat_prefer[:, :self.feat_embed_dim],
                                    self.fin_feat_prefer[:, -self.feat_embed_dim:], user_tensor_flatten, self.cl_tmp)
        mul_vt_cl_loss = self.cl_loss_weight * (i_mul_vt_cl_loss + u_mul_vt_cl_loss)

        causal_loss = self.causal_weight * ssl_loss(self.item_t2v_feat, self.item_v2t_feat,
                                                    item_tensor_flatten - self.n_users, self.cl_tmp)

        diff_loss_w = self.diff_weight * diff_loss
        total_loss = bpr_loss + mul_vt_cl_loss + diff_loss_w + causal_loss

        return total_loss, bpr_loss, mul_vt_cl_loss, diff_loss_w, causal_loss

    @torch.no_grad()
    def sample(self):
        h_id = self.item_id_embedding.weight
        h_v = self.image_i_trs(self.image_embedding.weight)
        h_t = self.text_i_trs(self.text_embedding.weight)
        cond_b = self.retriever(h_id)
        self.sample_x = self.diff_b.sample(h_id, cond_b)
        self.sample_v = self.diff_v.sample(h_v, h_id)
        self.sample_t = self.diff_t.sample(h_t, h_id)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        representation = self.forward(self.sample_x, self.sample_v, self.sample_t)
        u_reps, i_reps = torch.split(representation, [self.n_users, self.n_items], dim=0)
        score_mat_ui = torch.matmul(u_reps[user], i_reps.t())
        return score_mat_ui