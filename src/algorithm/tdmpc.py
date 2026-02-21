import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import algorithm.helper as h
import time
from torch.func import functional_call, vmap, jacrev

class TOLD(nn.Module):
    """
    Told model with new architecture components:
    - LayerNorm + Mish (via h.mlp / h.q)
    - SimNorm on latent representation
    - 5 Q-functions
    - Stochastic policy (MaxEnt RL)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg) 
        trainable_params = sum(p.numel() for p in self._encoder.parameters() if p.requires_grad)
        print(f"Trainable parameters in ENCODER : {trainable_params}")

        # Transition dynamic function with SimNorm
        self._dynamics = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, cfg.latent_dim, last_act=h.SimNorm(cfg))
        self._reward = h.mlp(cfg.latent_dim+cfg.action_dim, cfg.mlp_dim, 1)
        
        # Stochastic policy
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, 2*cfg.action_dim)
        
        # 5 Q-functions with Dropout
        self._Qs = nn.ModuleList([h.q(cfg) for _ in range(5)])
        
        self.apply(h.orthogonal_init)
        # Initialize the last layer to zero
        for m in [self._reward]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        for m in self._Qs:
            h.set_requires_grad(m, enable)

    def h(self, obs):
        return self._encoder(obs)

    def next(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def pi(self, z, eval_mode=False):
        """Gaussian policy with reparametrization (MaxEnt RL)"""
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -10, 2) # For stability
        std = log_std.exp()
        
        if eval_mode:
            return torch.tanh(mu)
        
        eps = torch.randn_like(mu)
        action = torch.tanh(mu + eps * std)
        
        # Compute entropy 
        log_prob = h.gaussian_logprob(eps, log_std) - torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

    def Q(self, z, a):
        """Return 5 predictions from the set"""
        x = torch.cat([z, a], dim=-1)
        return torch.stack([m(x) for m in self._Qs])
    
    
    def compute_zgr_from_z(self, z: torch.Tensor, eps: float = 1e-6) -> float:

        if z.grad is None:
            raise RuntimeError("z.grad is None. Did you forget retain_grad() or backward()?")

        with torch.no_grad():
            zero_grad_mask = (z.grad.abs() < eps)
            return zero_grad_mask.float().mean().item()

    def compute_fzar_from_obs(self, obs: torch.Tensor, eps: float = 1e-3) -> float:
        """
        Compute Feature Zero Activation Ratio from backbone output.
        
        obs: batch of real observations
        eps: threshold for considering a feature "effectively zero"
        """
        self._encoder.eval()  # stable evaluation mode
        with torch.no_grad():
            z = self.h(obs)
            zero_mask = (z.abs() < eps)
            return zero_mask.float().mean().item()

    def compute_srank(self, obs: torch.Tensor, delta: float = 0.01) -> int:
        """
        Compute the effective feature rank (srank) of the encoder output.
        obs: batch of observations
        delta: threshold for cumulative singular values (default 0.01)
        """
        self._encoder.eval()
        with torch.no_grad():
            z = self.h(obs)              # (batch_size, latent_dim)
            z_cpu = z.detach().cpu()
            z_centered = z_cpu - z_cpu.mean(dim=0, keepdim=True)

            sigma = torch.linalg.svdvals(z_centered)
            cum_sum = torch.cumsum(sigma, dim=0)
            total_sum = sigma.sum()
            k = torch.searchsorted(cum_sum / total_sum, 1 - delta)
            return int(k.item() + 1)
        
    def compute_gradient_covariance(self, loss_per_sample, device):
        """Compute gradient covariance matrix"""
        encoder_params = [p for p in self._encoder.parameters() if p.requires_grad]
        N = loss_per_sample.shape[0] // 6  # Sous-échantillonnage
        P = sum(p.numel() for p in encoder_params)
        
        G = torch.zeros(N, P, device=device)
        
        step = loss_per_sample.shape[0] // N  # = 6
        for i in range(N):
            idx = i * step  # Échantillonner uniformément
            grads = torch.autograd.grad(
                outputs=loss_per_sample[idx],
                inputs=encoder_params,
                retain_graph=True,
                create_graph=False  
            )
            grads_flat = torch.cat([g.view(-1) for g in grads])
            G[i] = grads_flat / (grads_flat.norm() + 1e-10)
        
        # Matrice de Gram (similarité cosinus)
        Gram = G @ G.T
        
        return Gram
    
    def get_k_center_indices(self, z, k=36):
        n = z.shape[0]  

        if n < k:
            k = n

        selected_indices = [torch.randint(0, n, (1,)).item()]
        
        min_distances = torch.cdist(z[selected_indices], z, p=2).min(dim=0).values

        ## Greedy selection of z elements far from each other
        for _ in range(1, k):
            new_idx = torch.argmax(min_distances).item()
            selected_indices.append(new_idx)
            new_dist = torch.norm(z - z[new_idx], dim=1)
            min_distances = torch.min(min_distances, new_dist)
            
        return selected_indices
    
    def compute_eNTK(self, obs):
        start_time= time.time()
        
        z = self.h(obs)
        indices = self.get_k_center_indices(z.detach(), k=36)
        obs_reduced	 = obs[indices]
        params = {k: v.detach() for k, v in self._encoder.named_parameters()}
        def fnet_single(params, x):
            return functional_call(self._encoder, params, (x.unsqueeze(0),)).squeeze(0)

        jacs = vmap(jacrev(fnet_single), in_dims=(None, 0))(params, obs_reduced)

        flat_jacs = torch.cat([j.flatten(2) for j in jacs.values()], dim=2)
        
        j_full = flat_jacs.view(36 * 50, -1) 
        eNTK_full = j_full @ j_full.T
        
        # j_pseudo = flat_jacs.sum(dim=1) # (36, P)
        # pNTK = j_pseudo @ j_pseudo.T
        
        print("Time to compute eNTK in s : ",(time.time()-start_time))
        return eNTK_full 

class TDMPC():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(cfg).to(self.device)
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.aug = h.RandomShiftsAug(cfg)
        
        # Coefficient d'entropie (Alpha) pour MaxEnt RL
        self.alpha = cfg.entropy_coef if hasattr(cfg, 'entropy_coef') else 0.05
        
        self.model.eval()
        self.model_target.eval()
        self.initial_model = deepcopy(self.model)
        self.initial_model.eval()
        h.set_requires_grad(self.initial_model, False)

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        
        # TD-MPC2 : Min de 2 Q-functions tirées aléatoirement
        qs = self.model.Q(z, self.model.pi(z, eval_mode=True))
        q_min = torch.min(qs[torch.randperm(5)[:2]], dim=0).values
        G += discount * q_min
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, device=self.device).uniform_(-1, 1)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        
        # Initialisation CEM (sans momentum entre itérations comme TD-MPC2)
        z = self.model.h(obs).repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(horizon, self.cfg.action_dim, device=self.device)
        std = 2*torch.ones(horizon, self.cfg.action_dim, device=self.device)
        
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device), -1, 1)

            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            
            # Mise à jour directe (Suppression du momentum du planificateur)
            mean, std = _mean, _std.clamp_(self.std, 2)

        self._prev_mean = mean
        a = mean[0]
        if not eval_mode:
            a += std[0] * torch.randn(self.cfg.action_dim, device=self.device)
        return a.clamp(-1, 1)

    def update_pi(self, zs):
            self.pi_optim.zero_grad(set_to_none=True)
            self.model.track_q_grad(False) # On ne veut pas entraîner les Q ici

            pi_loss = 0
            for t in range(len(zs) - 1): # On parcourt la séquence
                z = zs[t]
                action, log_prob = self.model.pi(z)
                qs = self.model.Q(z, action)
                q_avg = qs.mean(dim=0) # TD-MPC2 utilise la moyenne de l'ensemble pour pi
                
                rho = (self.cfg.rho ** t)
                # MaxEnt RL : Maximiser (Valeur + Entropie)
                pi_loss += (self.alpha * log_prob - q_avg).mean() * rho

            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
            self.pi_optim.step()
            self.model.track_q_grad(True)
            return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        next_z = self.model.h(next_obs)
        next_action, _ = self.model.pi(next_z)
        
        # Cible TD : Min de 2 Q-functions aléatoires du réseau TARGET
        target_qs = self.model_target.Q(next_z, next_action)
        target_q_min = torch.min(target_qs[torch.randperm(5)[:2]], dim=0).values
        return reward + self.cfg.discount * target_q_min

    def update(self, replay_buffer, step,compute_metrics=False,compute_K=False):
            """
            Version adaptée de TD-MPC2 :
            1. Rollout latent complet d'abord.
            2. Calcul des pertes normalisées par l'horizon.
            3. Échantillonnage uniforme.
            """
            # Échantillonnage (Uniforme par défaut dans TD-MPC2)
            obs, next_obses, action, reward, idxs, weights = replay_buffer.sample()
            self.optim.zero_grad(set_to_none=True)
            self.model.train()
            
            #Metrics :
            fzar = 0
            srank = 0
            zgr = 0

            # 1. Préparation des cibles TD (avec le réseau Target)
            with torch.no_grad():
                next_z = self.model.h(self.aug(next_obses[0])) # Premier état suivant réel
                # Note: TD-MPC2 calcule souvent toutes les cibles d'un coup
                # Ici on garde la logique de boucle pour la compatibilité
                td_targets = torch.stack([
                    self._td_target(self.aug(next_obses[t]), reward[t])
                    for t in range(self.cfg.horizon)
                ])

            # 2. Rollout Latent (Imaginer les états futurs)
            # On stocke tous les états z pour la politique plus tard
            zs = torch.empty(self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
            # Representation
            z_backbone = self.model.h(self.aug(obs))
            z_backbone.retain_grad()

            if compute_metrics:
                srank = self.model.compute_srank(self.aug(obs))
                fzar = self.model.compute_fzar_from_obs(self.aug(obs))
                
            if compute_K:
                eNTK = self.model.compute_eNTK(self.aug(obs))
    
            # Copy pour rollout
            z = z_backbone
            
            zs[0] = z
            
            consistency_loss = 0
            reward_loss = 0
            value_loss = 0

            for t in range(self.cfg.horizon):
                # Prédictions
                qs = self.model.Q(z, action[t]) # Ensemble de 5 Q
                z, reward_pred = self.model.next(z, action[t])
                zs[t+1] = z
                
                # Calcul des pertes avec pondération rho (temporelle)
                rho = (self.cfg.rho ** t)
                
                # Consistance : l'état imaginé doit ressembler à l'état réel encodé
                with torch.no_grad():
                    z_real = self.model_target.h(self.aug(next_obses[t]))
                consistency_loss += rho * torch.mean(h.mse(z, z_real), dim=1, keepdim=True)
                reward_loss += rho * h.mse(reward_pred, reward[t])
                
                # Valeur : chaque Q de l'ensemble doit prédire la cible TD
                for q_pred in qs:
                    value_loss += rho * h.mse(q_pred, td_targets[t])

            # 3. Normalisation (La clé de TD-MPC2)
            consistency_loss /= self.cfg.horizon
            reward_loss /= self.cfg.horizon
            value_loss /= (self.cfg.horizon * 5) # Divisé par l'horizon ET num_q

            total_loss = self.cfg.consistency_coef * consistency_loss + \
                self.cfg.reward_coef * reward_loss + \
                self.cfg.value_coef * value_loss
                
            if compute_K:
                grad_cov = self.model.compute_gradient_covariance(total_loss.squeeze(1), self.device)
            total_loss = total_loss.squeeze(1).mean()

            total_loss.backward()
            
            # 4. Optimisation du Modèle
            if compute_metrics :
                zgr = self.model.compute_zgr_from_z(z_backbone)

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optim.step()

            # 5. Mise à jour de la Politique (avec toute la séquence zs)
            pi_loss = self.update_pi(zs.detach())

            # 6. Mise à jour lente du réseau Target (EMA)
            if step % self.cfg.update_freq == 0:
                h.ema(self.model, self.model_target, self.cfg.tau)

            self.model.eval()
            return {
                'total_loss': total_loss.item(),
                'pi_loss': pi_loss,
                'grad_norm': grad_norm.item(),
                'weight_distance': self.calculate_weight_distance(), 
    			'weight_magnitude': self.calculate_weight_magnitude(),
       			'zgr' : zgr,
          		'fzar':fzar,
            	'srank':srank,
                'K':grad_cov if compute_K else 0,
                'eNTK' : eNTK if compute_K else 0

            }
            
            
    @torch.no_grad()
    def calculate_weight_magnitude(self):
        total_magnitude = 0
        for param in self.model.parameters():
            if param.requires_grad:
                layer_mean_sq = param.pow(2).mean()
                total_magnitude += layer_mean_sq
                
        return total_magnitude.item()

    @torch.no_grad()
    def calculate_weight_distance(self):
        """
        Calcule la Distance des Poids : 
        Somme des moyennes des distances quadratiques (w - w0)^2 par couche.
        """
        total_distance = 0
        for p, p0 in zip(self.model.parameters(), self.initial_model.parameters()):
            if p.requires_grad:
                layer_dist = (p - p0).pow(2).mean()
                total_distance += layer_dist
                
        return total_distance.item()
