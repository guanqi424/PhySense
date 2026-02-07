from typing import Optional

import einops
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from .torch_random_fields.utils.batch_lbp import batch_lbp
from .torch_random_fields.utils.loopy_belief_propagation import loopy_belief_propagation
from .torch_random_fields.utils.misc import batch_pad, chain
from .torch_random_fields.utils.naive_mean_field import naive_mean_field
from .torch_random_fields.constants import Inference, Learning


def batch_index_select_5d_rec(input, index):
    """
    input: (B, N, N, H, H)
    index: (B, I, 2)

    return: (B, I, H, H)
    """
    assert len(input.shape) == 5
    assert len(index.shape) == 3
    B, N, _, H, _ = input.shape
    B, I, _ = index.shape

    index_expanded = index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, H)
    
    selected = torch.empty(B, I, H, H, device=input.device)
    
    for b in range(B):
        for i in range(I):
            node1, node2 = index[b, i]
            selected[b, i] = input[b, node1, node2]

    return selected

def batch_index_select_5d(input, index):
    """
    input: (B, N, N, H, H)
    index: (B, I, 2)

    return: (B, I, H, H)
    """
    assert input.shape[1] == input.shape[2], "The second and third dimensions of input must be equal"
    assert index.shape[-1] == 2, "The last dimension of index must be 2"
    B, N, _, H, _ = input.shape
    B, I, _ = index.shape
    
    input_flat = input.view(B, N * N, H, H)
    
    flat_index = index[..., 0] * N + index[..., 1]
    
    selected = input_flat[torch.arange(B)[:, None], flat_index]

    return selected


class phySenseCRF(torch.nn.Module):
    def __init__(
            self,
            num_states,  # label space
            num_actions,
            num_inter_actions,
            beam_size=64,  # for beam search
            learning: str = Learning.PERCEPTRON,
            inference: str = Inference.BATCH_BELIEF_PROPAGATION,
    ) -> None:
        super().__init__()
        assert learning in (
            Learning.PIECEWISE,
            Learning.PERCEPTRON,
        )
        assert inference in (
            Inference.MEAN_FIELD,
            Inference.BELIEF_PROPAGATION,
            Inference.BATCH_BELIEF_PROPAGATION,
        )
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_inter_actions = num_inter_actions
        self.beam_size = beam_size
        self.learning = learning
        self.inference = inference
        self.weight_param_unary = nn.Parameter(torch.rand(self.num_states))
        self.weight_param_binary = nn.Parameter(torch.rand(self.num_states, self.num_states))

    def forward(
            self,
            *,
            unaries: torch.Tensor,  # marginal probability distribution over label space, [n_bsz, n_nodes, n_states]
            # marginal probability distribution over label space, [n_bsz, n_nodes, n_states, n_actions]
            behaviors: torch.Tensor,
            masks: torch.Tensor,  # node mask
            # show different node has different possible # behaviors/interaction
            behavior_masks: torch.Tensor,  # [n_states, n_actions]
            interaction_masks: torch.Tensor,  # [n_states, n_states, n_inter_actions]
            # interaction matrix, [n_bsz, n_nodes, n_nodes, n_states, n_states, n_inter_actions]
            interactions: torch.Tensor,
            binary_edges: torch.Tensor,  # node pairs which have interactions
            binary_masks: torch.Tensor,  # edges mask
            targets: torch.Tensor = None,  # ground truth
            device
    ):
        with torch.no_grad():
            batch_size, num_nodes, _ = unaries.shape
            masked_unaries = unaries * masks.unsqueeze(-1)
            behavior_mask_expanded = (behavior_masks.unsqueeze(0).unsqueeze(0).
                                      expand(batch_size, num_nodes, self.num_states, self.num_actions))
            masked_behaviors = behaviors * behavior_mask_expanded.float()

            action_counts = behavior_mask_expanded.sum(dim=-1).float()
            action_counts[action_counts == 0] = 1

            behaviors_unaries = masked_behaviors.sum(dim=-1) / action_counts

            behaviors_unaries = behaviors_unaries * masks.unsqueeze(-1)

        # assume that behaviors_unaries is arranged by default label space order
        weighted_unaries = masked_unaries + self.weight_param_unary * behaviors_unaries
        with torch.no_grad():
            self.beam_size = min(self.beam_size, masked_unaries.shape[2])
            """ build unary potentials """
            if targets is not None:
                # a nice property:
                #   the target word will be the first word in the beam,
                #   although it may have a low score
                _unaries = weighted_unaries.scatter(2, targets[:, :, None], float("inf"))
                beam_targets = _unaries.topk(self.beam_size, 2)[1]
                beam_unary_potentials = weighted_unaries.gather(2, beam_targets)
            else:
                beam_targets = weighted_unaries.topk(self.beam_size, 2)[1]
                beam_unary_potentials = weighted_unaries.gather(2, beam_targets)
            """ build binary potentials """
            # beam_targets: bsz x node x beam
            # edge:         bsz x edge
            # score:        bsz x edge x beam x rank
            # bin_phis:     bsz x edge x beam x beam
            #               the [bid][i][a][b] value denotes:
            temp_int_mask = interaction_masks.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            int_mask_expanded = temp_int_mask.expand(batch_size, num_nodes, num_nodes, self.num_states, self.num_states, self.num_inter_actions)
            int_mask_expanded = int_mask_expanded.to(device)

            masked_interactions = interactions * int_mask_expanded.float()

            interaction_counts = int_mask_expanded.sum(dim=-1).float()
            del int_mask_expanded
            interaction_counts[interaction_counts == 0] = 1

            masked_interactions = masked_interactions.sum(dim=-1) / interaction_counts
            # binary_edges: [n_bsz, n_edges, 2]
            selected_interactions = batch_index_select_5d(masked_interactions, binary_edges)
            # selected_interactions: [n_bsz, n_edges, n_states, n_states]

            # beam_targets: [n_bsz, n_nodes, beam_size]

            # prepare to index selected_interactions

            # select targets from beam_targets according to binary_edges
            beam_targets_node1 = beam_targets.gather(1, binary_edges[:, :, 0].unsqueeze(-1).expand(-1, -1, self.beam_size))
            beam_targets_node2 = beam_targets.gather(1, binary_edges[:, :, 1].unsqueeze(-1).expand(-1, -1, self.beam_size))

            # index from selected_interactions
            bin_phis_init = selected_interactions[
                torch.arange(batch_size)[:, None, None, None],
                torch.arange(binary_edges.shape[1])[None, :, None, None],
                beam_targets_node1[:, :, :, None],
                beam_targets_node2[:, :, None, :]
            ]
            # bin_phis_init: [n_bsz, n_edges, beam_size, beam_size]
        # involving learning
        bin_phis = (bin_phis_init *
                    self.weight_param_binary[beam_targets_node1[:, :, :, None], beam_targets_node2[:, :, None, :]])

        if targets is not None:
            if self.learning == Learning.PIECEWISE:
                # unary
                norm_unary = beam_unary_potentials.log_softmax(-1)
                gold_unary = norm_unary[:, :, 0]
                gold_unary = gold_unary.masked_fill(~masks, 0.)
                pll = gold_unary.sum(-1)
                # binary
                norm_bin_phis = chain(
                    bin_phis,  #
                    lambda __: einops.rearrange(__, "B E K1 K2 -> B E (K1 K2)"),
                    lambda __: __.log_softmax(-1),
                    lambda __: einops.rearrange(__, "B E (K1 K2) -> B E K1 K2", K1=self.beam_size, K2=self.beam_size),
                )
                gold_bin_phis = norm_bin_phis[:, :, 0, 0]
                gold_bin_phis = gold_bin_phis.masked_fill(~binary_masks, 0.)
                pll = pll + gold_bin_phis.sum(-1)
                # nll
                nll = -(pll / masks.sum(-1)).mean()
                return nll
            elif self.learning == Learning.PERCEPTRON:
                _, pred_idx, _ = self(
                    unaries=unaries,
                    masks=masks,
                    behaviors=behaviors,
                    behavior_masks=behavior_masks,
                    interaction_masks=interaction_masks,
                    interactions=interactions,
                    binary_edges=binary_edges,
                    binary_masks=binary_masks,
                    device=device
                )

                def score(index):
                    # Unary score calculation
                    unary = weighted_unaries.gather(2, index.unsqueeze(-1)).squeeze(-1) * masks

                    # Binary score calculation
                    # Gathering state indices for each edge
                    node_indices = binary_edges  # Assuming binary_edges has shape [n_bsz, n_edges, 2]
                    state_indices_1 = torch.gather(index, 1, node_indices[:, :, 0])  # For node1
                    state_indices_2 = torch.gather(index, 1, node_indices[:, :, 1])  # For node2

                    # Now state_indices_1 and state_indices_2 should have shape [n_bsz, n_edges]

                    # Expand state_indices to match the beam_size
                    state_indices_1_expanded = state_indices_1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.beam_size, self.beam_size)
                    state_indices_2_expanded = state_indices_2.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.beam_size, self.beam_size)

                    # Gather the binary scores using the expanded indices
                    # Assuming bin_phis has a shape of [n_bsz, n_edges, n_states, n_states]
                    selected_states_1 = bin_phis.gather(2, state_indices_1_expanded)
                    selected_states = selected_states_1.gather(3, state_indices_2_expanded)

                    # Apply binary mask
                    binary = selected_states.sum(dim=(-1, -2)) * binary_masks

                    # Final score calculation
                    ret = unary.sum(-1) + binary.sum(-1)
                    return ret

                delta = 1 + score(pred_idx) - score(targets)
                loss = (delta / masks.sum(-1)).mean()
                return loss
            else:
                raise NotImplementedError
        else:
            with torch.no_grad():
                if self.inference == Inference.BATCH_BELIEF_PROPAGATION:
                    _, pred_idx = batch_lbp(
                        bat_unary_potentials=beam_unary_potentials,
                        bat_unary_masks=masks,
                        bat_binary_potentials=bin_phis,
                        bat_binary_edges=binary_edges,
                        bat_binary_masks=binary_masks,
                        max_iter=10,
                        damping=0.5,
                    )
                    pred_idx[pred_idx == -1] = 0
                    pred_idx = chain(
                        pred_idx,
                        lambda __: beam_targets.gather(-1, __.unsqueeze(-1)).squeeze(-1),
                        lambda __: __.masked_fill_(~masks, 0),
                    )
                    return None, pred_idx
                elif self.inference in (Inference.MEAN_FIELD, Inference.BELIEF_PROPAGATION):

                    def infer_one_sentence(bid):
                        node_len = masks[bid].sum(-1)
                        bin_edge_len = binary_masks[bid].sum(-1)

                        kwargs = dict(
                            unary_potentials=beam_unary_potentials[bid][:node_len],
                            binary_potentials=bin_phis[bid][:bin_edge_len],
                            binary_edges=binary_edges[bid][:bin_edge_len],
                            max_iter=10,
                            track_best=True,
                        )
                        if self.inference == Inference.MEAN_FIELD:
                            ret, energy = naive_mean_field(**kwargs)
                        elif self.inference == Inference.BELIEF_PROPAGATION:
                            ret, energy = loopy_belief_propagation(**kwargs)
                        return ret, energy
                    pred_idx = []
                    pred_energies = []
                    for bid in range(batch_size):
                        my_ret, my_energy = infer_one_sentence(bid)
                        pred_idx.append(my_ret.tolist())
                        pred_energies.append(my_energy.item())
                    pred_idx = chain(
                        pred_idx,
                        lambda __: batch_pad(__, 0, pad_len=num_nodes),
                        lambda __: torch.LongTensor(__).to(beam_unary_potentials.device),
                        lambda __: beam_targets.gather(-1, __.unsqueeze(-1)).squeeze(-1),
                        lambda __: __.masked_fill_(~masks, 0),
                    )

                    return None, pred_idx, pred_energies

                elif self.inference == Inference.NONE:
                    pred_idx = beam_unary_potentials.argmax(-1)
                    return None, pred_idx
