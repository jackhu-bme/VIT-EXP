import torch
from torch.autograd import Function
import torch.distributed as distributed

from einops import rearrange

# distributed helpers

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, accelerator):
        x_gather = accelerator.gather(x)
        ctx.num_processes = accelerator.num_processes
        ctx.process_index = accelerator.process_index
        return x_gather

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.chunk(ctx.num_processes, dim = 0)[ctx.process_index]
        return grad_input




def all_gather_variable_dim(t, dim = 0, sizes = None):
    device, rank, world_size = t.device, distributed.get_rank(), distributed.get_world_size()

    # if not exists(sizes):
    #     size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
    #     sizes = [torch.empty_like(size, device = device, dtype = torch.long) for i in range(world_size)]
    #     distributed.all_gather(sizes, size)
    #     sizes = torch.stack(sizes)

    # max_size = sizes.amax().item()
    # padded_t = pad_dim_to(t, max_size, dim = dim)

    gathered_tensors = [torch.empty(t.shape, device = device, dtype = t.dtype) for i in range(world_size)]
    distributed.all_gather(gathered_tensors, t)

    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    # seq = torch.arange(max_size, device = device)

    # mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    # mask = rearrange(mask, 'i j -> (i j)')
    # seq = torch.arange(mask.shape[-1], device = device)
    # indices = seq[mask]

    # gathered_tensor = gathered_tensor.index_select(dim, indices)

    return gathered_tensor, sizes

# class AllGather(Function):
#     @staticmethod
#     def forward(ctx, x, dim, sizes):
#         assert distributed.is_initialized() and distributed.get_world_size() > 1
#         x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
#         ctx.batch_sizes = batch_sizes.tolist()
#         ctx.dim = dim
#         return x, batch_sizes

#     @staticmethod
#     def backward(ctx, grads, _):
#         batch_sizes, rank = ctx.batch_sizes, distributed.get_rank()
#         grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
#         return grads_by_rank[rank], None, None

# all_gather = AllGather.apply
