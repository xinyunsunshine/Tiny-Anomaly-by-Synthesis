
import torch 

quantized_patterns = [
    # 'weight'
    'model.downs.0.1.block1.proj.weight', 'model.downs.0.1.block2.proj.weight', 'model.downs.0.2.to_qkv.weight', 'model.downs.1.0.block1.proj.weight', 'model.downs.1.0.block2.proj.weight', 'model.downs.1.1.block1.proj.weight', 'model.downs.1.1.block2.proj.weight', 'model.downs.1.2.to_qkv.weight', 'model.downs.2.0.block1.proj.weight', 'model.downs.2.0.block2.proj.weight', 'model.downs.2.1.block1.proj.weight', 'model.downs.2.1.block2.proj.weight', 'model.downs.2.2.to_qkv.weight', 'model.downs.3.0.block1.proj.weight', 'model.downs.3.0.block2.proj.weight', 'model.downs.3.1.block1.proj.weight', 'model.downs.3.1.block2.proj.weight', 'model.downs.3.2.to_qkv.weight', 'model.downs.3.2.to_out.weight', 'model.ups.0.0.res_conv.weight', 'model.ups.0.1.block1.proj.weight', 'model.ups.0.1.block2.proj.weight', 'model.ups.0.1.res_conv.weight', 'model.ups.0.2.to_qkv.weight', 'model.ups.0.2.to_out.weight', 'model.ups.1.0.block1.proj.weight', 'model.ups.1.0.block2.proj.weight', 'model.ups.1.0.res_conv.weight', 'model.ups.1.1.block1.proj.weight', 'model.ups.1.1.block2.proj.weight', 'model.ups.1.1.res_conv.weight', 'model.ups.1.2.to_qkv.weight', 'model.ups.2.0.block1.proj.weight', 'model.ups.2.0.block2.proj.weight', 'model.ups.2.0.res_conv.weight', 'model.ups.2.1.block1.proj.weight', 'model.ups.2.1.block2.proj.weight', 'model.ups.2.1.res_conv.weight', 'model.ups.2.2.to_qkv.weight', 'model.ups.3.0.block1.proj.weight', 'model.ups.3.0.block2.proj.weight', 'model.ups.3.0.res_conv.weight', 'model.ups.3.1.block1.proj.weight', 'model.ups.3.1.block2.proj.weight', 'model.ups.3.1.res_conv.weight', 'model.ups.3.2.to_qkv.weight', 'model.mid_block1.block1.proj.weight', 'model.mid_block1.block2.proj.weight', 'model.mid_block2.block1.proj.weight', 'model.mid_block2.block2.proj.weight', 'model.final_res_block.res_conv.weight'
]


skipped_patterns = [
    'embed',
    'norm', 'mem_kv', 
    ### constants
    'betas',
    'alphas_cumprod',
    'alphas_cumprod_prev',
    'sqrt_alphas_cumprod',
    'sqrt_one_minus_alphas_cumprod',
    'log_one_minus_alphas_cumprod',
    'sqrt_recip_alphas_cumprod',
    'sqrt_recipm1_alphas_cumprod',
    'posterior_variance',
    'posterior_log_variance_clipped',
    'posterior_mean_coef1',
    'posterior_mean_coef2',
    'loss_weight',
]

def quantize_to_float16(tensor):
    float16_tensor = tensor.to(torch.float16)
    return float16_tensor

def quantize_to_int4(tensor):
    # Define the int4 range
    int4_min = -8
    int4_max = 7

    # Clip the tensor values to the int4 range
    clipped_tensor = torch.clamp(tensor, int4_min, int4_max)

    # Convert the clipped tensor to int4
    int4_tensor = clipped_tensor.to(torch.int8)

    return int4_tensor

def is_quantized_module(key):
    """Determine if a module should be quantized based on config patterns"""
    return any(pattern in key for pattern in quantized_patterns)
    # return not any(pattern in key for pattern in skipped_patterns)

def quantize_model(model, n_bits):
    """
    Quantize the model to n_bits
    n_bits: number of bits to quantize the model to. Currently only support 8 and 16
    """
    state_dict = model.state_dict()
    quantized_state_dict = {}
    quantized_cnt = 0
    unquantized_cnt = 0

    for key, value in state_dict.items():
        print(key, value.dtype)
        if value.dtype == torch.float32 and is_quantized_module(key):
            if n_bits == 8:
              quantized_state_dict[key] = quantize_to_int4(value)
              quantized_cnt += 1
            elif n_bits == 16:
              quantized_state_dict[key] = quantize_to_float16(value)
              quantized_cnt += 1
        else:
            quantized_state_dict[key] = value
            unquantized_cnt += 1
    print(f"Quantized cnt: {quantized_cnt}")
    print(f"Unquantized cnt: {unquantized_cnt}")    
    model.load_state_dict(quantized_state_dict)
    return quantized_state_dict

