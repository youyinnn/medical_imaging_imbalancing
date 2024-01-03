from timm import utils
from timm.layers import set_fast_norm
import torch
import torch.nn as nn
from contextlib import suppress
from timm.utils import ApexScaler, NativeScaler
from functools import partial

from timm.models import create_model

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    print("No apex")
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    print("No native amp")
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    print("No wandb")
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
    print("No functorch")
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')
print("Has torch compile:", has_compile)


class TimmModel(nn.Module):

    def __init__(self,
                 in_chans: int,
                 device=torch.device('cpu'),
                 amp: bool = False, amp_impl: str = 'native', amp_dtype: str = 'float16',
                 fuser: str = '', fast_norm: bool = False,
                 model_key: str = 'resnet50', pretrained: bool = False,
                 pretrained_path: str = None,
                 num_classes: int = None, drop_rate: float = 0.0, drop_path: float = None,
                 drop_block: float = None, gp: str = None,
                 bn_momentum: float = None, bn_eps: float = None, torchscript: bool = False,
                 initial_checkpoint: str = '', factory_kwargs: dict = {},
                 model_kwargs: dict = {}, torchcompile: str = None, use_amp: str = None,
                 sync_bn: bool = False, head_init_scale: float = None, head_init_bias: float = None,
                 grad_checkpointing: bool = False, split_bn: bool = False,
                 model_ema: bool = False, model_ema_decay: float = 0.9988, model_ema_force_cpu: bool = False) -> None:

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # resolve AMP arguments based on PyTorch / Apex availability
        use_amp = None
        amp_dtype = torch.float16
        if amp:
            if amp_impl == 'apex':
                assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
                use_amp = 'apex'
                assert amp_dtype == 'float16'
            else:
                assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
                use_amp = 'native'
                assert amp_dtype in ('float16', 'bfloat16')
            if amp_dtype == 'bfloat16':
                amp_dtype = torch.bfloat16

        if fuser:
            utils.set_jit_fuser(fuser)
        if fast_norm:
            set_fast_norm()

        if pretrained_path is not None:
            factory_kwargs['pretrained_cfg_overlay'] = dict(
                file=pretrained_path,
                num_classes=-1,  # force head adaptation
            )

        model = create_model(
            model_key,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path,
            drop_block_rate=drop_block,
            global_pool=gp,
            bn_momentum=bn_momentum,
            bn_eps=bn_eps,
            scriptable=torchscript,
            checkpoint_path=initial_checkpoint,
            **factory_kwargs,
            **model_kwargs,
        )

        if head_init_scale is not None:
            with torch.no_grad():
                model.get_classifier().weight.mul_(head_init_scale)
                model.get_classifier().bias.mul_(head_init_scale)
        if head_init_bias is not None:
            nn.init.constant_(model.get_classifier().bias, head_init_bias)

        if num_classes is None:
            assert hasattr(
                model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            # FIXME handle model default vs config num_classes more elegantly
            num_classes = model.num_classes

        if grad_checkpointing:
            model.set_grad_checkpointing(enable=True)

        if torchscript:
            assert not torchcompile
            assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            model = torch.jit.script(model)

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        amp_autocast = suppress  # do nothing
        loss_scaler = None
        if use_amp == 'apex':
            assert device.type == 'cuda'
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            loss_scaler = ApexScaler()

        elif use_amp == 'native':
            try:
                amp_autocast = partial(
                    torch.autocast, device_type=device.type, dtype=amp_dtype)
            except (AttributeError, TypeError):
                # fallback to CUDA only AMP for PyTorch < 1.10
                assert device.type == 'cuda'
                amp_autocast = torch.cuda.amp.autocast
            if device.type == 'cuda' and amp_dtype == torch.float16:
                # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
                loss_scaler = NativeScaler()

        # setup exponential moving average of model weights, SWA could be used here too
        model_ema = None
        if model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
            model_ema = utils.ModelEmaV2(
                model, decay=model_ema_decay, device='cpu' if model_ema_force_cpu else None)

        self.model = model

    def forward(self, x):
        return self.model(x)
