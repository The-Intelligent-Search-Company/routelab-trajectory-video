"""Strict all-component shape checks before candidate installation."""
from safetensors import safe_open
from safetensors.torch import load_file

def validate_component_shapes(modules, paths):
    if set(modules) != set(paths):
        raise ValueError('Candidate components differ from the live model')
    # Validate ALL components before changing any model tensor.
    for name, module in modules.items():
        expected = module.state_dict()
        with safe_open(str(paths[name]), framework='pt', device='cpu') as source:
            if set(source.keys()) != set(expected):
                raise ValueError('Candidate tensor inventory differs: ' + name)
            for key, value in expected.items():
                if tuple(source.get_slice(key).get_shape()) != tuple(value.shape):
                    raise ValueError('Candidate tensor shape differs: ' + name + '.' + key)



def install_components(modules, paths):
    validate_component_shapes(modules, paths)
    for name, module in modules.items():
        weights = load_file(str(paths[name]), device='cpu')
        module.load_state_dict(weights, strict=True)
        del weights
        module.requires_grad_(False).eval()
