import types

def _linear_forward(self, features, graph, *args, **kwargs):
    return self.linear(features)

def install_linear_perception(model) -> None:
    if model.Dynamic_train:
        raise ValueError("the released model requires Dynamic_train=False")
    for side in (model.x1_percep_model, model.x2_percep_model):
        for parameter in side.parameters():
            parameter.requires_grad_(False)
        for parameter in side.linear.parameters():
            parameter.requires_grad_(True)
        side.forward = types.MethodType(_linear_forward, side)

def arm_linear_perception_after_reset(model):
    original_reset = model.reset_parameters

    def reset(self):
        original_reset()
        install_linear_perception(self)

    model.reset_parameters = types.MethodType(reset, model)
    return model
