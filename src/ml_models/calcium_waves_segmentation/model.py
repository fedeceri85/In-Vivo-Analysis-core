from monai.networks.nets import UNet


def build_model(config):
    model_config = config["model"]
    return UNet(
        spatial_dims=model_config["spatial_dims"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        channels=tuple(model_config["channels"]),
        strides=tuple(model_config["strides"]),
        num_res_units=model_config["num_res_units"],
    )
