from typing import Dict
import torch
import logging

import torch.nn.functional as F
from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere

logger = logging.getLogger()

class MultiTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',        
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict}
        
        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = [] 
        
        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            empty = False
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                
                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )
                
                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"]
                )
                
            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )
                
            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )
            
            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )
                
            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")
        
        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")
                
        logger.info(f"Initialized gaussians from pcd")
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )

        outputs["custom_tensor"] = self.custom_tensor
        if list(outputs["rgb"].shape) == [450, 800, 3] : 
        
            outputs["rgb_light_effect"], outputs["rgb_light_mask"] = self.simulate_light_sources(outputs["emitted_light"])
            #outputs["rgb"] = outputs["rgb"] + outputs["rgb_light_effect"] + outputs["custom_tensor"]

            outputs["rgb"] = self.raw_to_srgb(outputs["rgb"]) + outputs["custom_tensor"] + torch.abs(outputs["emitted_light"]) + outputs["rgb_light_effect"]
            outputs["rgb_light_mask"] = outputs["emitted_light"]
            print("MAX MAX",torch.abs(outputs["emitted_light"]).max())

            #outputs["rgb"] = torch.clamp(outputs["rgb"] , min=0, max=1)
        
        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity, emtted_light = render_fn(gaussian_mask)
                    outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity, emtted_light = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
        
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict

    def simulate_light_sources_o(self, image):
        """
        Simulate the effect of light sources in an image.

        Args:
            image (torch.Tensor): Input image tensor of shape [W, H, 3].

        Returns:
            torch.Tensor: Image with simulated light effects.
        """
        # Step 1: Identify light sources (all channels > 0.9)
        light_mask = (image > 0.9).all(dim=-1).float()  # Shape: [W, H]

        # Step 2: Apply 1D convolution to spread light effect
        x = torch.arange(self.kernel_size) - self.kernel_size // 2
        gaussian = torch.exp(-(x.to(self.device)**2) / (2 * self.sigma_kernel**2))
        gaussian = gaussian / gaussian.sum()  # torch.Size([150])        
        
        kernel = gaussian.view(1, 1, -1)  # Shape: [out_channels, in_channels, kernel_size]

        print("self.sigma_kernel",self.sigma_kernel)
        print(f"self.sigma_kernel: {self.sigma_kernel.item():.16f}")

        # Calculate padding to ensure output size matches input size
        padding = (self.kernel_size ) // 2

        # Convolve the light mask column-by-column (vertical spread)
        light_effect = F.conv1d(
            light_mask.t().unsqueeze(1),  # Transpose to [H, 1, W]
            kernel,
            padding=padding,
        ).squeeze(1).t()  # Transpose back to [W, H]

        # Step 3: Add the light effect back to the original image
        light_effect = light_effect.unsqueeze(-1).repeat(1, 1, 3)  # Broadcasting to [W, H, 3]

        print("light_mask",torch.sum(light_mask) )

        return light_effect, light_mask

    def simulate_light_sources(self, image):
        """
        Simulate the effect of light sources in an image by applying 1D convolution
        to each channel of the image.

        Args:
            image (torch.Tensor): Input image tensor of shape [W, H, 3].

        Returns:
            torch.Tensor: Image with simulated light effects.
        """
        # Step 1: Prepare the Gaussian kernel
        x = torch.arange(self.kernel_size) - self.kernel_size // 2
        gaussian = torch.exp(-(x.to(self.device)**2) / (2 * self.sigma_kernel**2))
        gaussian = gaussian / gaussian.sum()  # Normalize the kernel
        kernel = gaussian.view(1, 1, -1)  # Shape: [out_channels, in_channels, kernel_size]

        # Calculate padding to ensure the output size matches the input size
        padding = (self.kernel_size) // 2

        # Step 2: Apply convolution to each channel independently
        convolved_channels = []
        for c in range(image.shape[-1]):  # Iterate over channels (R, G, B)
            channel = image[..., c]  # Shape: [W, H]

            # Reshape channel for convolution
            channel = channel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, W, H]

            # Apply vertical convolution
            convolved_v = F.conv2d(
                channel,
                kernel.unsqueeze(0).transpose(-1, -2),  # Shape: [1, 1, kernel_size, 1]
                padding=(padding, 0)
            ).squeeze(0).squeeze(0)  # Shape: [W, H]

            # Combine horizontal and vertical convolutions
            convolved_channel = convolved_v 

            convolved_channels.append(convolved_channel)

        # Combine the convolved channels back into an image
        convolved_image = torch.stack(convolved_channels, dim=-1)  # Shape: [W, H, 3]

        return convolved_image , (image > 0.9).all(dim=-1).float()   # Ensure pixel values are in a valid range
    

    def raw_to_srgb(self, x) : 
        
        # 1. White Balance
        wb_corrected = x * self.wb_gain + 1e-7
        
        # 2. Color Matrix Transformation
        color_transformed = torch.matmul(wb_corrected, self.color_matrix.t())
        
        # 3. Clip to prevent out-of-range values
        #color_transformed = torch.clamp(color_transformed, 0.0, 1.0)
        #color_transformed = torch.abs(color_transformed)
        color_transformed = torch.clamp(x, 0.0001, 3.0)
        
        # 4. Gamma Correction (sRGB)
        srgb = torch.pow(color_transformed, 1.0 / 2.2)
        
        return x         
    