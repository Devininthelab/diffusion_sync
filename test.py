# Get scheduler from the model
        if hasattr(self.model, 'scheduler'):
            scheduler = self.model.scheduler
        else:
            # For deepfloyd model
            scheduler = self.stage_1.scheduler
            
        # Get alpha values for current and previous timesteps
        alpha_t = scheduler.alphas_cumprod[timestep]
        
        # Get previous timestep
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        prev_timestep = torch.where(prev_timestep < 0, torch.zeros_like(prev_timestep), prev_timestep)
        
        alpha_t_prev = scheduler.alphas_cumprod[prev_timestep]
        
        # Compute DDIM reverse step: ψ^(t)(x^(t), x^(0))
        # ψ^(t)(x^(t), x^(0)) = √α_{t-1} * x^(0) + √((1-α_{t-1})/(1-α_t)) * (x^(t) - √α_t * x^(0))
        
        alpha_t = alpha_t.view(-1, 1, 1, 1)  # Reshape for broadcasting
        alpha_t_prev = alpha_t_prev.view(-1, 1, 1, 1)
        
        # Compute the direction pointing from x_t to x_0
        pred_dir = xts - torch.sqrt(alpha_t) * pred_x0s
        
        # Compute previous sample
        pred_prev_sample = (
            torch.sqrt(alpha_t_prev) * pred_x0s + 
            torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * pred_dir
        )
        
        return pred_prev_sample