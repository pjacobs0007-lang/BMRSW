import torch

class NormalMeanSimulator(torch.nn.Module):
    """
    Differentiable Normal Mean with Identity Covariance.
    Model: X ~ N(theta, sigma^2 * I_d)
    
    Parameters: 
    - If sigma_fixed is set: theta (Batch, d)
    - If sigma_fixed is None: theta = [mu_1, ..., mu_d, raw_sigma] (Batch, d+1)
      sigma = softplus(raw_sigma)
    """
    def __init__(self, d, sigma_fixed=1.0, parameterization='softplus'):
        super().__init__()
        self.d = d
        self.sigma_fixed = sigma_fixed # If None, learn sigma
        self.parameterization = parameterization
        
    def sample_noise(self, n, device):
        return torch.randn(n, self.d, device=device)
        
    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, D_param) tensor.
            noise: (BatchSize, d) tensor of standard normal noise.
        """
        batch_size = params.shape[0]
        
        if self.sigma_fixed is not None:
            # Fixed Sigma
            mu = params
            sigma = self.sigma_fixed
        else:
            # Learn Sigma
            mu = params[:, :-1]
            raw_sigma = params[:, -1:]
            
            if self.parameterization == 'softplus':
                sigma = torch.nn.functional.softplus(raw_sigma)
            elif self.parameterization == 'linear':
                sigma = raw_sigma
                
                # Check for negative sigma in linear mode
                # Since gradients are computed during backward, we are in forward pass here.
                # If optimization pushed sigma to negative, we should detect it.
                if (sigma < 0).any():
                    print("FAILURE: Negative sigma detected in linear parameterization.")
                    raise ValueError("Negative sigma detected.")
            else:
                raise ValueError(f"Unknown parameterization: {self.parameterization}")
            
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        return mu + noise * sigma

class GandKSimulator(torch.nn.Module):
    """
    Differentiable g-and-k distribution simulator.
    A common benchmark in SBI.
    
    Model:
    z ~ N(0, 1)
    x(z) = A + B * [1 + c * tanh(g*z/2)] * (1 + z^2)^k * z
    
    Note: Standard formulation uses c=0.8.
    
    Parameters: theta = [A, B, g, k]
    Constraints: B > 0, k > -0.5 (usually).
    
    Output:
    Returns d order statistics (sorted samples) to make the summary characteristic.
    """
    def __init__(self, d, c=0.8, use_log_k=True):
        super().__init__()
        self.d = d
        self.c = c
        self.use_log_k = use_log_k
        
    def sample_noise(self, n, device):
        return torch.randn(n, self.d, device=device)
        
    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 4) [A, B, g, log_k or k]
            noise: (BatchSize, d) standardized gaussian noise
            
        Returns:
            X: (BatchSize, d) sorted samples (order statistics)
        """
        # Unpack parameters
        A = params[:, 0:1]
        B = params[:, 1:2]
        g = params[:, 2:3]
        k_param = params[:, 3:4]
        
        k = torch.exp(k_param) if self.use_log_k else k_param
        
        # Generate noise
        if noise is None:
            noise = self.sample_noise(params.shape[0], params.device)
            
        z = noise
        
        # g-and-k transformation
        # term1 = 1 + c * (1 - exp(-g*z)) / (1 + exp(-g*z))
        # This is equivalent to 1 + c * tanh(g*z/2)
        term1 = 1 + self.c * torch.tanh(g * z / 2.0)
        
        term2 = (1 + z**2)**k
        
        x = A + B * term1 * term2 * z 
        
        return x
