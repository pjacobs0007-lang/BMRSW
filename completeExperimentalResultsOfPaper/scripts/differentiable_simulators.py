import torch

class MASimulatorTorch(torch.nn.Module):
    """
    Differentiable Moving Average MA(q) Simulator using PyTorch.
    Model: X_t = mu + epsilon_t + sum_{j=1}^q theta_j * epsilon_{t-j}
    where epsilon_t ~ N(0, sigma^2)
    """
    def __init__(self, q, d):
        super().__init__()
        self.q = q
        self.d = d

    def sample_noise(self, n, device):
        return torch.randn(n, self.d + self.q, device=device)

    def forward(self, params, noise=None):
        """
        Simulate a batch of time series.
        
        Args:
            params: (BatchSize, q+2) tensor. Columns: [mu, sigma, theta_1, ..., theta_q]
            noise: Optional (BatchSize, d+q) tensor of standard normal noise. 
                   If None, it is sampled (but this breaks reparameterization if not careful).
                   For SGDA, noise should be sampled outside and passed in.
                   
        Returns:
            X: (BatchSize, d) tensor
        """
        batch_size = params.shape[0]
        
        # Unpack parameters
        mu = params[:, 0:1]      # (Batch, 1)
        sigma = params[:, 1:2]   # (Batch, 1)
        thetas = params[:, 2:]   # (Batch, q)
        
        # Generate noise if not provided
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        # Scale noise: epsilon = sigma * z
        epsilon = noise * sigma
        
        # We want to compute X[:, t] for t = 0...d-1
        # X_t = mu + eps_{t+q} + sum_{j=1}^q theta_j * eps_{t+q-j}
        
        # Pre-allocate output
        X_cols = []
        
        for t in range(self.d):
            # Base index for epsilon corresponding to time t is t + q
            idx = t + self.q
            
            # Start with mu + epsilon_t
            val = mu + epsilon[:, idx:idx+1]
            
            # Add lag components
            for j in range(1, self.q + 1):
                # theta_{j-1} * epsilon_{t-j}
                # thetas[:, j-1] is the j-th lag coefficient
                # epsilon index is idx - j
                val = val + thetas[:, j-1:j] * epsilon[:, idx-j:idx-j+1]
            
            X_cols.append(val)
            
        X = torch.cat(X_cols, dim=1)
        return X

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
        
        # Return sorted samples as summary statistics
        # Sorting is differentiableish (gradients flow through the selected indices)
        x_sorted, _ = torch.sort(x, dim=1)
        
        return x_sorted

class SimpleRegressionSimulator(torch.nn.Module):
    """
    Differentiable Simple Linear Regression Simulator.
    Model: X ~ Unif(-1, 1)
           Y = beta_0 + beta_1 * X + sigma * epsilon, where epsilon ~ N(0, 1)
    
    Data is 2D: (Y, X) as per user's h(z1, z2) = (z1 sigma + z2 beta1 + beta0, z2)
    where z1 ~ N(0, 1) and z2 ~ Unif(-1, 1).
    
    Parameters: theta = [beta_0, beta_1, raw_sigma]
    sigma = softplus(raw_sigma)
    """
    def __init__(self, parameterization='softplus'):
        super().__init__()
        self.parameterization = parameterization

    def sample_noise(self, n, device):
        """
        Returns (n, 2) noise: col 0 is N(0,1), col 1 is Unif(-1, 1)
        """
        z1 = torch.randn(n, 1, device=device)
        z2 = torch.rand(n, 1, device=device) * 2 - 1
        return torch.cat([z1, z2], dim=1)

    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 3) [beta_0, beta_1, raw_sigma]
            noise: (BatchSize, 2) where col 0 is N(0,1) and col 1 is Unif(-1,1)
        """
        batch_size = params.shape[0]
        beta0 = params[:, 0:1]
        beta1 = params[:, 1:2]
        raw_sigma = params[:, 2:3]
        
        if self.parameterization == 'softplus':
            sigma = torch.nn.functional.softplus(raw_sigma)
        else:
            sigma = raw_sigma
            
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        z1 = noise[:, 0:1] # Normal(0, 1)
        z2 = noise[:, 1:2] # Unif(-1, 1)
        
        y = beta0 + beta1 * z2 + sigma * z1
        x = z2
        
        # Return (Y, X) as per user's H(z1, z2) = (z1 sigma + z2 beta1 + beta0, z2)
        return torch.cat([y, x], dim=1)

class MeanFreqSimulator(torch.nn.Module):
    """
    Differentiable Mean-Frequency Simulator.
    Model: X ~ Unif(-1, 1)
           Y | X ~ N(mu * X + 0.1 * sin(omega * X), 1)
    
    Data is 2D: (Y, X)
    Parameters: theta = [mu, omega]
    """
    def __init__(self, sd=1.0):
        super().__init__()
        self.sd = sd

    def sample_noise(self, n, device):
        """
        Returns (n, 2) noise: col 0 is N(0,1), col 1 is Unif(-1, 1)
        """
        z1 = torch.randn(n, 1, device=device)
        z2 = torch.rand(n, 1, device=device) * 2 - 1
        return torch.cat([z1, z2], dim=1)

    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 2) [mu, omega]
            noise: (BatchSize, 2) where col 0 is N(0,1) and col 1 is Unif(-1,1)
        """
        batch_size = params.shape[0]
        mu = params[:, 0:1]
        omega = params[:, 1:2]
        
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        z1 = noise[:, 0:1] # Normal(0, 1)
        x = noise[:, 1:2]  # Unif(-1, 1)
        
        y = mu * x + 0.1 * torch.sin(omega * x) + self.sd * z1
        
        return torch.cat([y, x], dim=1)

class InterceptFreqSimulator(torch.nn.Module):
    """
    Differentiable Intercept-Frequency Simulator.
    Model: X ~ Unif(-1, 1)
           Y | X ~ N(mu + X + 0.1 * sin(omega * X), sd^2)
    
    Data is 2D: (Y, X)
    Parameters: theta = [mu, omega]
    """
    def __init__(self, sd=1.0):
        super().__init__()
        self.sd = sd

    def sample_noise(self, n, device):
        """
        Returns (n, 2) noise: col 0 is N(0,1), col 1 is Unif(-1, 1)
        """
        z1 = torch.randn(n, 1, device=device)
        z2 = torch.rand(n, 1, device=device) * 2 - 1
        return torch.cat([z1, z2], dim=1)

    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 2) [mu, omega]
            noise: (BatchSize, 2) where col 0 is N(0,1) and col 1 is Unif(-1,1)
        """
        batch_size = params.shape[0]
        mu = params[:, 0:1]
        omega = params[:, 1:2]
        
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        z1 = noise[:, 0:1] # Normal(0, 1)
        x = noise[:, 1:2]  # Unif(-1, 1)
        
        y = mu + x + 0.1 * torch.sin(omega * x) + self.sd * z1
        
        return torch.cat([y, x], dim=1)

class UnivariateMeanFreqSimulator(torch.nn.Module):
    """
    Differentiable Univariate Mean-Frequency Simulator.
    Y = X + 0.1 * sin(omega * (X - mu) / sd) where X ~ Normal(mu, sd^2)
    This simplifies to Y = mu + sd * z + 0.1 * sin(omega * z) where z ~ Normal(0, 1).
    
    Data is 1D: Y
    Parameters: theta = [mu, omega]
    """
    def __init__(self, sd=1.0):
        super().__init__()
        self.sd = sd

    def sample_noise(self, n, device):
        """
        Returns (n, 1) Standard Normal noise.
        """
        return torch.randn(n, 1, device=device)

    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 2) [mu, omega]
            noise: (BatchSize, 1) N(0,1) noise
        """
        batch_size = params.shape[0]
        mu = params[:, 0:1]
        omega = params[:, 1:2]
        
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        z = noise # (B, 1) standard normal
        
        y = mu + self.sd * z + 0.1 * torch.sin(omega * z)
        
        return y

class DiscretizedNormalSimulator(torch.nn.Module):
    """
    Non-differentiable Discretized Normal Simulator.
    Generates a random variable X ~ Normal(mu, sd^2), and then transforms it as floor(X / eps) * eps.
    
    Data is 1D: Y
    Parameters: theta = [mu, eps]
    """
    def __init__(self, sd=1.0):
        super().__init__()
        self.sd = sd

    def sample_noise(self, n, device):
        """
        Returns (n, 1) Standard Normal noise.
        """
        return torch.randn(n, 1, device=device)

    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 2) [mu, eps]
            noise: (BatchSize, 1) N(0,1) noise
        """
        batch_size = params.shape[0]
        mu = params[:, 0:1]
        eps = params[:, 1:2]
        
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        z = noise # (B, 1) standard normal
        
        # X ~ Normal(mu, sd^2)
        x = mu + self.sd * z
        
        # Transform: floor(X / eps) * eps
        y = torch.floor(x / eps) * eps
        
        return y

class SmoothStepSimulator(torch.nn.Module):
    """
    Differentiable Smooth Step Simulator.
    Generates a random variable X ~ Normal(mu, sd^2), and applies a mathematically rigorous 
    differentiable smooth step approximation using piecewise shifted sigmoids:
    
    i(X) = floor(X / eps + 0.5)
    Y = (i(X) - 1) * eps + eps / (1 + exp(-k * (X - i(X) * eps)))
    
    Data is 1D: Y
    Parameters: theta = [mu, rho], where eps = softplus(rho)
    """
    def __init__(self, sd=1.0, k=50.0):
        super().__init__()
        self.sd = sd
        self.k = k

    def sample_noise(self, n, device):
        """
        Returns (n, 1) Standard Normal noise.
        """
        return torch.randn(n, 1, device=device)

    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 2) [mu, eps]
            noise: (BatchSize, 1) N(0,1) noise
        """
        batch_size = params.shape[0]
        mu = params[:, 0:1]
        rho = params[:, 1:2]
        eps = torch.nn.functional.softplus(rho) + 1e-5
        
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        z = noise # (B, 1) standard normal
        
        # X ~ Normal(mu, sd^2)
        x = mu + self.sd * z
        
        # Differentiable continuous piecewise sigmoid transformation
        # We use a Straight-Through Estimator (STE) for the floor index to ensure 
        # the optimizer knows that shrinking eps inversely increases the step index, 
        # preventing it from blindly driving rho -> -inf thinking y is linear in eps.
        i_soft = x / eps + 0.5
        i_hard = torch.floor(i_soft)
        i = i_hard.detach() + i_soft - i_soft.detach()
        
        # Use torch.clamp rather than hard numpy.clip to remain differentiable while preventing exp overflow
        # Max of float32 exp is ~88.7. Setting clamp to [-80, 80] ensures forward and BACKWARD passes don't yield inf/nans.
        t = torch.clamp(-self.k * (x - i * eps), min=-80.0, max=80.0)
        
        # Compute smooth steps
        y = (i - 1.0) * eps + eps / (1.0 + torch.exp(t))
        
        return y

class ProjectedSinWaveSimulator(torch.nn.Module):
    """
    Differentiable Projected Sine Wave Simulator.
    Model: X ~ Unif(-1, 1)
           Y = mu + sin(omega * X)
    
    Data is 2D: (Y, X)
    Parameters: theta = [mu, omega]
    """
    def __init__(self):
        super().__init__()

    def sample_noise(self, n, device):
        """
        Returns (n, 1) Unif(-1, 1) noise.
        """
        return torch.rand(n, 1, device=device) * 2.0 - 1.0

    def forward(self, params, noise=None):
        """
        Args:
            params: (BatchSize, 2) [mu, omega]
            noise: (BatchSize, 1) Unif(-1,1) noise
        """
        batch_size = params.shape[0]
        mu = params[:, 0:1]
        omega = params[:, 1:2]
        
        if noise is None:
            noise = self.sample_noise(batch_size, params.device)
            
        x = noise # (B, 1) Unif(-1, 1)
        
        y = mu + torch.sin(omega * x)
        
        return torch.cat([y, x], dim=1)
