
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from src.eval_metrics import *

from modules.transformer import TransformerEncoder

class custom_autograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, theta):
        ctx.save_for_backward(input, theta)
        return input / (1 - theta.item())

    @staticmethod
    def backward(ctx, grad_output):
        input, theta = ctx.saved_tensors
        input_grad = 1 / (1 - theta.item()) * grad_output.clone()
        return input_grad, None


class Modality_drop:
    def __init__(self, dim_list, p_exe=0.7, device='cuda'):
        self.dim_list = dim_list  # List of feature_dimensions for each modality
        self.p_exe = p_exe        # Probability of applying modality dropout
        self.device = device      # Device for computation

    def execute_drop(self, fead_list, q):
        seq_len = fead_list[0].shape[0]
        batch_size = fead_list[0].shape[1]
        feature_dim = fead_list[0].shape[2]
        # print(f"At Execute_drop, seq_len: {seq_len}, batch_size: {batch_size}, feature_dim: {feature_dim}")
        # print(f"q values are: text: {self.q[0]}, audio: {self.q[1]}, vision: {self.q[2]}")

        exe_drop = torch.tensor(np.random.rand(1)).to(self.device) >= (1 - self.p_exe)
        if not exe_drop:
            return fead_list, torch.ones([batch_size], dtype=torch.int32, device=self.device)

        num_mod = len(fead_list)
        d_sum = sum(self.dim_list)
        q_sum = sum(self.dim_list * q)
        theta = q_sum / d_sum

        # since q=[mod_1, mod_2, mod_3]. Its shape is [3,...]
        # We just drop batch_size, not seq_len (too complexity :v, cannot reshape both of them :b)
        mask = torch.distributions.Bernoulli(1 - torch.tensor(q)).sample([batch_size, 1]).permute(2, 1, 0).contiguous().reshape(num_mod, batch_size, -1).unsqueeze(-1).to(self.device) # [num_mod, batch_size, seq_len=1, dim=1] add(unsqueeze) one more shape for seq_len
        # print(f"Mask at initial step: {mask.shape}")
        concat_list = torch.stack(fead_list, dim=0).permute(0, 2, 1, 3) # [num_mod, batch_size, seq_len, dim]
        concat_list=torch.mul(concat_list,mask)# [num_mod, batch_size, seq_len, dim]
        concat_list = custom_autograd.apply(concat_list, theta)

        # print(f"concat_list shape: {concat_list.shape}")

        # mask = mask.squeeze(-1).permute(1, 2, 0) #[batch_size, seq_len, num_mod]
        mask = torch.transpose(mask, 0,1).squeeze(-1).squeeze(-1) #[batch_size, num_mod]
        # print(f"Mask at transpose Step: {mask.shape}")
        update_flag = torch.sum(mask, dim=-1) > 0 #[!=0_batch_size] !0 : keeps features.
        
        # print(f"update_flag shape at execute_drop: {update_flag.shape}")

        #  masked_select selects features (!=0 batch_size) , and the result is a 1D tensor containing all the selected elements
        cleaned_fea = torch.masked_select(concat_list, update_flag.unsqueeze(-1).unsqueeze(-1)) #[total]
        # print(f"cleaned_feature masked_select shape: {cleaned_fea.shape}")
        
        cleaned_fea = cleaned_fea.reshape(num_mod, -1, seq_len, feature_dim) #[num_mod, new_batch_size, seq_len, dim]
        # print(f"cleaned_fea(after reshape) final size: {cleaned_fea.shape}")
        
        cleaned_fea = torch.chunk(cleaned_fea, num_mod, dim=0) # Converted into a list of num_mod tensors: [1, new_batch_size, seq_len, dim]
        cleaned_fea = [_.squeeze(0).permute(1, 0, 2) for _ in cleaned_fea]# a list of tensor; each has format:[new_batch_size, seq_len, dim]=>permute:[seq_len, new_batch_size, dim]
        # print(f"Final cleaned_feature shape with new batch_size modalities 1 text: {cleaned_fea[0].shape}")
        # print(f"Final cleaned_feature shape with new batch_size modalities 2 audio: {cleaned_fea[1].shape}")
        # print(f"Final cleaned_feature shape with new batch_size modalities 3 vision: {cleaned_fea[2].shape}")

        return cleaned_fea, update_flag


def calcu_q(performance_1, performance_2, performance_3, q_base, fix_lambda):
    performance_1 = torch.tensor(performance_1, dtype=torch.float32)
    performance_2 = torch.tensor(performance_2, dtype=torch.float32)
    performance_3 = torch.tensor(performance_3, dtype=torch.float32)
    q = torch.tensor([0.0, 0.0, 0.0])
    relu = nn.ReLU(inplace=True)
    ratio_1 = torch.tanh(relu(1/2*(performance_1 / performance_2 + performance_1 / performance_3) - 1))
    ratio_2 = torch.tanh(relu(1/2*(performance_2 / performance_1 + performance_2 / performance_3) - 1))
    ratio_3 = torch.tanh(relu(1/2*(performance_3 / performance_1 + performance_3 / performance_2) - 1))

    lamda = fix_lambda

    q[0] = q_base * (1 + lamda * ratio_1) if ratio_1 > 0 else 0
    q[1] = q_base * (1 + lamda * ratio_2) if ratio_2 > 0 else 0
    q[2] = q_base * (1 + lamda * ratio_3) if ratio_3 > 0 else 0

    q = torch.clip(q, 0.0, 1.0)
    return q


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=5, layers=2,
                 relu_dropout=0.1, embed_dropout=0.3,
                 attn_dropout=0.25, res_dropout=0.1):
        super(Classifier, self).__init__()
        self.bone = TransformerEncoder(embed_dim=in_dim, num_heads=num_heads,
                                       layers=layers, attn_dropout=attn_dropout, res_dropout=res_dropout,
                                       relu_dropout=relu_dropout, embed_dropout=embed_dropout)

        self.proj1 = nn.Linear(in_dim, in_dim)
        self.out_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.bone(x)
        x = self.proj1(x[0])
        x = F.relu(self.proj1(x))
        x = self.out_layer(x)
        return x


class ClassifierGuided(nn.Module):
    def __init__(self, output_dim, num_mod, proj_dim=30, num_heads=5, layers=5,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, attn_dropout=0.25):
        super(ClassifierGuided, self).__init__()
        self.num_mod = num_mod
        self.classifiers = nn.ModuleList([
            Classifier(in_dim=proj_dim, out_dim=output_dim, layers=layers,
                       num_heads=num_heads, attn_dropout=attn_dropout, res_dropout=res_dropout,
                       relu_dropout=relu_dropout, embed_dropout=embed_dropout)
            for _ in range(self.num_mod)
        ])

    def cal_coeff(self, y, cls_res): #cls_res: A list containing prediction outputs of each single modality = modality_preds
        performance = []
        for r in cls_res: # For every modality's output predicted by one single modality (r)
            # Note r and label y are tensors
            # print("r shape: ", r.shape) #[64,1]
            # print("y shape: ", y.shape) #[64,1]
            acc = train_eval_senti(r, y)
            performance.append(acc)
        return performance

    def forward(self, x):
        self.cls_res = []
        for i in range(len(x)):
            self.cls_res.append(self.classifiers[i](x[i]))
        return self.cls_res


class MSAModel(nn.Module):
    def __init__(self, output_dim, orig_dim, proj_dim=30, num_heads=5, layers=5,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, out_dropout=0.1,
                 attn_dropout=0.25, cls_layers=2, q_base=0.5, p_exe=0.7, d=[40, 40, 40],
                 lam=0.5, cls_optim='Adam', cls_lr=5e-4, device='cuda'):
        super(MSAModel, self).__init__()

        self.output_dim = output_dim
        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.cls_layers = cls_layers
        self.q_base = q_base
        self.p_exe = p_exe
        self.d = d
        self.lam = lam
        self.cls_optim = cls_optim
        self.cls_lr = cls_lr
        self.device = device

        self.modality_drop = Modality_drop(dim_list=torch.tensor(self.d), p_exe=self.p_exe, device=self.device)

        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mod)
        ])

        self.classifiers_guide = ClassifierGuided(self.output_dim, self.num_mod, self.proj_dim,
                                                  self.num_heads, self.cls_layers,
                                                  self.relu_dropout, self.embed_dropout,
                                                  self.res_dropout, self.attn_dropout)
        self.cls_optimizer = getattr(optim, self.cls_optim)(self.classifiers_guide.parameters(), lr=self.cls_lr)

        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim=proj_dim, num_heads=self.num_heads,
                                layers=self.layers, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
                                relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout)
            for _ in range(self.num_mod)
        ])

        self.fusion = TransformerEncoder(
            embed_dim=proj_dim, num_heads=self.num_heads,
            layers=self.layers - 2, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
            relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout
        )

        self.proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

        self.cls_grad = []

    def forward(self, x, label, warm_up=1, use_grad=False):
        hs = []
        hs_detach = []
        performance = []
        modality_preds = []

        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp)
            hs_detach.append(h_tmp.clone().detach())

        modality_preds = self.classifiers_guide(hs_detach)# Raw score ouputs of each modality that are used as input for prediction.
        performance = self.classifiers_guide.cal_coeff(label, modality_preds)# Raw-detached score ouputs of each modality that are used as input for prediction.
        
        if use_grad: # If we train
            self.cls_optimizer.zero_grad()
            # modality_preds = self.classifiers_guide(hs_detach)# Raw score ouputs of each modality that are used as input for prediction.

            # Calculate + backward the loss with nn.L1Loss()
            loss_fn = nn.L1Loss()  # Initialize the loss function'
            mod_loss = loss_fn(modality_preds[0], label)  # Apply it to the first prediction and label
            for i in range(1, self.num_mod):
                mod_loss += loss_fn(modality_preds[i], label)
            mod_loss.backward()

            # Get the gradients of each modality single prediction
            for name, para in self.classifiers_guide.named_parameters():
                if 'out_layer.weight' in name:
                    self.cls_grad.append(para)

            # performance = self.classifiers_guide.cal_coeff(label, modality_preds)# Raw-detached score ouputs of each modality that are used as input for prediction.
        
            if warm_up==0:    # When epoch >10, we drop  features probability.
                self.q = calcu_q(performance[0], performance[1], performance[2], self.q_base, fix_lambda=self.lam)
                hs, update_flag = self.modality_drop.execute_drop(hs, self.q)

                last_hs = self.fusion(torch.cat(hs))[0]
                # print(f"last_hs fusion shape with new batch_size: {last_hs.shape}")
                last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
                last_hs_proj += last_hs
                output = self.out_layer(last_hs_proj)
                # print(f"Final output shape with new batch_size at MSAModel: {output.shape}")
                # print(f"Final update_flag shape with new batch_size at MSA model: {update_flag.shape}")

                return modality_preds[0], modality_preds[1], modality_preds[2], output, update_flag, performance[0], performance[1], performance[2], self.cls_grad, self.cls_optimizer
            
            else:            # For first epochs, we do not need drop features.
                last_hs = self.fusion(torch.cat(hs))[0]
                last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
                last_hs_proj += last_hs
                output = self.out_layer(last_hs_proj)
                # hs[0].shape[1] = batch_size
                return modality_preds[0], modality_preds[1], modality_preds[2], output, torch.ones([hs[0].shape[1]], dtype=torch.int32, device=self.device), performance[0], performance[1], performance[2], self.cls_grad, self.cls_optimizer

        else: # If we valid/test
            last_hs = self.fusion(torch.cat(hs))[0]
            last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
            last_hs_proj += last_hs
            output = self.out_layer(last_hs_proj)
            # hs[0].shape[1] = batch_size
            return modality_preds[0], modality_preds[1], modality_preds[2], output, torch.ones([hs[0].shape[1]], dtype=torch.int32, device=self.device), performance[0], performance[1], performance[2]
