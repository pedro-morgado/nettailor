import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import collections
from .student_resnet import NetTailorBlock, BasicProxy, conv, conv1x1, conv3x3
from proj_utils import AverageMeter

__all__ = [ 
    'wide_resnet26',
]

PRETRAINED_PATH = {
    'wide_resnet26': 'checkpoints/wide_resnet26.pth.tar'
}


class BasicWideResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicWideResBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Conv1
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, use_bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv2
        self.conv2 = conv3x3(out_channels, out_channels, use_bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ConvRes
        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(2)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))

        res = x
        if self.stride > 1:
            res = self.avgpool(x)
        if self.in_channels != self.out_channels:
            res = torch.cat((res, res*0),1)

        return F.relu(y + res, inplace=True)

    def num_params(self):
        n_params = 0.
        for p in self.parameters():
            n_params += np.prod(p.size())
        return n_params

    def FLOPS(self, in_shape):
        spatial_dim = float(in_shape[1] * in_shape[2]) / (float(self.stride)**2.)
        flops =  spatial_dim * self.in_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        flops += spatial_dim * self.out_channels * self.out_channels * 9
        flops += spatial_dim * self.out_channels * 2
        return flops


class NetTailor(nn.Module):
    def __init__(self, backbone, num_classes, max_skip=1):
        super(NetTailor, self).__init__()

        if backbone == 'wide_resnet26':
            universal_block = BasicWideResBlock
            proxy_block = BasicProxy
            num_layers = [4,4,4]
        else:
            raise ValueError('Unsuported backbone.')

        self.universal_block = universal_block
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_skip = max_skip

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Track tensor shapes of last max_skip layers
        cur_shape = (32, 64, 64)
        skip_shapes = collections.deque(maxlen=self.max_skip)
        skip_shapes.appendleft(cur_shape)

        # Track FLOPS per block
        self.flops =[[float(cur_shape[1] * cur_shape[2]) * (3 * 32 * 7**2 + 32 * 2)]]

        layers = []
        for b in range(len(num_layers)):
            # Filter progression: 64 -> 128 -> 256
            num_channels = 64*(2**b)

            for bb in range(num_layers[b]):
                # Add NetTailor layer consisting of 1 universal block and max_skip task-specific blocks
                stride = 2 if bb == 0 else 1
                layers.append(NetTailorBlock(universal_block, proxy_block, num_channels, stride, skip_shapes))

                # Track tensor shapes of last max_skip layers
                cur_shape = (num_channels*universal_block.expansion, cur_shape[1]//stride, cur_shape[2]//stride)
                skip_shapes.appendleft(cur_shape)

                # Track FLOPS per block
                self.flops.append(layers[-1].FLOPS())

        self.layers = nn.ModuleList(layers)

        # Add final classifier
        self.ends_bn = nn.BatchNorm2d(256)
        self.classifier = nn.Linear(cur_shape[0], num_classes)
        self.flops.append([cur_shape[0] * num_classes + num_classes])
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load universal blocks
        self.load_universal_blocks(backbone)

        # Freeze universal layers
        self.freeze_backbone()

    def forward(self, x):
        ends = collections.deque(maxlen=self.max_skip)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        ends.appendleft(x)
        proxies = []
        for l in range(sum(self.num_layers)):
            p = self.layers[l]([e.detach() if e is not None else None for e in ends])
            proxies.append(p)
            x = self.layers[l](ends)
            ends.appendleft(x)
        x = self.ends_bn(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.classifier(x)
        return (x, proxies)

    def load_universal_blocks(self, base_arch):
        fn = PRETRAINED_PATH[base_arch]
        checkpoint = torch.load(fn)['state_dict']
        state = self.state_dict()

        for k_st in state:
            if 'proxies' in k_st or 'alphas' in k_st or 'classifier' in k_st or 'ends_bn' in k_st:
                continue

            k_tkn = k_st.split('.')
            if k_tkn[-1] == 'num_batches_tracked':
                continue
            if k_tkn[0] in ['conv1', 'bn1']:
                k_ckp = k_st
            elif k_tkn[0] == 'layers':
                k_ckp = k_st.replace('.main', '')
            else:
                raise ValueError

            assert all([s1 == s2 for s1, s2 in zip(checkpoint[k_ckp].size(), state[k_st].size())])
            state[k_st] = checkpoint[k_ckp]
        self.load_state_dict(state)

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if 'proxies' not in name and 'alphas' not in name and 'classifier' not in name and 'ends_bn' not in name:
                param.requires_grad = False

    def get_keep_flags(self):
        return [layer.keep_flag for layer in self.layers]

    def load_keep_flags(self, keep_flags):
        assert len(keep_flags)==len(self.layers)
        for keep, layer in zip(keep_flags, self.layers):
            assert len(keep)==len(layer.keep_flag)
            layer.keep_flag = keep[:]

    def expected_complexity(self):
        alphas_list, complexity_list = [], []
        global_flops = float(sum([f[0] for f in self.flops[:-1]]))
        for layer, l_flops in zip(self.layers, self.flops[1:-1]):
            alphas_list.append([a for a in layer.alphas()])
            complexity_list.append([f/global_flops for f in l_flops])
        
        # Probability that incoming connections are clear
        incoming_alphas = [torch.prod(1.-torch.stack(alphas)) for alphas in alphas_list]

        # Probability that outgoing connections are clear
        outgoing_alphas = [[] for i in range(len(alphas_list))]
        for i, alphas in enumerate(alphas_list):
            for j in range(len(alphas)):
                src = i if j <= 1 else i-(j-1)
                outgoing_alphas[src].append(alphas[j])

        for i in range(len(outgoing_alphas)):
            outgoing_alphas[i] = torch.prod(1.-torch.stack(outgoing_alphas[i]))

        # Complexity computation
        C = 0.
        for i, (alphas, complixities) in enumerate(zip(alphas_list, complexity_list)):
            for j in range(len(alphas)):
                c = complixities[j]
                p = alphas[j]
                p_in = torch.tensor(1., requires_grad=False).to(p.device) if i == 0 else incoming_alphas[i-1]
                p_out = torch.tensor(1., requires_grad=False).to(p.device) if i == len(outgoing_alphas)-1 else outgoing_alphas[i+1]
                c_layer = c * (p - p_in - p_out)
                C += c_layer
        return C

    def threshold_alphas(self, num_global=None, thr_global=None, percent_global=None, num_proxies=None, thr_proxies=None, percent_proxies=None, only_top=False):
        assert sum([num_global is not None, thr_global is not None, percent_global is not None]) <= 1
        assert sum([num_proxies is not None, thr_proxies is not None, percent_proxies is not None]) <= 1

        alphas_proxies, alphas_layer, keep = [], [], []
        for layer in self.layers:
            aa = layer.alphas().data.cpu().numpy()
            alphas_layer.append(aa[0])
            alphas_proxies.append(aa[1:])
            keep.append([1.]*aa.size)

        # Remove global layers
        alphas_layer = np.array(alphas_layer)
        if num_global is not None:
            to_rm = np.argsort(alphas_layer)[:num_global]
        elif percent_global is not None:
            to_rm = np.argsort(alphas_layer)[:int(len(alphas_layer)*percent_global)]
        elif thr_global is not None:
            to_rm = [i for i, aa in enumerate(alphas_layer) if aa <= thr_global]
        for rm_idx in to_rm:
            keep[rm_idx][0] = 0.

        # Remove task-specific layers
        meta_proxies = [(i, j) for i in range(len(alphas_proxies)) for j in range(alphas_proxies[i].size)]
        alphas_proxies = np.concatenate(alphas_proxies)
        to_rm = []
        if num_proxies is not None:
            to_rm.extend(np.argsort(alphas_proxies)[:num_proxies].tolist())
        elif percent_proxies is not None:
            to_rm.extend(np.argsort(alphas_proxies)[:int(len(alphas_proxies)*percent_proxies)].tolist())
        elif thr_proxies is not None:
            to_rm.extend([i for i, aa in enumerate(alphas_proxies) if aa <= thr_proxies])
        
        if only_top:
            for i in range(len(self.layers)):
                adp_a = [aa for ii, (mm, aa) in enumerate(zip(meta_proxies, alphas_proxies)) if mm[0]==i]
                adp_i = [ii for ii, (mm, aa) in enumerate(zip(meta_proxies, alphas_proxies)) if mm[0]==i]
                to_rm.extend([ii for ii, aa in zip(adp_i, adp_a) if aa/max(adp_a)<0.5])

        for rm_idx in to_rm:
            layer_idx = meta_proxies[rm_idx][0]
            proxies_idx = meta_proxies[rm_idx][1]+1
            keep[layer_idx][proxies_idx] = 0.

        # Update keep variables
        for layer, k in zip(self.layers, keep):
            layer.keep_flag = k[:]

    def global_params(self):
        global_params = [sum([np.prod(p.size()) for p in self.conv1.parameters()])+sum([np.prod(p.size()) for p in self.bn1.parameters()])]
        for layer in self.layers:
            if layer.keep_flag[0] > 0:
                global_params.append(layer.num_params()[0])
            else:
                global_params.append(0)
        return global_params

    def task_params(self):
        task_params = []
        for layer in self.layers:
            task_params.append([c*float(k) for c, k in zip(layer.num_params()[1:], layer.keep_flag[1:])])
        task_params.append([self.universal_block.expansion * 512 * self.num_classes + self.num_classes])
        return task_params

    def alphas_and_complexities(self):
        alphas = "\n{}   Alphas (Proxy distances, Num params, FLOPS)   {}\n".format("="*30, "="*30)
        for layer, l_flops in zip(self.layers, self.flops[1:-1]):
            pp = layer.num_params()
            ff = l_flops
            aa = layer.alphas().data.cpu().numpy()
            kk = layer.keep_flag
            for a, k, p, f in zip(aa, kk, pp, ff):
                alphas += '{} {:.3f} ({:.3f}, {:.3f}) |  '.format('X' if k else ' ', a, p/10**6, f/10**6)
            alphas += '\n'
        return alphas

    def stats(self):
        network_stats = self.alphas_and_complexities()

        network_stats += "\n{}   Parameters   {}\n".format("="*30, "="*30)
        global_params, task_params = 0, 0

        global_params += self.global_params()[0]
        network_stats += "{}\n".format(self.global_params()[0]/10.**6)
        for gp, tp in zip(self.global_params()[1:], self.task_params()[:-1]):
            global_params += gp
            task_params += sum(tp) if isinstance(tp, list) else tp
            p = [gp] + tp if isinstance(tp, list) else [tp]
            network_stats += "{}\n".format(''.join(["{:15}".format(str(pp/10.**6)) for pp in p]))
        task_params += self.task_params()[-1][0]
        network_stats += "{}\n".format(self.task_params()[-1][0]/10.**6)

        network_stats +=  "\nGlobal Parameters {}\n".format(global_params/10.**6)
        network_stats +=  "Task Parameters {}\n".format(task_params/10.**6)
        network_stats +=  "Total {}\n".format(global_params/10.**6+task_params/10.**6)

        network_stats += "\n{}   FLOPS   {}\n".format("="*30, "="*30)
        total_flops = self.flops[0][0]
        network_stats += "{}\n".format(self.flops[0][0]/10.**9)
        for l, ff in zip(self.layers,self.flops[1:-1]):
            total_flops += sum([f if k else 0 for f, k in zip(ff, l.keep_flag)])
            network_stats += "{}\n".format(''.join(["{:15}".format(str(f/10.**9 if k else 0.)) for f, k in zip(ff, l.keep_flag)]))
        total_flops += self.flops[-1][0]
        network_stats += "{}\n".format(self.flops[-1][0]/10.**9)
        network_stats +=  "\nTotal {}\n".format(total_flops/10.**9)

        return network_stats


def create_model(backbone, num_classes, max_skip):
    return NetTailor(backbone, num_classes, max_skip=max_skip)

##########################################################################################

if __name__ == '__main__':
    device = torch.device("cpu")
    model = create_model('wide_resnet26', 10, max_skip=3)
    model.to(device)

    print('\n'+'='*30+'  Model  '+'='*30)
    print(model)

    print('\n'+'='*30+'  Parameters  '+'='*30)
    for n, p in model.named_parameters():
        print("{:50} | {:10} | {:30} | {:20} | {}".format(
            n, 'Trainable' if p.requires_grad else 'Frozen' , 
            str(p.size()), str(np.prod(p.size())), str(p.type()))
        )

    print(model.stats())
    print(model.expected_complexity())
    
    inp = torch.rand(2, 3, 64, 64)
    inp = inp.to(device)
    print('Input:', inp.shape)
    out, proxies = model(inp)
    for p in proxies:
        print('Proxy:', p.shape)
    print('Output', out.shape)
    labels = torch.zeros(2,).type(torch.LongTensor)
    loss = nn.CrossEntropyLoss()(out, labels)
    loss.backward()
