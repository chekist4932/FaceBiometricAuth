import torch
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# To assist you in designing the feature extractor you may want to print out
# the available nodes for resnet50.
m = resnet50()
train_nodes, eval_nodes = get_graph_node_names(resnet50())

# The lists returned, are the names of all the graph nodes (in order of
# execution) for the input model traced in train mode and in eval mode
# respectively. You'll find that `train_nodes` and `eval_nodes` are the same
# for this example. But if the model contains control flow that's dependent
# on the training mode, they may be different.

# To specify the nodes you want to extract, you could select the final node
# that appears in each of the main layers:
return_nodes = {
    # node_name: user-specified key for output dict
    'layer1.2.relu_2': 'layer1',
    'layer2.3.relu_2': 'layer2',
    'layer3.5.relu_2': 'layer3',
    'layer4.2.relu_2': 'layer4',
}

# But `create_feature_extractor` can also accept truncated node specifications
# like "layer1", as it will just pick the last node that's a descendent of
# of the specification. (Tip: be careful with this, especially when a layer
# has multiple outputs. It's not always guaranteed that the last operation
# performed is the one that corresponds to the output you desire. You should
# consult the source code for the input model to confirm.)
return_nodes = {
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4',
}

# Now you can build the feature extractor. This returns a module whose forward
# method returns a dictionary like:
# {
#     'layer1': output of layer 1,
#     'layer2': output of layer 2,
#     'layer3': output of layer 3,
#     'layer4': output of layer 4,
# }
create_feature_extractor(m, return_nodes=return_nodes)


# Let's put all that together to wrap resnet50 with MaskRCNN

# MaskRCNN requires a backbone with an attached FPN
class Resnet50WithFPN(torch.nn.Module):
    def __init__(self):
        super(Resnet50WithFPN, self).__init__()
        # Get a resnet50 backbone
        m = resnet50()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


# Now we can build our model!
model = MaskRCNN(Resnet50WithFPN(), num_classes=91).eval()
